// vx_raster - from-scratch Vulkan compute 3D Gaussian-Splatting rasterizer for YoNoSplat.
// image -> encoder(GPU) -> Gaussians -> THIS -> rendered view. Two custom compute kernels do the
// heavy lifting on the GPU: raster_preprocess (project mean+covariance to a 2D conic per gaussian)
// and raster_composite (per-tile front-to-back alpha compositing). Tile binning + the depth sort
// run on the host between them (cheap; a GPU radix sort is the next step). Matches the gsplat
// 'classic' math in scripts/yonosplat/ref_rasterizer.py (the numerical golden).
//
//   vx_raster <gaussians_dir> <extr.bin> <intr.bin> <out_dir> [--view N] [--num N] [--hw H W]
// gaussians_dir holds means.bin/covariances.bin/harmonics.bin/opacities.bin (fp32). extr = [V,4,4]
// camera-to-world, intr = [V,3,3] normalized. Writes <out_dir>/vk_view<N>.{u8,f32}.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#if defined(VXRT_ENABLE_VULKAN)
#include "backends/vulkan/vk_buffer.h"
#include "backends/vulkan/vk_command.h"
#include "backends/vulkan/vk_pipeline.h"

using namespace vx;
static const float C0 = 0.28209479177387814f;

static std::vector<float> readBin(const std::string& p, size_t n) {
  std::vector<float> v(n, 0.f);
  std::ifstream f(p, std::ios::binary);
  if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); }
  else f.read(reinterpret_cast<char*>(v.data()), n * 4);
  return v;
}
static const char* arg(int c, char** v, const char* k, const char* d) {
  for (int i = 1; i < c - 1; ++i) if (!strcmp(v[i], k)) return v[i + 1];
  return d;
}

int main(int argc, char** argv) {
  if (argc < 5) { printf("usage: vx_raster <gdir> <extr.bin> <intr.bin> <outdir> [--view N] [--num N] [--hw H W]\n"); return 1; }
  std::string gdir = argv[1], extp = argv[2], intp = argv[3], outdir = argv[4];
  int view = atoi(arg(argc, argv, "--view", "0"));
  int N = atoi(arg(argc, argv, "--num", "100352"));
  int H = 224, W = 224;
  for (int i = 1; i < argc - 2; ++i) if (!strcmp(argv[i], "--hw")) { H = atoi(argv[i+1]); W = atoi(argv[i+2]); }
  float NEAR = 0.2f;

  // ---- load gaussians ----
  std::vector<float> means = readBin(gdir + "/means.bin", (size_t)N * 3);
  std::vector<float> covs = readBin(gdir + "/covariances.bin", (size_t)N * 9);
  std::vector<float> harm = readBin(gdir + "/harmonics.bin", (size_t)N * 3);  // [N][3][1] -> [N][3]
  std::vector<float> opac = readBin(gdir + "/opacities.bin", (size_t)N);
  std::vector<float> cols((size_t)N * 3);
  for (size_t i = 0; i < (size_t)N * 3; ++i) cols[i] = std::max(0.f, C0 * harm[i] + 0.5f);

  // ---- camera: w2c = rigid inverse of c2w, K in pixels ----
  std::vector<float> extr = readBin(extp, (size_t)(view + 1) * 16);
  std::vector<float> intr = readBin(intp, (size_t)(view + 1) * 9);
  const float* c2w = &extr[(size_t)view * 16];
  const float* K = &intr[(size_t)view * 9];
  float Rt[9], tp[3];
  for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) Rt[r*3+c] = c2w[c*4 + r];  // R^T
  for (int r = 0; r < 3; ++r) tp[r] = -(Rt[r*3]*c2w[3] + Rt[r*3+1]*c2w[7] + Rt[r*3+2]*c2w[11]);
  float fx = K[0]*W, fy = K[4]*H, cx = K[2]*W, cy = K[5]*H;

  vk::VulkanContext ctx;
  if (!ctx.initialized()) { fprintf(stderr, "no Vulkan\n"); return 1; }
  vk::CommandRunner runner(ctx);

  // ---- preprocess on GPU ----
  vk::Buffer mB(ctx, means.size()*4), cvB(ctx, covs.size()*4), clB(ctx, cols.size()*4), opB(ctx, opac.size()*4);
  vk::Buffer gdB(ctx, (size_t)N*12*4, vk::MemPref::kReadback);
  mB.upload(means.data(), means.size()*4); cvB.upload(covs.data(), covs.size()*4);
  clB.upload(cols.data(), cols.size()*4); opB.upload(opac.data(), opac.size()*4);
  struct PrePC { int32_t dims[4]; float cam[4]; float cam2[4]; float r0[4], r1[4], r2[4]; } ppc{};
  ppc.dims[0]=N; ppc.dims[1]=H; ppc.dims[2]=W;
  ppc.cam[0]=NEAR; ppc.cam[1]=fx; ppc.cam[2]=fy; ppc.cam[3]=cx; ppc.cam2[0]=cy;
  for (int c=0;c<3;++c){ ppc.r0[c]=Rt[c]; ppc.r1[c]=Rt[3+c]; ppc.r2[c]=Rt[6+c]; }
  ppc.r0[3]=tp[0]; ppc.r1[3]=tp[1]; ppc.r2[3]=tp[2];
  {
    vk::ComputePipeline pre(ctx, "raster_preprocess", 5, sizeof(PrePC));
    VkCommandBuffer cmd = runner.allocate(); runner.begin(cmd);
    pre.dispatch(cmd, {mB.handle(), cvB.handle(), clB.handle(), opB.handle(), gdB.handle()}, &ppc,
                 sizeof(ppc), (uint32_t)((N + 63) / 64));
    runner.end(cmd); runner.submitAndWait(cmd);
    vkFreeCommandBuffers(ctx.device(), runner.pool(), 1, &cmd);
  }
  std::vector<float> gd((size_t)N*12);
  gdB.download(gd.data(), gd.size()*4);

  // ---- host: tile binning + per-tile depth sort ----
  int ntx = (W + 15) / 16, nty = (H + 15) / 16, nTiles = ntx * nty;
  std::vector<std::vector<int>> tileG(nTiles);
  for (int i = 0; i < N; ++i) {
    const float* g = &gd[(size_t)i*12];
    if (g[11] < 0.5f) continue;  // invalid
    float u = g[0], v = g[1], r = g[10];
    int tx0 = std::max(0, (int)((u - r) / 16)), tx1 = std::min(ntx - 1, (int)((u + r) / 16));
    int ty0 = std::max(0, (int)((v - r) / 16)), ty1 = std::min(nty - 1, (int)((v + r) / 16));
    for (int ty = ty0; ty <= ty1; ++ty)
      for (int tx = tx0; tx <= tx1; ++tx) tileG[ty * ntx + tx].push_back(i);
  }
  std::vector<int> sortedIdx, ranges(nTiles * 2);
  for (int t = 0; t < nTiles; ++t) {
    auto& lst = tileG[t];
    std::sort(lst.begin(), lst.end(), [&](int a, int b) { return gd[(size_t)a*12+2] < gd[(size_t)b*12+2]; });
    ranges[2*t] = (int)sortedIdx.size();
    ranges[2*t+1] = (int)lst.size();
    sortedIdx.insert(sortedIdx.end(), lst.begin(), lst.end());
  }
  if (sortedIdx.empty()) sortedIdx.push_back(0);
  printf("binned: %zu tile-gaussian entries over %d tiles\n", sortedIdx.size(), nTiles);

  // ---- composite on GPU ----
  vk::Buffer siB(ctx, sortedIdx.size()*4), rgB(ctx, ranges.size()*4);
  vk::Buffer imB(ctx, (size_t)H*W*3*4, vk::MemPref::kReadback);
  siB.upload(sortedIdx.data(), sortedIdx.size()*4);
  rgB.upload(ranges.data(), ranges.size()*4);
  struct CompPC { int32_t dims[4]; } cpc{}; cpc.dims[0]=H; cpc.dims[1]=W; cpc.dims[2]=ntx;
  {
    vk::ComputePipeline comp(ctx, "raster_composite", 4, sizeof(CompPC));
    VkCommandBuffer cmd = runner.allocate(); runner.begin(cmd);
    comp.dispatch(cmd, {gdB.handle(), siB.handle(), rgB.handle(), imB.handle()}, &cpc, sizeof(cpc),
                  (uint32_t)((W + 15) / 16), (uint32_t)((H + 15) / 16));
    runner.end(cmd); runner.submitAndWait(cmd);
    vkFreeCommandBuffers(ctx.device(), runner.pool(), 1, &cmd);
  }
  std::vector<float> img((size_t)H*W*3);
  imB.download(img.data(), img.size()*4);

  // ---- save ----
  std::string base = outdir + "/vk_view" + std::to_string(view);
  { std::ofstream f(base + ".f32", std::ios::binary); f.write(reinterpret_cast<char*>(img.data()), img.size()*4); }
  { std::vector<uint8_t> u8(img.size()); for (size_t i=0;i<img.size();++i) u8[i]=(uint8_t)(std::min(1.f,std::max(0.f,img[i]))*255+0.5f);
    std::ofstream f(base + ".u8", std::ios::binary); f.write(reinterpret_cast<char*>(u8.data()), u8.size()); }
  double s=0; for (float x : img) s += x;
  printf("rendered %dx%d  mean=%.4f  -> %s.{u8,f32}\n", W, H, s/img.size(), base.c_str());
  return 0;
}
#else
int main() { fprintf(stderr, "vx_raster needs Vulkan\n"); return 1; }
#endif
