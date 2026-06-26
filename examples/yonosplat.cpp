// vx_yonosplat - the full YoNoSplat 3D Gaussian-Splatting pipeline in one program, all on the GPU:
//   image + intrinsics --> encoder (vknn Vulkan) --> Gaussians --> Vulkan rasterizer --> rendered
//   view.
// The encoder runs as a normal vknn Session; its 6 Gaussian outputs feed the from-scratch Vulkan
// compute rasterizer (preprocess -> GPU tile-bin -> bitonic sort -> per-tile alpha compositing).
// The rendered view is written as a PPM. See scripts/yonosplat/ for how the encoder .vxm + inputs
// are made.
//
//   vx_yonosplat <encoder.vxm> <image.bin> <intrinsics.bin> <out.ppm> [--extr extr.bin] [--view N]
// image.bin = fp32 [1,V,3,224,224], intrinsics.bin = fp32 [1,V,3,3] (normalized). extr.bin
// (optional) = fp32 [V,4,4] camera-to-world (the encoder's predicted pose, dumpable via
// VKNN_DUMP_NAMES); identity if omitted. Renders view N.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "vknn/session.h"
#if defined(VKNN_ENABLE_VULKAN)
#include "backends/vulkan/vk_buffer.h"
#include "backends/vulkan/vk_command.h"
#include "backends/vulkan/vk_pipeline.h"
#endif

using namespace vknn;
static const float C0 = 0.28209479177387814f;

static std::vector<uint8_t> readFile(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f)
    return {};
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> v((size_t)n);
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}
static const char* opt(int c, char** v, const char* k, const char* d) {
  for (int i = 1; i < c - 1; ++i)
    if (!strcmp(v[i], k))
      return v[i + 1];
  return d;
}

int main(int argc, char** argv) {
#if !defined(VKNN_ENABLE_VULKAN)
  fprintf(stderr, "vx_yonosplat needs Vulkan\n");
  return 1;
#else
  if (argc < 5) {
    printf(
        "usage: vx_yonosplat <encoder.vxm> <image.bin> <intrinsics.bin> <out.ppm> [--extr "
        "extr.bin] [--view N]\n");
    return 1;
  }
  std::string enc = argv[1], imgp = argv[2], intp = argv[3], outp = argv[4];
  std::string extp = opt(argc, argv, "--extr", "");
  int view = atoi(opt(argc, argv, "--view", "0"));
  const int H = 224, W = 224;
  const float NEAR = 0.2f;

  // ---------- 1. encoder: image + intrinsics -> 6 Gaussian outputs (vknn Vulkan) ----------
  Config cfg;
  cfg.backend = BackendKind::kVulkan;
  cfg.precision = Precision::kFp16;
  cfg.cacheWeights = false;
  cfg.freeWeightsAfterUpload = true;
  auto sess = Runtime::load(enc, cfg);
  if (!sess) {
    fprintf(stderr, "failed to load encoder %s\n", enc.c_str());
    return 1;
  }
  std::vector<IOTensor> ins, outs;
  for (auto& info : sess->inputInfo()) {
    IOTensor t;
    t.name = info.name;
    t.shape = info.shape;
    t.dtype = DType::kFloat32;
    t.data = readFile(info.name == sess->inputInfo()[0].name ? imgp : intp);
    t.data.resize((size_t)numElements(info.shape) * 4, 0);
    ins.push_back(std::move(t));
  }
  printf("[encoder] running on GPU ...\n");
  if (sess->run(ins, outs) != Status::kOk) {
    fprintf(stderr, "encoder run failed\n");
    return 2;
  }
  auto get = [&](const char* nm) -> const float* {
    for (auto& o : outs)
      if (o.name == nm)
        return o.f32();
    fprintf(stderr, "encoder output '%s' missing\n", nm);
    return nullptr;
  };
  const float *means = get("means"), *covs = get("covariances"), *harm = get("harmonics"),
              *opac = get("opacities");
  if (!means || !covs || !harm || !opac)
    return 2;
  int N = 0;
  for (auto& o : outs)
    if (o.name == "means")
      N = (int)(numElements(o.shape) / 3);
  printf("[encoder] %d Gaussians\n", N);
  std::vector<float> mv(means, means + (size_t)N * 3), cvv(covs, covs + (size_t)N * 9),
      opv(opac, opac + (size_t)N), cols((size_t)N * 3);
  for (size_t i = 0; i < (size_t)N * 3; ++i)
    cols[i] = std::max(0.f, C0 * harm[i] + 0.5f);
  std::vector<float> intr(36, 0);
  {
    auto d = readFile(intp);
    memcpy(intr.data(), d.data(), std::min(d.size(), intr.size() * 4));
  }
  std::vector<float> extr;
  if (!extp.empty()) {
    auto d = readFile(extp);
    extr.assign((size_t)(view + 1) * 16, 0);
    memcpy(extr.data(), d.data(), std::min(d.size(), extr.size() * 4));
  }
  sess.reset();  // free the encoder (+ its Vulkan context) before the rasterizer

  // ---------- 2. camera: w2c = rigid inverse of c2w, K in pixels ----------
  float c2w[16];
  if (!extr.empty())
    memcpy(c2w, &extr[(size_t)view * 16], 64);
  else {
    memset(c2w, 0, 64);
    c2w[0] = c2w[5] = c2w[10] = c2w[15] = 1.f;
  }  // identity
  const float* K = &intr[(size_t)view * 9];
  float Rt[9], tp[3];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      Rt[r * 3 + c] = c2w[c * 4 + r];
  for (int r = 0; r < 3; ++r)
    tp[r] = -(Rt[r * 3] * c2w[3] + Rt[r * 3 + 1] * c2w[7] + Rt[r * 3 + 2] * c2w[11]);
  float fx = K[0] * W, fy = K[4] * H, cx = K[2] * W, cy = K[5] * H;

  // ---------- 3. rasterizer: Gaussians -> view, entirely on the GPU ----------
  vk::VulkanContext ctx;
  if (!ctx.initialized()) {
    fprintf(stderr, "no Vulkan for rasterizer\n");
    return 1;
  }
  vk::CommandRunner runner(ctx);
  vk::Buffer mB(ctx, mv.size() * 4), cvB(ctx, cvv.size() * 4), clB(ctx, cols.size() * 4),
      opB(ctx, opv.size() * 4);
  vk::Buffer gdB(ctx, (size_t)N * 12 * 4);
  mB.upload(mv.data(), mv.size() * 4);
  cvB.upload(cvv.data(), cvv.size() * 4);
  clB.upload(cols.data(), cols.size() * 4);
  opB.upload(opv.data(), opv.size() * 4);
  struct PrePC {
    int32_t dims[4];
    float cam[4];
    float cam2[4];
    float r0[4], r1[4], r2[4];
  } ppc{};
  ppc.dims[0] = N;
  ppc.dims[1] = H;
  ppc.dims[2] = W;
  ppc.cam[0] = NEAR;
  ppc.cam[1] = fx;
  ppc.cam[2] = fy;
  ppc.cam[3] = cx;
  ppc.cam2[0] = cy;
  for (int c = 0; c < 3; ++c) {
    ppc.r0[c] = Rt[c];
    ppc.r1[c] = Rt[3 + c];
    ppc.r2[c] = Rt[6 + c];
  }
  ppc.r0[3] = tp[0];
  ppc.r1[3] = tp[1];
  ppc.r2[3] = tp[2];

  int ntx = (W + 15) / 16, nty = (H + 15) / 16, nTiles = ntx * nty;
  int tileBits = 1;
  while ((1 << tileBits) < nTiles)
    ++tileBits;
  int DB = 32 - tileBits;
  int64_t CAP = 1;
  while (CAP < (int64_t)N * 8 || CAP < (1 << 16))
    CAP <<= 1;
  float invRange = 1.0f / (256.0f - NEAR);
  vk::Buffer cntB(ctx, 4), keyB(ctx, CAP * 4), valB(ctx, CAP * 4), rgB(ctx, (size_t)nTiles * 2 * 4);
  vk::Buffer imB(ctx, (size_t)H * W * 3 * 4, vk::MemPref::kReadback);
  vk::ComputePipeline pre(ctx, "raster_preprocess", 5, sizeof(PrePC));
  struct FillPC {
    uint32_t count, value;
  };
  vk::ComputePipeline fill(ctx, "raster_fill", 1, sizeof(FillPC));
  struct DupPC {
    int32_t N, ntx, nty, DB, CAP;
    float near, invRange;
  } dpc{N, ntx, nty, DB, (int)CAP, NEAR, invRange};
  vk::ComputePipeline dup(ctx, "raster_duplicate", 4, sizeof(DupPC));
  struct BitPC {
    uint32_t j, k, n;
  };
  vk::ComputePipeline bit(ctx, "raster_bitonic", 2, sizeof(BitPC));
  struct RngPC {
    int32_t mode, nTiles, CAP, DB;
  };
  vk::ComputePipeline rng(ctx, "raster_ranges", 2, sizeof(RngPC));
  struct CompPC {
    int32_t dims[4];
  } cpc{};
  cpc.dims[0] = H;
  cpc.dims[1] = W;
  cpc.dims[2] = ntx;
  vk::ComputePipeline comp(ctx, "raster_composite", 4, sizeof(CompPC));

  VkCommandBuffer cmd = runner.allocate();
  runner.begin(cmd);
  pre.dispatch(cmd, {mB.handle(), cvB.handle(), clB.handle(), opB.handle(), gdB.handle()}, &ppc,
               sizeof(ppc), (uint32_t)((N + 63) / 64));
  vk::computeBarrier(cmd);
  FillPC fk{(uint32_t)CAP, 0xffffffffu}, fc{1u, 0u};
  fill.dispatch(cmd, {keyB.handle()}, &fk, sizeof(fk), (uint32_t)((CAP + 255) / 256));
  fill.dispatch(cmd, {cntB.handle()}, &fc, sizeof(fc), 1);
  vk::computeBarrier(cmd);
  dup.dispatch(cmd, {gdB.handle(), cntB.handle(), keyB.handle(), valB.handle()}, &dpc, sizeof(dpc),
               (uint32_t)((N + 63) / 64));
  vk::computeBarrier(cmd);
  for (uint32_t k = 2; k <= (uint32_t)CAP; k <<= 1)
    for (uint32_t j = k >> 1; j > 0; j >>= 1) {
      BitPC bp{j, k, (uint32_t)CAP};
      bit.dispatch(cmd, {keyB.handle(), valB.handle()}, &bp, sizeof(bp),
                   (uint32_t)((CAP + 255) / 256));
      vk::computeBarrier(cmd);
    }
  RngPC rc{0, nTiles, (int)CAP, DB}, ra{1, nTiles, (int)CAP, DB};
  rng.dispatch(cmd, {keyB.handle(), rgB.handle()}, &rc, sizeof(rc),
               (uint32_t)((nTiles + 255) / 256));
  vk::computeBarrier(cmd);
  rng.dispatch(cmd, {keyB.handle(), rgB.handle()}, &ra, sizeof(ra), (uint32_t)((CAP + 255) / 256));
  vk::computeBarrier(cmd);
  comp.dispatch(cmd, {gdB.handle(), valB.handle(), rgB.handle(), imB.handle()}, &cpc, sizeof(cpc),
                (uint32_t)((W + 15) / 16), (uint32_t)((H + 15) / 16));
  runner.end(cmd);
  double ms = runner.submitAndWait(cmd);
  vkFreeCommandBuffers(ctx.device(), runner.pool(), 1, &cmd);
  std::vector<float> img((size_t)H * W * 3);
  imB.download(img.data(), img.size() * 4);
  printf("[rasterizer] view %d rendered on GPU in %.2f ms\n", view, ms);

  // ---------- 4. save PPM ----------
  std::ofstream f(outp, std::ios::binary);
  f << "P6\n" << W << " " << H << "\n255\n";
  for (size_t i = 0; i < img.size(); ++i)
    f.put((char)(uint8_t)(std::min(1.f, std::max(0.f, img[i])) * 255 + 0.5f));
  printf("[done] image -> %s  (mean=%.4f)\n", outp.c_str(), [&] {
    double s = 0;
    for (float x : img)
      s += x;
    return s / img.size();
  }());
  return 0;
#endif
}
