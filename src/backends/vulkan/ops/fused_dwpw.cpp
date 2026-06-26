// Fused depthwise-3x3 + 1x1-project on the GPU. One workgroup per output pixel; depthwise output
// staged in LDS (computed once), then projected. Gated by supportsNode to large-spatial fp16 blocks.
#include "vk_op_common.h"
#include "vx/op.h"

namespace vx {
namespace {
struct DwPwPC {
  int N, E, H, W, Cout, OH, OW, KH, KW, SH, SW, PT, PL, DH, DWd, dwAct, pwAct;
  float pwLo, pwHi;
};
struct FusedDwPwOp : VulkanOp {
  std::unique_ptr<vk::ComputePipeline> pipe;
  std::shared_ptr<vk::Buffer> dww, dwb, pww, pwb;
  DwPwPC pc{};
  int64_t groups_ = 0;
  bool hasRes = false;

  void prepare(const Node& node, VkOpEnv& env) override {
    const Graph& g = *env.graph;
    NCHW x = NCHW::from(g.desc(node.inputs[0]).shape);   // exp [N,E,H,W]
    NCHW y = NCHW::from(g.desc(node.outputs[0]).shape);  // out [N,Cout,OH,OW]
    const Shape& dws = g.desc(node.inputs[1]).shape;     // [E,1,KH,KW]
    const Shape& pws = g.desc(node.inputs[3]).shape;     // [Cout,E,1,1]
    int64_t E = x.c, Cout = pws[0], KH = dws[2], KW = dws[3];
    int64_t Eb = cBlocks(E), Coutb = cBlocks(Cout);
    auto a = [&](const char* k, std::vector<int64_t> d) {
      const auto& v = node.attr.getints(k); return v.empty() ? d : v;
    };
    auto st = a("strides", {1, 1}), pad = a("pads", {0, 0, 0, 0}), dil = a("dilations", {1, 1});
    hasRes = (node.fusedResidual != kNoTensor);
    pc = {(int)x.n, (int)E, (int)x.h, (int)x.w, (int)Cout, (int)y.h, (int)y.w, (int)KH, (int)KW,
          (int)st[0], (int)st[1], (int)pad[0], (int)pad[1], (int)dil[0], (int)dil[1],
          (int)node.subOp, (int)node.fusedAct, node.actLo, node.actHi};
    groups_ = (int64_t)x.n * y.h * y.w;
    std::vector<float> dwsrcv = initFloats(g, node.inputs[1]);
    std::vector<float> pwsrcv = initFloats(g, node.inputs[3]);
    const float* dwsrc = dwsrcv.data();
    const float* pwsrc = pwsrcv.data();
    dww = uploadCached(env, node.name + "#dww", [&] {  // [Eb][KH][KW][4]
      std::vector<float> wp(Eb * KH * KW * 4, 0.f);
      for (int64_t c = 0; c < E; ++c) { int64_t cb = c / 4, l = c % 4;
        for (int64_t ky = 0; ky < KH; ++ky) for (int64_t kx = 0; kx < KW; ++kx)
          wp[(((cb * KH + ky) * KW + kx) * 4) + l] = dwsrc[((c * KH + ky) * KW + kx)]; }
      return wp;
    });
    pww = uploadCached(env, node.name + "#pww", [&] {  // [Cout][Eb][4]
      std::vector<float> wp((size_t)Cout * Eb * 4, 0.f);
      for (int64_t oc = 0; oc < Cout; ++oc) for (int64_t ic = 0; ic < E; ++ic) {
        int64_t icb = ic / 4, l = ic % 4; wp[((oc * Eb + icb) * 4) + l] = pwsrc[oc * E + ic]; }
      return wp;
    });
    dwb = uploadCached(env, node.name + "#dwb", [&] {
      std::vector<float> b(Eb * 4, 0.f);
      if (node.inputs[2] != kNoTensor) { std::vector<float> sv = initFloats(g, node.inputs[2]); const float* s = sv.data();
        for (int64_t i = 0; i < E; ++i) b[i] = s[i]; }
      return b;
    });
    pwb = uploadCached(env, node.name + "#pwb", [&] {
      std::vector<float> b(Coutb * 4, 0.f);
      if (node.inputs[4] != kNoTensor) { std::vector<float> sv = initFloats(g, node.inputs[4]); const float* s = sv.data();
        for (int64_t i = 0; i < Cout; ++i) b[i] = s[i]; }
      return b;
    });
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("fused_dwpw", env.useFp16),
                                                 hasRes ? 7 : 6, sizeof(DwPwPC),
                                                 std::vector<uint32_t>{(uint32_t)(hasRes ? 1 : 0)},
                                                 env.cache->handle());
  }
  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* exp = env.devBuf(node.inputs[0]);
    vk::Buffer* dst = env.devBuf(node.outputs[0]);
    std::vector<VkBuffer> b = {exp->handle(), dww->handle(), dwb->handle(), pww->handle(),
                               pwb->handle(), dst->handle()};
    if (hasRes) b.push_back(env.devBuf(node.fusedResidual)->handle());
    pipe->dispatch(cmd, b, &pc, sizeof(pc), (uint32_t)groups_);
  }
};
}  // namespace
VX_REGISTER_VK_OP(OpType::kFusedDwPw, FusedDwPwOp);
}  // namespace vx
