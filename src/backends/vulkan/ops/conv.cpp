// Conv2D on the GPU. One op handles both the group==1 case (the "conv" shader, which also
// covers 1x1 pointwise) and the depthwise case (the "dwconv" shader). Weights are repacked to
// NC4HW4 on the host and uploaded once. For the group==1 path we also autotune the workgroup
// size the first time we see a given shape and cache the winner.
#include "vk_op_common.h"
#include "vx/logging.h"

namespace vx {
namespace {

struct ConvOp : VulkanOp {
  bool depthwise = false;
  std::unique_ptr<vk::ComputePipeline> pipe;
  std::shared_ptr<vk::Buffer> wbuf, bbuf;
  ConvPC pc{};
  DwPC dpc{};
  int64_t total = 0;
  uint32_t localSize = 64;

  // Try a few workgroup sizes for this exact shape, keep the fastest, and remember it so the
  // next session (or run) skips the measurement. Only the group==1 conv shader is tunable.
  uint32_t pickLocalSize(VkOpEnv& env, vk::Buffer* src, vk::Buffer* dst) {
    char buf[96];
    snprintf(buf, sizeof(buf), "convls_%d_%d_%d_%d_%d_%d_%d_%d", pc.Cin, pc.H, pc.W, pc.Cout, pc.OH,
             pc.OW, pc.KH, pc.SH);
    std::string sig = buf;
    if (env.weights) {
      int cached = env.weights->tuned(sig, 0);
      if (cached > 0) return (uint32_t)cached;
    }
    uint32_t best = 64;
    if (env.tuning != TuningLevel::kOff && env.runner) {
      std::vector<uint32_t> cands = (env.tuning == TuningLevel::kThorough)
                                        ? std::vector<uint32_t>{32, 64, 128, 256}
                                        : std::vector<uint32_t>{64, 128, 256};
      double bestMs = 1e30;
      for (uint32_t ls : cands) {
        vk::ComputePipeline p(*env.ctx, shader("conv", env.useFp16), 4, sizeof(ConvPC), {ls},
                              env.cache->handle());
        VkCommandBuffer cmd = env.runner->allocate();
        env.runner->begin(cmd);
        for (int rep = 0; rep < 8; ++rep)
          p.dispatch(cmd, {src->handle(), wbuf->handle(), bbuf->handle(), dst->handle()}, &pc,
                     sizeof(pc), groups(total, ls));
        env.runner->end(cmd);
        double ms = env.runner->submitAndWait(cmd);
        vkFreeCommandBuffers(env.ctx->device(), env.runner->pool(), 1, &cmd);
        if (ms < bestMs) {
          bestMs = ms;
          best = ls;
        }
      }
      VX_DEBUG << "autotune " << sig << " -> local_size_x=" << best;
    }
    if (env.weights) env.weights->setTuned(sig, (int)best);
    return best;
  }

  void prepare(const Node& node, VkOpEnv& env) override {
    const Graph& g = *env.graph;
    NCHW x = NCHW::from(g.desc(node.inputs[0]).shape);
    NCHW y = NCHW::from(g.desc(node.outputs[0]).shape);
    const Shape& ws = g.desc(node.inputs[1]).shape;  // [Cout, Cin/group, KH, KW]
    int64_t Cout = ws[0], inCg = ws[1], KH = ws[2], KW = ws[3];
    auto st = attrInts(node, "strides", {1, 1});
    auto pad = attrInts(node, "pads", {0, 0, 0, 0});
    auto dil = attrInts(node, "dilations", {1, 1});
    int64_t group = node.attr.geti("group", 1);
    depthwise = (group == x.c && group == Cout && inCg == 1);

    const float* wsrc = g.initializers.at(node.inputs[1]).f32();
    int64_t Coutb = cBlocks(Cout);

    // bias, padded out to a multiple of 4 so the kernel can read whole vec4s
    bbuf = uploadCached(env, node.name + "#b", [&] {
      std::vector<float> bias(Coutb * 4, 0.f);
      if (node.inputs.size() > 2 && node.inputs[2] != kNoTensor) {
        const float* bsrc = g.initializers.at(node.inputs[2]).f32();
        for (int64_t i = 0; i < Cout; ++i) bias[i] = bsrc[i];
      }
      return bias;
    });

    if (depthwise) {
      int64_t Cb = cBlocks(x.c);
      wbuf = uploadCached(env, node.name + "#w", [&] {
        // [C,1,KH,KW] -> [Cb][KH][KW][4]
        std::vector<float> wp(Cb * KH * KW * 4, 0.f);
        for (int64_t c = 0; c < x.c; ++c) {
          int64_t cb = c / 4, l = c % 4;
          for (int64_t ky = 0; ky < KH; ++ky)
            for (int64_t kx = 0; kx < KW; ++kx)
              wp[(((cb * KH + ky) * KW + kx) * 4) + l] = wsrc[((c * KH + ky) * KW + kx)];
        }
        return wp;
      });
      dpc = {(int)x.n,    (int)x.c,    (int)x.h,           (int)x.w,   (int)y.h,    (int)y.w,
             (int)KH,     (int)KW,     (int)st[0],         (int)st[1], (int)pad[0], (int)pad[1],
             (int)dil[0], (int)dil[1], (int)node.fusedAct, 0,          node.actLo,  node.actHi};
      total = x.n * Cb * y.h * y.w;
      pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("dwconv", env.useFp16), 4,
                                                   sizeof(DwPC), std::vector<uint32_t>{},
                                                   env.cache->handle());
    } else {
      int64_t Cinb = cBlocks(x.c);
      pc = {(int)x.n,    (int)x.c,    (int)x.h,    (int)x.w,           (int)Cout,  (int)y.h,
            (int)y.w,    (int)KH,     (int)KW,     (int)st[0],         (int)st[1], (int)pad[0],
            (int)pad[1], (int)dil[0], (int)dil[1], (int)node.fusedAct, node.actLo, node.actHi};
      total = x.n * Coutb * y.h * y.w;
      wbuf = uploadCached(env, node.name + "#w", [&] {
        // [Cout,Cin,KH,KW] -> [Cout][Cinb][KH][KW][4]
        std::vector<float> wp((size_t)Cout * Cinb * KH * KW * 4, 0.f);
        for (int64_t oc = 0; oc < Cout; ++oc)
          for (int64_t ic = 0; ic < x.c; ++ic) {
            int64_t icb = ic / 4, l = ic % 4;
            for (int64_t ky = 0; ky < KH; ++ky)
              for (int64_t kx = 0; kx < KW; ++kx)
                wp[(((((oc * Cinb + icb) * KH + ky) * KW + kx) * 4) + l)] =
                    wsrc[(((oc * x.c + ic) * KH + ky) * KW + kx)];
          }
        return wp;
      });
      localSize = pickLocalSize(env, env.devBuf(node.inputs[0]), env.devBuf(node.outputs[0]));
      pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("conv", env.useFp16), 4,
                                                   sizeof(ConvPC), std::vector<uint32_t>{localSize},
                                                   env.cache->handle());
    }
  }

  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* src = env.devBuf(node.inputs[0]);
    vk::Buffer* dst = env.devBuf(node.outputs[0]);
    std::vector<VkBuffer> bufs = {src->handle(), wbuf->handle(), bbuf->handle(), dst->handle()};
    if (depthwise)
      pipe->dispatch(cmd, bufs, &dpc, sizeof(dpc), groups(total, 64));
    else
      pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(total, localSize));
  }
};

}  // namespace

VX_REGISTER_VK_OP(OpType::kConv, ConvOp);

}  // namespace vx
