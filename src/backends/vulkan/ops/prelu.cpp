// PRelu on the GPU (NC4HW4). Per-channel slope packed to [Cb][4] and uploaded once.
#include "vk_op_common.h"

namespace vx {
namespace {
struct PReluPC { int count, HW, Cb; };
struct PReluOp : VulkanOp {
  std::unique_ptr<vk::ComputePipeline> pipe;
  std::shared_ptr<vk::Buffer> slope;
  PReluPC pc{};
  void prepare(const Node& node, VkOpEnv& env) override {
    const Graph& g = *env.graph;
    NCHW x = NCHW::from(g.desc(node.outputs[0]).shape);
    int64_t Cb = cBlocks(x.c);
    pc = {(int)((int64_t)x.n * Cb * x.h * x.w), (int)(x.h * x.w), (int)Cb};
    const auto& si = g.initializers.at(node.inputs[1]);
    int64_t ns = (int64_t)si.bytes.size() / 4;
    slope = uploadCached(env, node.name + "#slope", [&] {
      std::vector<float> sp(Cb * 4, 0.f);
      const float* s = si.f32();
      for (int64_t c = 0; c < x.c; ++c) sp[c] = ns == 1 ? s[0] : s[c];
      return sp;
    });
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("prelu", env.useFp16), 3,
                                                 sizeof(PReluPC), std::vector<uint32_t>{},
                                                 env.cache->handle());
  }
  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* s = env.devBuf(node.inputs[0]);
    vk::Buffer* d = env.devBuf(node.outputs[0]);
    pipe->dispatch(cmd, {s->handle(), slope->handle(), d->handle()}, &pc, sizeof(pc),
                   groups(pc.count, 256));
  }
};
}  // namespace
VX_REGISTER_VK_OP(OpType::kPRelu, PReluOp);
}  // namespace vx
