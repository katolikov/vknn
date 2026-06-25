// Channel-axis Softmax on the GPU (NC4HW4, HW==1). supportsNode gates to the channel-softmax case
// (logits [N,C] / [N,C,1,1]); other axes fall back to the CPU softmax.
#include "vk_op_common.h"
#include "vx/op.h"

namespace vx {
namespace {
struct SmPC { int N, C; };
struct SoftmaxOp : VulkanOp {
  std::unique_ptr<vk::ComputePipeline> pipe;
  SmPC pc{};
  void prepare(const Node& node, VkOpEnv& env) override {
    NCHW x = NCHW::from(env.graph->desc(node.inputs[0]).shape);
    pc = {(int)x.n, (int)x.c};
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("softmax", env.useFp16), 2,
                                                 sizeof(SmPC), std::vector<uint32_t>{},
                                                 env.cache->handle());
  }
  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* s = env.devBuf(node.inputs[0]);
    vk::Buffer* d = env.devBuf(node.outputs[0]);
    pipe->dispatch(cmd, {s->handle(), d->handle()}, &pc, sizeof(pc), (uint32_t)pc.N);
  }
};
}  // namespace
VX_REGISTER_VK_OP(OpType::kSoftmax, SoftmaxOp);
}  // namespace vx
