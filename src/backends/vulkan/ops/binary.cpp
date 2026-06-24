// Elementwise binary family on the GPU (Mul/Sub/Div/Max/Min/Pow). Handles same-shape inputs and
// the channel-broadcast case (second operand [N,C,1,1], the Squeeze-Excite scale). Other broadcast
// patterns fall back to the CPU binary op (gated in the backend's supportsNode()).
#include "vk_op_common.h"

namespace vx {
namespace {

struct BinaryPC {
  int count, HW, op;
};

struct BinaryOp : VulkanOp {
  std::unique_ptr<vk::ComputePipeline> pipe;
  BinaryPC pc{};

  void prepare(const Node& node, VkOpEnv& env) override {
    NCHW y = NCHW::from(env.graph->desc(node.outputs[0]).shape);
    NCHW b = NCHW::from(env.graph->desc(node.inputs[1]).shape);
    int HW = (int)(y.h * y.w);
    bool bcast = (b.h * b.w == 1 && y.h * y.w != 1);
    pc = {(int)((int64_t)y.n * cBlocks(y.c) * HW), HW, node.subOp};
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("binary", env.useFp16), 3,
                                                 sizeof(BinaryPC),
                                                 std::vector<uint32_t>{(uint32_t)(bcast ? 1 : 0)},
                                                 env.cache->handle());
  }

  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* a = env.devBuf(node.inputs[0]);
    vk::Buffer* b = env.devBuf(node.inputs[1]);
    vk::Buffer* c = env.devBuf(node.outputs[0]);
    pipe->dispatch(cmd, {a->handle(), b->handle(), c->handle()}, &pc, sizeof(pc),
                   groups(pc.count, 256));
  }
};

}  // namespace
VX_REGISTER_VK_OP(OpType::kBinary, BinaryOp);
}  // namespace vx
