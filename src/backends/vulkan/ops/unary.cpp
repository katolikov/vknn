// Elementwise unary family on the GPU (Sigmoid/Tanh/HardSwish/HardSigmoid/LeakyRelu/Elu/...).
// Operates on the packed buffer, so it's correct for any layout.
#include "vk_op_common.h"

namespace vx {
namespace {

struct UnaryPC {
  uint32_t count;
  int op;
  float a, b;
};

struct UnaryOp : VulkanOp {
  std::unique_ptr<vk::ComputePipeline> pipe;
  UnaryPC pc{};
  std::shared_ptr<vk::Buffer> hold;  // when the input is a constant initializer
  void prepare(const Node& node, VkOpEnv& env) override {
    const auto& od = env.graph->desc(node.outputs[0]);
    // flat (B,N,C) buffers are numElements-sized; NC4HW4 buffers are packedElems-sized.
    uint32_t count = (uint32_t)(od.gpuFlat ? numElements(od.shape) : packedElems(od.shape));
    pc = {count, node.subOp, node.actLo, node.actHi};
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("unary", env.useFp16), 2,
                                                 sizeof(UnaryPC), std::vector<uint32_t>{},
                                                 env.cache->handle());
  }
  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* s = operandBuf(env, node.inputs[0], hold);
    vk::Buffer* d = env.devBuf(node.outputs[0]);
    pipe->dispatch(cmd, {s->handle(), d->handle()}, &pc, sizeof(pc), groups(pc.count, 256));
  }
};

}  // namespace
VX_REGISTER_VK_OP(OpType::kUnary, UnaryOp);
}  // namespace vx
