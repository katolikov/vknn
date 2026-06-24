// Residual add over NC4HW4 buffers. Both inputs have the same packed layout, so it's just a
// flat elementwise add over the buffer.
#include "vk_op_common.h"

namespace vx {
namespace {

struct AddOp : VulkanOp {
  std::unique_ptr<vk::ComputePipeline> pipe;
  uint32_t count = 0;

  void prepare(const Node& node, VkOpEnv& env) override {
    count = (uint32_t)packedElems(env.graph->desc(node.outputs[0]).shape);
    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("add", env.useFp16), 3,
                                                 sizeof(uint32_t), std::vector<uint32_t>{},
                                                 env.cache->handle());
  }

  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* a = env.devBuf(node.inputs[0]);
    vk::Buffer* b = env.devBuf(node.inputs[1]);
    vk::Buffer* y = env.devBuf(node.outputs[0]);
    pipe->dispatch(cmd, {a->handle(), b->handle(), y->handle()}, &count, sizeof(count),
                   groups(count, 256));
  }
};

}  // namespace

VX_REGISTER_VK_OP(OpType::kAdd, AddOp);

}  // namespace vx
