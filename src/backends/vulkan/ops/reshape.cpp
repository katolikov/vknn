// Reshape. In NC4HW4 a reshape that only moves the trailing 1x1 dims (e.g. [1,1280,1,1] ->
// [1,1280]) leaves the packed bytes identical, so we just copy the buffer.
#include "vk_op_common.h"

namespace vx {
namespace {

struct ReshapeOp : VulkanOp {
  size_t bytes = 0;

  void prepare(const Node& node, VkOpEnv& env) override {
    int elem = env.useFp16 ? 2 : 4;
    bytes = (size_t)packedElems(env.graph->desc(node.outputs[0]).shape) * elem;
  }

  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* src = env.devBuf(node.inputs[0]);
    vk::Buffer* dst = env.devBuf(node.outputs[0]);
    VkBufferCopy c{0, 0, std::min({bytes, src->bytes(), dst->bytes()})};
    vkCmdCopyBuffer(cmd, src->handle(), dst->handle(), 1, &c);
  }
};

}  // namespace

VX_REGISTER_VK_OP(OpType::kReshape, ReshapeOp);

}  // namespace vx
