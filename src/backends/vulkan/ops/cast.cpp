// Cast on the GPU. Every runtime Cast in the transformer is to float32 (mixed-precision shims that
// became no-ops once the whole graph runs in one precision), so on the flat row-major path it's a
// pure buffer copy. The layout pass marks Cast layout-agnostic; supportsNode gates to float outputs
// (an int-target Cast — none here — would fall back to the CPU op).
#include "vk_op_common.h"

namespace vx {
namespace {

struct CastOp : VulkanOp {
  std::shared_ptr<vk::Buffer> hold0;  // when input is a constant initializer
  void prepare(const Node&, VkOpEnv&) override {}
  void record(VkCommandBuffer cmd, const Node& node, VkOpEnv& env) override {
    vk::Buffer* src = operandBuf(env, node.inputs[0], hold0);
    vk::Buffer* dst = env.devBuf(node.outputs[0]);
    VkBufferCopy c{0, 0, std::min(src->bytes(), dst->bytes())};
    vkCmdCopyBuffer(cmd, src->handle(), dst->handle(), 1, &c);
  }
};

}  // namespace
VX_REGISTER_VK_OP(OpType::kCast, CastOp);
}  // namespace vx
