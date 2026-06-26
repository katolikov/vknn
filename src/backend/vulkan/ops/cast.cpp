// Cast on the GPU. Every runtime Cast here targets float32, so on the flat row-major path it is a
// pure buffer copy (the graph runs in one precision, leaving these casts as no-ops). The layout pass
// marks Cast layout-agnostic; supportsNode gates to float outputs (an int-target Cast would fall
// back to the CPU op).
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct CastOp: VulkanOp {
            std::shared_ptr<vk::Buffer> hold0; // when input is a constant initializer
            void                        prepare(const Node &, VkOpEnv &) override {
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer  *src = operandBuf(env, node.inputs[0], hold0);
                vk::Buffer  *dst = env.devBuf(node.outputs[0]);
                VkBufferCopy c {0, 0, std::min(src->bytes(), dst->bytes())};
                vkCmdCopyBuffer(cmd, src->handle(), dst->handle(), 1, &c);
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::kCast, CastOp);
} // namespace vknn
