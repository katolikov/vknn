// Flatten on the GPU. Like Reshape, in NC4HW4 a flatten of trailing 1x1 dims leaves the packed
// bytes unchanged, so it's just a buffer copy.
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct FlattenOp: VulkanOp {
            std::shared_ptr<vk::Buffer> hold0; // when input is a constant initializer
            void                        prepare(const Node &, VkOpEnv &) override {
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *src = operandBuf(env, node.inputs[0], hold0);
                vk::Buffer *dst = env.devBuf(node.outputs[0]);
                if (src->handle() == dst->handle())
                {
                    return; // output aliases the input buffer (geometry-as-metadata): no copy needed
                }
                // Flatten preserves the element count and (per the layout pass) the layout, so src/dst are
                // equal-sized: copy the whole buffer. Do NOT size the copy from packedElems(output): NCHW::from
                // collapses a rank>4 shape to (1,1,1,1) -> 4 elements, which would copy only the first 4 and
                // leave the tail of the (correctly numElements-sized flat) buffer garbage.
                VkBufferCopy c {0, 0, std::min(src->bytes(), dst->bytes())};
                vkCmdCopyBuffer(cmd, src->handle(), dst->handle(), 1, &c);
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Flatten, FlattenOp);
} // namespace vknn
