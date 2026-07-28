// Unsqueeze on the GPU: inserting size-1 dims is a pure metadata reshape, so on the flat row-major
// path the bytes are identical and we just copy the buffer (mirrors Squeeze/Reshape). The layout
// pass marks Unsqueeze layout-agnostic, so input and output share the flat row-major layout.
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct UnsqueezeOp: VulkanOp {
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
                // Unsqueeze preserves the element count and (per the layout pass) the layout, so src/dst are
                // equal-sized: copy the whole buffer by byte count. Do NOT size the copy from
                // packedElems(output): that returns the NC4HW4 padded count (cBlocks(C)*4*H*W), which differs
                // from the flat row-major buffer's numElements size on this path and would copy the wrong span.
                VkBufferCopy c {0, 0, std::min(src->bytes(), dst->bytes())};
                vkCmdCopyBuffer(cmd, src->handle(), dst->handle(), 1, &c);
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Unsqueeze, UnsqueezeOp);
} // namespace vknn
