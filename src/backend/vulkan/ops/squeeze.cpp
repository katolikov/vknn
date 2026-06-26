// Squeeze on the GPU: removing size-1 dims is a pure metadata reshape, so on the flat row-major
// path the bytes are identical and we just copy the buffer (same as ReshapeOp). The layout pass
// marks Squeeze layout-agnostic, so input and output share the flat row-major layout.
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct SqueezeOp: VulkanOp {
            std::shared_ptr<vk::Buffer> hold0; // when input is a constant initializer

            void prepare(const Node &, VkOpEnv &) override {
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *src = operandBuf(env, node.inputs[0], hold0);
                vk::Buffer *dst = env.devBuf(node.outputs[0]);
                // A squeeze preserves the element count + layout, so src/dst are equal-sized — copy the whole
                // buffer. Do NOT cap by packedElems(output): NCHW::from collapses rank>4 to (1,1,1,1) -> 4,
                // which would truncate a rank-5 flat squeeze (e.g. the quaternion [1,2,N,1,1,1]->[1,2,N,1,1])
                // to 4 elems.
                VkBufferCopy c {0, 0, std::min(src->bytes(), dst->bytes())};
                vkCmdCopyBuffer(cmd, src->handle(), dst->handle(), 1, &c);
            }
        };

    } // namespace

    VKNN_REGISTER_VK_OP(OpType::kSqueeze, SqueezeOp);

} // namespace vknn
