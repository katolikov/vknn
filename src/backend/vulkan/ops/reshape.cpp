// Reshape. In NC4HW4 a reshape that only moves the trailing 1x1 dims (e.g. [1,1280,1,1] ->
// [1,1280]) leaves the packed bytes identical, so we just copy the buffer.
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct ReshapeOp: VulkanOp {
            std::shared_ptr<vk::Buffer> hold0; // when input is a constant initializer
            void                        prepare(const Node &node, VkOpEnv &env) override {
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *src = operandBuf(env, node.inputs[0], hold0);
                vk::Buffer *dst = env.devBuf(node.outputs[0]);
                // A reshape preserves the element count, and the layout pass guarantees input and output share
                // a layout (else it inserts a ConvertLayout), so their buffers are the same size — copy it
                // whole. Do NOT size the copy from packedElems(output): NCHW::from collapses to (1,1,1,1) for
                // rank>4 (e.g. ShuffleNet's rank-5 channel-shuffle reshape [N,116,H,W]->[N,2,58,H,W]), which
                // copies only 4 elements and leaves the rest of the (correctly numElements-sized flat) buffer
                // garbage.
                VkBufferCopy c {0, 0, std::min(src->bytes(), dst->bytes())};
                vkCmdCopyBuffer(cmd, src->handle(), dst->handle(), 1, &c);
            }
        };

    } // namespace

    VKNN_REGISTER_VK_OP(OpType::kReshape, ReshapeOp);

} // namespace vknn
