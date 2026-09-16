// Flat (row-major) BitwiseNot on the GPU: the complement of each operand's integer value at the width the
// node's int_bits / int_signed attributes give (absent: 64-bit signed), a 1:1 map with no geometry SSBO.
// vkNodeGate keeps a node with an invalid width on the CPU op, and prepare() rejects one by name.
// Precision and constant operand upload follow bitwise_vk.h. Push-constant layout byte-matches
// shaders/bitwise_not.comp.
#include "bitwise_vk.h"
#include "core/bitwise_attrs.h"

namespace vknn {
    namespace {

        struct BitwiseNotVk: VulkanOp {
            /// Byte-matches the push_constant block of shaders/bitwise_not.comp.
            struct PushConstants {
                int total, intBits, intSigned;
            } pc {};
            static constexpr uint32_t            kBufferCount = 2; // operand (binding 0), output (binding 1)
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          constantOperand; // when the operand is a constant initializer

            void prepare(const Node &node, VkOpEnv &env) override {
                const bitwise::IntegerWidth width = bitwise::readIntegerWidth(node);
                pc.total                          = (int) numElements(env.graph->desc(node.outputs[0]).shape);
                pc.intBits                        = width.bits;
                pc.intSigned                      = width.isSigned ? 1 : 0;
                if (env.graph->isInitializer(node.inputs[0]))
                {
                    constantOperand = bitwise_vk::uploadConstantOperand(env, node.inputs[0]);
                }
                pipe = env.pipeline(shader("bitwise_not", env.useFp16), kBufferCount, sizeof(PushConstants), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *source = constantOperand ? constantOperand.get() : env.devBuf(node.inputs[0]);
                // One flat invocation per element; bitwise_not.comp is local_size_x=256 == flat::kFlatLocalSize.
                pipe->dispatch(cmd, {source->handle(), env.devBuf(node.outputs[0])->handle()}, &pc, sizeof(pc), groups(pc.total, flat::kFlatLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::BitwiseNot, BitwiseNotVk);
} // namespace vknn
