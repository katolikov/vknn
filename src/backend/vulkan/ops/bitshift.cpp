// Flat (row-major) BitShift on the GPU: operand a shifted by operand b at the node's int_bits width
// (absent: 64) with N-D broadcasting at any rank, on shaders/bitshift.comp. The direction is the string
// attribute "LEFT" / "RIGHT"; vkNodeGate keeps any other value, and an invalid int_bits / int_signed, on
// the CPU op, and prepare() rejects either by name. Geometry, constant operands and precision follow
// bitwise_vk.h. Push-constant layout byte-matches shaders/bitshift.comp.
#include "bitwise_vk.h"
#include "core/bitwise_attrs.h"

namespace vknn {
    namespace {

        struct BitShiftVk: VulkanOp {
            /// Byte-matches the push_constant block of shaders/bitshift.comp.
            struct PushConstants {
                int rank, total, directionLeft, intBits;
            } pc {};
            bitwise_vk::BroadcastOperands        operands;
            std::shared_ptr<vk::ComputePipeline> pipe;

            void prepare(const Node &node, VkOpEnv &env) override {
                const bitwise::ShiftDirection direction = bitwise::readShiftDirection(node);
                const bitwise::IntegerWidth   width     = bitwise::readIntegerWidth(node);
                operands.prepare(node, env);
                pc.rank          = operands.rank;
                pc.total         = operands.total;
                pc.directionLeft = direction == bitwise::ShiftDirection::Left ? 1 : 0;
                pc.intBits       = width.bits;
                pipe = env.pipeline(shader("bitshift", env.useFp16), bitwise_vk::kBroadcastBufferCount, sizeof(PushConstants), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                // One flat invocation per output element; bitshift.comp is local_size_x=256 == flat::kFlatLocalSize.
                pipe->dispatch(cmd,
                               {operands.operandBuffer(node, env, bitwise_vk::kOperandA)->handle(), operands.operandBuffer(node, env, bitwise_vk::kOperandB)->handle(),
                                env.devBuf(node.outputs[0])->handle(), operands.geometry->handle()},
                               &pc, sizeof(pc), groups(pc.total, flat::kFlatLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::BitShift, BitShiftVk);
} // namespace vknn
