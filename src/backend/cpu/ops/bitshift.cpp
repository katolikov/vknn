// ONNX BitShift: elementwise shift of operand 0 by operand 1 with NumPy broadcasting, at the operand's
// integer width (`int_bits`, stamped by the ONNX importer; absent: 64). `direction` is the string "LEFT"
// or "RIGHT"; any other value throws InvalidArgument naming the node. A shift count outside [0, int_bits)
// yields 0; LEFT keeps the low `int_bits` bits (uint8: 200 << 1 = 144); RIGHT is a logical shift of the
// operand's low `int_bits` bits. Operand reading and output storage follow backend/cpu/bitwise_int.h.
#include "backend/cpu/bitwise_int.h"

namespace vknn {
    namespace {

        struct BitShiftCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const vknn::bitwise::ShiftDirection direction = vknn::bitwise::readShiftDirection(node);
                const int                           bits      = vknn::bitwise::readIntegerWidth(node).bits;
                cpu::bitwise::runBroadcastInteger(node, ctx, [direction, bits](int64_t value, int64_t shift) {
                    return cpu::bitwise::bitShift(value, shift, direction, bits);
                });
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::BitShift, BitShiftCpu);
} // namespace vknn
