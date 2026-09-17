// ONNX BitwiseNot: elementwise complement at the operand's integer width, same shape. A signed type or a
// 64-bit width complements all bits (~v); an unsigned type narrower than 64 bits keeps its low `int_bits`
// bits (uint8: 255 - v). The width comes from the `int_bits` / `int_signed` attributes the ONNX importer
// stamps (absent: 64-bit signed). Operand reading and output storage follow backend/cpu/bitwise_int.h.
#include "backend/cpu/bitwise_int.h"

namespace vknn {
    namespace {

        struct BitwiseNotCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const vknn::bitwise::IntegerWidth width = vknn::bitwise::readIntegerWidth(node);
                cpu::bitwise::runElementwiseInteger(node, ctx, [width](int64_t value) {
                    return cpu::bitwise::bitwiseNot(value, width);
                });
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::BitwiseNot, BitwiseNotCpu);
} // namespace vknn
