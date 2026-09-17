// ONNX BitwiseXor: elementwise two's-complement bitwise XOR with NumPy broadcasting. The operator applied
// to sign-extended operands yields the sign-extended result, so the op needs no width attribute.
// Operand reading and output storage follow backend/cpu/bitwise_int.h.
#include "backend/cpu/bitwise_int.h"

namespace vknn {
    namespace {

        struct BitwiseXorCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                cpu::bitwise::runBroadcastInteger(node, ctx, cpu::bitwise::bitwiseXor);
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::BitwiseXor, BitwiseXorCpu);
} // namespace vknn
