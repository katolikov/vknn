// ONNX Xor (elementwise boolean exclusive OR) with NumPy-style broadcasting:
// y = ((a != 0) != (b != 0)) ? 1 : 0. Operands are read by the shared logical rule (int64 exact,
// everything else through fp32, NaN true) and the result is canonical fp32 1.0 / 0.0; see
// backend/cpu/logical_ops.h.
#include "backend/cpu/logical_ops.h"

namespace vknn {
    namespace {

        struct XorCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                cpu::runLogicalBinary(node, ctx, [](bool aTrue, bool bTrue) {
                    return aTrue != bTrue;
                });
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Xor, XorCpu);
} // namespace vknn
