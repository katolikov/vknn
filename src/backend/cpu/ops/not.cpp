// ONNX Not (elementwise boolean negation): y = (x != 0) ? 0 : 1, same shape as x. The operand is read
// by the shared logical rule, so an int64 input (a Cast-to-BOOL result or an int64 mask) is tested
// exactly rather than reinterpreted as fp32 bytes; the result is canonical fp32 1.0 / 0.0. See
// backend/cpu/logical_ops.h.
#include "backend/cpu/logical_ops.h"

namespace vknn {
    namespace {

        constexpr size_t kNotOperands = 1;

        struct NotCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                cpu::requireLogicalOperands(node, kNotOperands);
                const RtTensor           &X     = ctx.t(node.inputs[0]);
                RtTensor                 &Y     = ctx.t(node.outputs[0]);
                const Shape               shape = X.shape;
                const int64_t             count = cpu::elemCount(shape); // a rank-0 scalar carries its one element
                const cpu::LogicalOperand x(X);
                float                    *y = cpu::allocOut(Y, shape);
                cpu::parallelFor(cpu::threadCount(ctx.config), 0, count, cpu::minChunkForWork(1), [&](int64_t chunkBegin, int64_t chunkEnd) {
                    for (int64_t index = chunkBegin; index < chunkEnd; ++index)
                    {
                        y[index] = cpu::logicalValue(!x.isTrue(index));
                    }
                });
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Not, NotCpu);
} // namespace vknn
