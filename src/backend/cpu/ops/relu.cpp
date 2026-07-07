// ReLU: elementwise y = max(x, 0), fp32. The output mirrors the input's shape exactly (no
// broadcasting, no axis), so this is a straight flat walk over all elements.
//
// The rectifier is expressed as `x > 0 ? x : 0` rather than a true max: because `NaN > 0` is
// false, a NaN input falls through to 0, whereas std::max(NaN, 0) would propagate the NaN. The
// -0.0 input case also yields +0 here (`-0.0 > 0` is false).
#include "backend/cpu/cpu_backend.h"

namespace vknn {
    namespace {

        struct ReluCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X = ctx.t(node.inputs[0]);
                RtTensor       &Y = ctx.t(node.outputs[0]);
                int64_t         n = cpu::elemCount(X.shape); // a rank-0 scalar carries its one element
                float          *y = cpu::allocOut(Y, X.shape);
                const float    *x = X.host.f32();
                for (int64_t i = 0; i < n; ++i)
                {
                    // Positive values pass through unchanged; everything else (including NaN, which
                    // fails the comparison) becomes 0. See header for the NaN/-0.0 rationale.
                    y[i] = x[i] > 0 ? x[i] : 0;
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Relu, ReluCpu);
} // namespace vknn
