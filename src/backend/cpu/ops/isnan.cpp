// ONNX IsNaN: elementwise y = isnan(x), float input -> bool output. The output mirrors the input's
// shape exactly (no broadcasting, no axis), so this is a straight flat walk over all elements. The
// bool result is the canonical fp32 1.0/0.0 the flat comparison ops emit, so it can feed a
// downstream Where/And over fp32 tensors (the softmax NaN-guard form).
#include "backend/cpu/cpu_backend.h"
#include <cmath>

namespace vknn {
    namespace {

        struct IsNaNCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X = ctx.t(node.inputs[0]);
                RtTensor       &Y = ctx.t(node.outputs[0]);
                int64_t         n = cpu::elemCount(X.shape); // a rank-0 scalar carries its one element
                float          *y = cpu::allocOut(Y, X.shape);
                const float    *x = X.host.f32();
                for (int64_t i = 0; i < n; ++i)
                {
                    y[i] = std::isnan(x[i]) ? 1.0f : 0.0f;
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::IsNaN, IsNaNCpu);
} // namespace vknn
