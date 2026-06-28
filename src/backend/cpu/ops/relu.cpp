// ReLU.
#include "backend/cpu/cpu_backend.h"

namespace vknn {
    namespace {

        struct ReluCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X = ctx.t(node.inputs[0]);
                RtTensor       &Y = ctx.t(node.outputs[0]);
                int64_t         n = X.elems();
                float          *y = cpu::allocOut(Y, X.shape);
                const float    *x = X.host.f32();
                for (int64_t i = 0; i < n; ++i)
                {
                    y[i] = x[i] > 0 ? x[i] : 0;
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Relu, ReluCpu);
} // namespace vknn
