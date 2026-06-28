// Shape: emit the input's dims as an int64 vector.
#include "backend/cpu/cpu_backend.h"

namespace vknn {
    namespace {

        struct ShapeCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X = ctx.t(node.inputs[0]);
                RtTensor       &Y = ctx.t(node.outputs[0]);
                int64_t         r = (int64_t) X.shape.size();
                int64_t        *y = cpu::allocOutI64(Y, {r});
                for (int64_t i = 0; i < r; ++i)
                {
                    y[i] = X.shape[i];
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Shape, ShapeCpu);
} // namespace vknn
