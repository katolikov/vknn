// GlobalAveragePool: collapse each channel's HxW plane to its mean.
#include "backend/cpu/cpu_backend.h"

namespace vknn {
    namespace {

        struct GlobalAvgPoolCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X  = ctx.t(node.inputs[0]);
                RtTensor       &Y  = ctx.t(node.outputs[0]);
                NCHW            x  = NCHW::from(X.shape);
                float          *y  = cpu::allocOut(Y, {x.n, x.c, 1, 1});
                const float    *xd = X.host.f32();
                int64_t         hw = x.h * x.w;
                for (int64_t n = 0; n < x.n; ++n)
                {
                    for (int64_t c = 0; c < x.c; ++c)
                    {
                        const float *p = xd + (n * x.c + c) * hw;
                        double       s = 0;
                        for (int64_t i = 0; i < hw; ++i)
                        {
                            s += p[i];
                        }
                        y[n * x.c + c] = (float) (s / hw);
                    }
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::kGlobalAvgPool, GlobalAvgPoolCpu);
} // namespace vknn
