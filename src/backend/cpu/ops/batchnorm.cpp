// BatchNormalization (inference form). MobileNetV2 ships with BN already folded into the
// convolutions; this path serves models where BN is not folded and validates the BN->Conv fold.
#include "backend/cpu/cpu_backend.h"
#include <cmath>

namespace vknn {
    namespace {

        struct BatchNormCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X     = ctx.t(node.inputs[0]);
                const float    *scale = ctx.t(node.inputs[1]).host.f32();
                const float    *bias  = ctx.t(node.inputs[2]).host.f32();
                const float    *mean  = ctx.t(node.inputs[3]).host.f32();
                const float    *var   = ctx.t(node.inputs[4]).host.f32();
                float           eps   = node.attr.getf("epsilon", 1e-5f);
                RtTensor       &Y     = ctx.t(node.outputs[0]);
                NCHW            x     = NCHW::from(X.shape);
                float          *y     = cpu::allocOut(Y, X.shape);
                const float    *xd    = X.host.f32();
                int64_t         hw    = x.h * x.w;
                for (int64_t n = 0; n < x.n; ++n)
                {
                    for (int64_t c = 0; c < x.c; ++c)
                    {
                        float        a = scale[c] / std::sqrt(var[c] + eps);
                        float        b = bias[c] - mean[c] * a;
                        const float *p = xd + (n * x.c + c) * hw;
                        float       *q = y + (n * x.c + c) * hw;
                        for (int64_t i = 0; i < hw; ++i)
                        {
                            q[i] = p[i] * a + b;
                        }
                    }
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::kBatchNorm, BatchNormCpu);
} // namespace vknn
