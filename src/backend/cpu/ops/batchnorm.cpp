// BatchNormalization in inference (test) mode: per-channel affine normalization with the running
// statistics baked into the model, y = scale*(x-mean)/sqrt(var+eps) + bias. The training-mode outputs
// (running-mean/var updates, saved mean/var) are not produced; this is the frozen-graph form.
// MobileNetV2 ships with BN already folded into the convolutions, so this path serves models where BN
// is left unfolded and provides the reference against which the BN->Conv fold is validated.
#include "backend/cpu/cpu_backend.h"
#include <cmath>

namespace vknn {
    namespace {

        /// @brief Reference BatchNormalization (ONNX, spatial inference form) over a dense NCHW tensor.
        ///
        /// scale/bias/mean/var are per-channel vectors of length C; the same coefficients apply to every
        /// spatial location and every batch item of a channel. The normalization is refactored into a
        /// single fused multiply-add per element by precomputing, once per channel, the affine pair
        /// (a, b) below — matching the folded-Conv reference and halving the per-element work.
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
                // Elements in one channel plane; the innermost loop is a contiguous run over H*W.
                int64_t hw = x.h * x.w;
                for (int64_t n = 0; n < x.n; ++n)
                {
                    for (int64_t c = 0; c < x.c; ++c)
                    {
                        // Fold the normalization into y = a*x + b: a folds scale with the inverse std,
                        // b re-centers with bias. std::sqrt(var+eps) with eps>0 (default 1e-5) keeps the
                        // divisor away from zero for channels with vanishing variance.
                        float a = scale[c] / std::sqrt(var[c] + eps);
                        float b = bias[c] - mean[c] * a;
                        // Base of channel c of batch item n in the dense NCHW layout: (n*C + c)*H*W.
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
    VKNN_REGISTER_CPU_OP(OpType::BatchNorm, BatchNormCpu);
} // namespace vknn
