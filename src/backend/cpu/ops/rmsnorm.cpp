// RMSNormalization (root-mean-square norm, the transformer decoder norm). Inputs: X, Scale(gamma).
// Normalizes over the LAST axis (the norm width): per outer row,
//   y = x * rsqrt(mean_j(x[j]^2) + eps) * gamma
// -- no mean subtraction and no bias, unlike LayerNorm. gamma is a 1-D [norm] initializer, eps an
// attribute. This is the numeric oracle the fused Vulkan kernel is diffed against; the graph creates
// RMSNorm only via lowerRMSNorm (never parsed from ONNX). CPU host buffers are canonical NCHW fp32.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <cmath>

namespace vknn {
    namespace {

        struct RMSNormCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X = ctx.t(node.inputs[0]);
                const RtTensor &G = ctx.t(node.inputs[1]);

                RtTensor    &Y     = ctx.t(node.outputs[0]);
                const Shape &shape = X.shape;
                int          rank  = (int) shape.size();
                // The norm region is the last axis (the RMSNorm class always reduces over the feature
                // width). `norm` is that trailing dim; every leading dim collapses into `outer`. gamma
                // indexes by position within a row, so it carries `norm` elements.
                int64_t norm = rank > 0 ? shape[rank - 1] : 1;
                if (norm < 1)
                {
                    norm = 1;
                }
                int64_t outer = X.elems() / norm;
                float   eps   = node.attr.getf("epsilon", 1e-6f); // RMSNorm default 1e-6

                float       *y     = cpu::allocOut(Y, shape);
                const float *x     = X.host.f32();
                const float *gamma = G.host.f32();
                for (int64_t r = 0; r < outer; ++r)
                {
                    const float *xr = x + r * norm;
                    float       *yr = y + r * norm;
                    // Accumulate the sum of squares in double to blunt cancellation over a wide row
                    // (896 elements in Qwen2.5-0.5B); the scale folds back to fp32 to match the device
                    // kernel's fp32-accumulate / fp16-store output precision.
                    double sumsq = 0.0;
                    for (int64_t j = 0; j < norm; ++j)
                    {
                        sumsq += (double) xr[j] * (double) xr[j];
                    }
                    float inv = (float) (1.0 / std::sqrt(sumsq / (double) norm + (double) eps));
                    for (int64_t j = 0; j < norm; ++j)
                    {
                        // Scale by the reciprocal RMS, then apply the per-position gain gamma[j]
                        // (broadcast across every outer row).
                        yr[j] = xr[j] * inv * gamma[j];
                    }
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::RMSNorm, RMSNormCpu);
} // namespace vknn
