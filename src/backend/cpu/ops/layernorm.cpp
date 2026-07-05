// LayerNormalization (ONNX opset 17). Inputs: X, Scale(gamma), B(bias, optional).
// Normalizes over the axes from `axis` to the end: per outer row, y =
// (x-mean)/sqrt(var+eps)*gamma+beta. CPU reference (canonical NCHW fp32 host buffers); validated
// against by the host tests.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <cmath>

namespace vknn {
    namespace {

        struct LayerNormCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X       = ctx.t(node.inputs[0]);
                const RtTensor &G       = ctx.t(node.inputs[1]);
                bool            hasBeta = pwCoreInputs(node) > 2 && node.inputs[2] != kNoTensor;
                const float    *beta    = hasBeta ? ctx.t(node.inputs[2]).host.f32() : nullptr;

                RtTensor    &Y     = ctx.t(node.outputs[0]);
                const Shape &shape = X.shape;
                int          rank  = (int) shape.size();
                // `axis` (ONNX default -1) marks where the normalization region begins; a negative value
                // counts from the end, then clamps to 0 so it stays in range for degenerate inputs.
                int64_t      axis  = node.attr.geti("axis", -1);
                if (axis < 0)
                {
                    axis += rank;
                }
                if (axis < 0)
                {
                    axis = 0;
                }

                // Split the row-major tensor into `outer` independent rows of `norm` contiguous elements:
                // `norm` is the product of the dims from `axis` to the end (the region reduced over), and
                // every leading dim collapses into `outer`. gamma/beta are indexed by position within a
                // row, so they carry `norm` elements each.
                int64_t norm = 1;
                for (int k = (int) axis; k < rank; ++k)
                {
                    norm *= shape[k];
                }
                if (norm < 1)
                {
                    norm = 1;
                }
                int64_t outer = X.elems() / norm;
                float   eps   = node.attr.getf("epsilon", 1e-5f); // ONNX default 1e-5

                float       *y     = cpu::allocOut(Y, shape);
                const float *x     = X.host.f32();
                const float *gamma = G.host.f32();
                for (int64_t r = 0; r < outer; ++r)
                {
                    const float *xr   = x + r * norm;
                    float       *yr   = y + r * norm;
                    // Accumulate the row mean and variance in double precision to blunt cancellation over
                    // long normalization regions; the final scale is folded back to fp32 to match the
                    // device kernel's output precision.
                    double       mean = 0.0;
                    for (int64_t j = 0; j < norm; ++j)
                    {
                        mean += xr[j];
                    }
                    mean /= (double) norm;
                    double var = 0.0;
                    for (int64_t j = 0; j < norm; ++j)
                    {
                        double c = xr[j] - mean;
                        var += c * c;
                    }
                    var /= (double) norm; // population (biased) variance: divide by N, not N-1
                    float inv = (float) (1.0 / std::sqrt(var + eps));
                    for (int64_t j = 0; j < norm; ++j)
                    {
                        // Standardize, then apply the per-position affine: gamma/beta index by column j
                        // within the row (broadcast across every outer row).
                        float v = ((float) (xr[j] - mean)) * inv * gamma[j];
                        if (hasBeta)
                        {
                            v += beta[j];
                        }
                        yr[j] = v;
                    }
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::LayerNorm, LayerNormCpu);
} // namespace vknn
