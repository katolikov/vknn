// Softmax over the axis range [axis, rank). Standard max-subtract for numerical stability.
#include "backend/cpu/cpu_backend.h"
#include <algorithm>
#include <cmath>

namespace vknn {
    namespace {

        /// CPU reference kernel for Softmax over a contiguous suffix of axes.
        ///
        /// The `axis` attribute (default -1, negative values wrap modulo rank)
        /// splits the tensor into two flattened extents of the row-major layout:
        /// the `inner` block spans axes `[axis, rank)` and is the set over which
        /// the distribution is normalized, while the `outer` block spans the
        /// leading axes `[0, axis)` and indexes independent softmax rows. Each of
        /// the `outer` rows of length `inner` is a contiguous slice of the input.
        ///
        /// For every row the kernel computes the numerically stable softmax
        ///   y[i] = exp(x[i] - max_j x[j]) / sum_j exp(x[j] - max_j x[j])
        /// subtracting the per-row maximum before exponentiating so the largest
        /// exponent argument is 0 and cannot overflow. The denominator is
        /// accumulated in `double` to limit rounding error, then each quotient is
        /// stored back as `float`. Input and output are f32.
        struct SoftmaxCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X    = ctx.t(node.inputs[0]);
                RtTensor       &Y    = ctx.t(node.outputs[0]);
                int64_t         axis = node.attr.geti("axis", -1);
                int64_t         rank = (int64_t) X.shape.size();
                if (axis < 0)
                {
                    axis += rank;
                }
                // Clamp an out-of-range axis (axis < -rank stays negative after the wrap; axis >= rank
                // from a malformed graph) so X.shape[i] stays in bounds and inner never collapses to 0.
                axis = std::max<int64_t>(0, std::min(axis, rank > 0 ? rank - 1 : 0));
                // inner = product of the normalized axes [axis, rank); outer = the
                // count of independent rows formed by the leading axes [0, axis).
                int64_t inner = 1;
                for (int64_t i = axis; i < rank; ++i)
                {
                    inner *= X.shape[i];
                }
                int64_t      outer = X.elems() / inner;
                float       *y     = cpu::allocOut(Y, X.shape);
                const float *x     = X.host.f32();
                for (int64_t o = 0; o < outer; ++o)
                {
                    const float *xr = x + o * inner;
                    float       *yr = y + o * inner;
                    // Pass 1: row maximum for the stability shift.
                    float mx = xr[0];
                    for (int64_t i = 1; i < inner; ++i)
                    {
                        mx = std::max(mx, xr[i]);
                    }
                    // Pass 2: shifted exponentials, kept in Y, summed for the denominator.
                    double sum = 0;
                    for (int64_t i = 0; i < inner; ++i)
                    {
                        yr[i] = std::exp(xr[i] - mx);
                        sum += yr[i];
                    }
                    // Pass 3: normalize each exponential by the row sum.
                    for (int64_t i = 0; i < inner; ++i)
                    {
                        yr[i] = (float) (yr[i] / sum);
                    }
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Softmax, SoftmaxCpu);
} // namespace vknn
