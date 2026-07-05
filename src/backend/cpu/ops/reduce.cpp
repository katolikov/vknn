// Reduce family (Mean/Sum/Max/Min/Prod/L2), generic N-D fp32. `node.subOp` selects the ReduceType.
// The reduced `axes` come from the `axes` attribute, else from a runtime int64 input[1], else (when
// neither is present) every axis. Reduction is always keepdims=false here: the output shape is
// precomputed in the graph desc, so reduced axes are dropped and the remaining ("kept") axes stay in
// input order. Accumulation walks the input once in row-major (flat) order, so the summation order is
// deterministic and observable — do not reorder it.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace vknn {
    namespace {
        struct ReduceCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor      &X    = ctx.t(node.inputs[0]);
                RtTensor            &Y    = ctx.t(node.outputs[0]);
                int                  rank = (int) X.shape.size();
                // Axis source precedence: `axes` attribute, else the optional int64 `axes` input[1]
                // (opset-18 form), else — when neither is supplied — reduce over all axes.
                std::vector<int64_t> axes = node.attr.getints("axes");
                if (axes.empty() && pwCoreInputs(node) > 1 && node.inputs[1] != kNoTensor)
                {
                    const RtTensor &A = ctx.t(node.inputs[1]);
                    axes.assign(A.host.i64(), A.host.i64() + A.elems());
                }
                if (axes.empty())
                {
                    for (int i = 0; i < rank; ++i)
                    {
                        axes.push_back(i);
                    }
                }
                // Normalize into a set of non-negative axis indices; negative axes count from the end.
                // The set also deduplicates any repeated axis so each element is counted exactly once.
                std::set<int> ax;
                for (int64_t a: axes)
                {
                    ax.insert((int) (a < 0 ? a + rank : a));
                }
                ReduceType           op       = (ReduceType) node.subOp;
                Shape                out      = ctx.graph->desc(node.outputs[0]).shape;
                int64_t              outElems = numElements(out), n = X.elems();
                // Row-major strides of the full input tensor, used to decompose a flat index back into
                // per-axis coordinates.
                std::vector<int64_t> inStride(rank, 1);
                for (int i = rank - 2; i >= 0; --i)
                {
                    inStride[i] = inStride[i + 1] * X.shape[i + 1];
                }
                // `kept` = the non-reduced axes, in input order; they form the output layout (keepdims
                // is false). `outStrideK` are the row-major strides of the output over just those axes,
                // so an input coordinate can be mapped straight to its destination element.
                std::vector<int64_t> kept;
                for (int i = 0; i < rank; ++i)
                {
                    if (!ax.count(i))
                    {
                        kept.push_back(i);
                    }
                }
                std::vector<int64_t> outStrideK(kept.size(), 1);
                for (int i = (int) kept.size() - 2; i >= 0; --i)
                {
                    outStrideK[i] = outStrideK[i + 1] * X.shape[kept[i + 1]];
                }
                // Identity element per reduction: -inf/+inf for Max/Min, 1 for Prod, and 0 for the
                // additive ops (Sum, Mean, and L2's sum-of-squares).
                float                init = op == ReduceType::Max  ? -std::numeric_limits<float>::infinity() :
                                            op == ReduceType::Min  ? std::numeric_limits<float>::infinity() :
                                            op == ReduceType::Prod ? 1.f :
                                                                     0.f;
                std::vector<float>   acc(outElems, init);
                std::vector<int64_t> cnt(outElems, 0); // element count per output bin, for the Mean divide
                const float         *x = X.host.f32();
                // Single pass over the flat input. For each element, project its kept-axis coordinates
                // to the output bin `oi` and fold the value in; the reduced axes contribute nothing to
                // `oi`, so every input touching the same bin accumulates there.
                for (int64_t i = 0; i < n; ++i)
                {
                    int64_t rem = i, oi = 0;
                    for (size_t k = 0; k < kept.size(); ++k)
                    {
                        int64_t c = (i / inStride[kept[k]]) % X.shape[kept[k]];
                        oi += c * outStrideK[k];
                    }
                    float v = x[i];
                    if (op == ReduceType::Max)
                    {
                        acc[oi] = std::max(acc[oi], v);
                    } else if (op == ReduceType::Min)
                    {
                        acc[oi] = std::min(acc[oi], v);
                    } else if (op == ReduceType::Prod)
                    {
                        acc[oi] *= v;
                    } else if (op == ReduceType::L2)
                    {
                        acc[oi] += v * v; // sum of squares; sqrt below
                    } else
                    {
                        acc[oi] += v;
                    }
                    cnt[oi]++;
                    (void) rem;
                }
                // Finalize each bin: Mean divides the accumulated sum by its element count (guarded so
                // an empty bin stays at its 0 init rather than dividing by zero); L2 takes the square
                // root of the accumulated sum of squares; all other ops emit the accumulator as-is.
                float *y = cpu::allocOut(Y, out);
                for (int64_t i = 0; i < outElems; ++i)
                {
                    if (op == ReduceType::Mean && cnt[i])
                    {
                        y[i] = acc[i] / cnt[i];
                    } else if (op == ReduceType::L2)
                    {
                        y[i] = std::sqrt(acc[i]);
                    } else
                    {
                        y[i] = acc[i];
                    }
                }
            }
        };
    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Reduce, ReduceCpu);
} // namespace vknn
