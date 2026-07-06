/// @file
/// Expand (ONNX): broadcast X to the shape given by the int64 `shape` input.
///
/// The target output is the *numpy broadcast* of X.shape and `shape` (not a copy of `shape`): each
/// output dim is the max of the two right-aligned dims, and either operand may broadcast a size-1
/// dim. Consequently an output dim can exceed both X and `shape` when they disagree, and the output
/// rank can exceed X's rank.
///
/// The kernel is a pure gather with no arithmetic on the values, hence dtype-agnostic (fp32 / int64):
///     out[outCoord] = X[ right-align(outCoord) with per-dim wrap ]
/// A size-1 input dim contributes stride 0, so every output coordinate along that axis reads the same
/// input element (broadcasting); a non-1 input dim indexes normally. Because a broadcast source dim is
/// always 1, the general `outCoord_k % inDim_k` reduces to "stride 0 when inDim==1, else stride acc".
#include "backend/cpu/cpu_backend.h"
#include "import/passes.h" // readI64Param
#include "vknn/op.h"
#include <algorithm>

namespace vknn {
    namespace {

        struct ExpandCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X  = ctx.t(node.inputs[0]);
                RtTensor       &Y  = ctx.t(node.outputs[0]);
                const Shape    &in = X.shape;
                // Target shape lives in inputs[1]. Prefer the constant-folded value (graph initializer)
                // read by readI64Param; fall back to the runtime int64 tensor's host contents when the
                // shape is data-dependent and only materializes at execution time.
                std::vector<int64_t> tgt = readI64Param(*ctx.graph, node, "shape", 1);
                if (tgt.empty() && node.inputs.size() > 1 && node.inputs[1] != kNoTensor)
                {
                    const RtTensor &S = ctx.t(node.inputs[1]);
                    tgt.assign(S.host.i64(), S.host.i64() + S.elems());
                }
                // Broadcasting aligns dims from the trailing (fastest-varying) axis, so the shared rank
                // is the larger of the two ranks and both operands are right-justified into it.
                int rank = (int) std::max(in.size(), tgt.size());
                // pin = X.shape right-aligned into `rank`, missing leading dims implied as 1.
                // out = per-dim numpy broadcast: max of the two aligned dims, where a 1 broadcasts.
                Shape out(rank, 1), pin(rank, 1);
                for (int k = 0; k < (int) in.size(); ++k)
                {
                    pin[rank - (int) in.size() + k] = in[k];
                }
                for (int k = 0; k < rank; ++k)
                {
                    // Aligned target dim for axis k, or 1 for the leading axes `shape` does not cover.
                    int64_t t = (k >= rank - (int) tgt.size()) ? tgt[k - (rank - (int) tgt.size())] : 1;
                    // A negative `shape` entry is treated as 1 so it never shrinks the axis: the output
                    // dim is then driven solely by X (matching numpy/ONNX Expand, which forbids a real
                    // dimension from broadcasting down).
                    out[k]    = std::max<int64_t>(pin[k], t < 0 ? 1 : t);
                }
                // Row-major (C-contiguous) strides for both tensors. `inStride` walks the *aligned*
                // input shape `pin`: a broadcast axis (pin==1) gets stride 0 so it never advances the
                // input offset, otherwise it gets the running product `acc` of trailing input dims.
                std::vector<int64_t> inStride(rank, 0), outStride(rank, 1);
                int64_t              acc = 1;
                for (int k = rank - 1; k >= 0; --k)
                {
                    inStride[k] = (pin[k] == 1) ? 0 : acc;
                    acc *= pin[k];
                }
                // Output strides are the standard product of trailing output dims; the innermost axis
                // keeps its initialized stride of 1.
                for (int k = rank - 2; k >= 0; --k)
                {
                    outStride[k] = outStride[k + 1] * out[k + 1];
                }
                int64_t        elems = numElements(out);
                // Two type lanes only: int64 for shape/index tensors, fp32 for everything else. The
                // gather never touches the values, so no other dtype needs a distinct path here.
                bool           i64   = X.dtype == DType::Int64;
                const float   *xf    = i64 ? nullptr : X.host.f32();
                const int64_t *xi    = i64 ? X.host.i64() : nullptr;
                float         *yf    = i64 ? nullptr : cpu::allocOut(Y, out);
                int64_t       *yi    = i64 ? cpu::allocOutI64(Y, out) : nullptr;
                for (int64_t oi = 0; oi < elems; ++oi)
                {
                    // Decompose the flat output index into per-axis coordinates and simultaneously
                    // fold them into the source offset `inf`. A broadcast axis has inStride 0, so its
                    // coordinate drops out and the shared input element is reused.
                    int64_t rem = oi, inf = 0;
                    for (int k = 0; k < rank; ++k)
                    {
                        int64_t c = rem / outStride[k];
                        rem %= outStride[k];
                        inf += c * inStride[k];
                    }
                    if (i64)
                    {
                        yi[oi] = xi[inf];
                    } else
                    {
                        yf[oi] = xf[inf];
                    }
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Expand, ExpandCpu);
} // namespace vknn
