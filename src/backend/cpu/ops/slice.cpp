// ONNX Slice (generic N-D): extracts a strided sub-region of X. `starts`/`ends` are required;
// `axes`/`steps` are optional. Each list is read either from a node attribute (opset < 10) or from
// initializer inputs[1..4] (opset 10+); readParamList resolves whichever form is present. Only the axes named
// in `starts` are sliced — every other axis is copied whole. Handles both fp32 and int64 tensors
// (the int64 path serves const-folded shape arithmetic feeding downstream Slice/Reshape bounds).
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <algorithm>

namespace vknn {
    namespace {
        struct SliceCpu: CpuOp {
            /// Resolve one parameter list (`starts`/`ends`/`axes`/`steps`) from whichever opset form
            /// carries it: the attribute `attr` (opset < 10) takes precedence, else the int64
            /// initializer at inputs[`idx`] (opset 10+). Returns empty when neither is provided,
            /// which the caller treats as the ONNX default (all axes / unit step). Reads are bounded
            /// by pwCoreInputs: inputs appended past it are fused-unit operands, and misreading one
            /// as `steps` would silently stride the slice by a float's bit pattern.
            static std::vector<int64_t> readParamList(const Node &n, ExecContext &ctx, const char *attr, int idx) {
                const auto &a = n.attr.getints(attr);
                if (!a.empty())
                {
                    return a;
                }
                if (idx < (int) pwCoreInputs(n) && n.inputs[idx] != kNoTensor)
                {
                    const RtTensor &t = ctx.t(n.inputs[idx]);
                    return std::vector<int64_t>(t.host.i64(), t.host.i64() + t.elems());
                }
                return {};
            }
            void run(const Node &node, ExecContext &ctx) override {
                if (node.attr.has("view_stride"))
                {
                    cpu::runViewGather(node, ctx);
                    return; // folded movement chain: the composed map replaces starts/steps entirely
                }
                const RtTensor      &X      = ctx.t(node.inputs[0]);
                RtTensor            &Y      = ctx.t(node.outputs[0]);
                int                  rank   = (int) X.shape.size();
                auto                 starts = readParamList(node, ctx, "starts", 1), ends = readParamList(node, ctx, "ends", 2);
                auto                 axes = readParamList(node, ctx, "axes", 3), steps = readParamList(node, ctx, "steps", 4);
                // Per-axis begin/step default to a whole-axis copy (start 0, step 1); `out` starts as
                // X's shape and is overwritten only on the sliced axes. The k-th entry of each list
                // refers to axis `axes[k]` (or axis k when `axes` is absent, per ONNX default).
                std::vector<int64_t> begin(rank, 0), step(rank, 1);
                Shape                out = X.shape;
                for (size_t k = 0; k < starts.size() && k < ends.size(); ++k)
                {
                    int ax = (int) ((axes.empty() || k >= axes.size()) ? (int64_t) k : axes[k]);
                    if (ax < 0)
                    {
                        ax += rank; // negative axis counts from the end
                    }
                    if (ax < 0 || ax >= rank)
                    {
                        continue; // out-of-range axis: leave it a whole-axis copy
                    }
                    // Negative start/end index from the end of the axis, then clamp to [0, dim] so an
                    // over-large bound (e.g. INT_MAX, the ONNX "to the end" idiom) saturates instead of
                    // reading past the axis. `steps` defaults to 1 when the list is shorter than `starts`.
                    int64_t dim = X.shape[ax], sp = k < steps.size() ? steps[k] : 1;
                    int64_t st = starts[k] < 0 ? starts[k] + dim : starts[k];
                    int64_t en = ends[k] < 0 ? ends[k] + dim : ends[k];
                    st         = std::max<int64_t>(0, std::min(st, dim));
                    en         = std::max<int64_t>(0, std::min(en, dim));
                    begin[ax]  = st;
                    step[ax]   = sp;
                    // Output length = ceil((en - st) / sp): the count of indices st, st+sp, ... below en.
                    // Only positive steps are supported here; a non-positive step yields an empty axis.
                    out[ax]    = sp > 0 ? std::max<int64_t>(0, (en - st + sp - 1) / sp) : 0;
                }
                // Row-major (C-contiguous) strides for the source and the sliced output: stride[i] is
                // the product of the dims after axis i, built right to left. They differ because the
                // output's sliced axes are shorter, so a single flat output index maps to a scattered
                // (strided) source location rather than a contiguous copy.
                std::vector<int64_t> inStride(rank, 1), outStride(rank, 1);
                for (int i = rank - 2; i >= 0; --i)
                {
                    inStride[i]  = inStride[i + 1] * X.shape[i + 1];
                    outStride[i] = outStride[i + 1] * out[i + 1];
                }
                int64_t        elems = numElements(out);
                bool           i64   = X.dtype == DType::Int64;
                const float   *xf    = i64 ? nullptr : X.host.f32();
                const int64_t *xi    = i64 ? X.host.i64() : nullptr;
                float         *yf    = i64 ? nullptr : cpu::allocOut(Y, out);
                int64_t       *yi    = i64 ? cpu::allocOutI64(Y, out) : nullptr;
                // Walk the output in flat row-major order. Decode each linear index `oi` into its
                // per-axis output coordinate `c` (via `outStride`), map it to the source coordinate
                // `begin[i] + c*step[i]`, and re-linearize through `inStride` to the source offset `inf`.
                for (int64_t oi = 0; oi < elems; ++oi)
                {
                    int64_t rem = oi, inf = 0;
                    for (int i = 0; i < rank; ++i)
                    {
                        int64_t c = rem / outStride[i];
                        rem %= outStride[i];
                        inf += (begin[i] + c * step[i]) * inStride[i];
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
    VKNN_REGISTER_CPU_OP(OpType::Slice, SliceCpu);
} // namespace vknn
