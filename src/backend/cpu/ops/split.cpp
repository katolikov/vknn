// Split a tensor along an axis into N outputs (generic N-D, dtype-agnostic). Segment sizes come
// from the `split` attribute, else from the optional int64 `split` input[1] (opset-13 form), else
// (when neither is present) an equal division of the runtime axis extent — never from the static
// graph desc, so each output's shape follows the runtime input shape.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        struct SplitCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X    = ctx.t(node.inputs[0]);
                int             rank = (int) X.shape.size();
                int64_t         axis = node.attr.geti("axis", 0);
                if (axis < 0)
                {
                    axis += rank;
                }
                int64_t              nout = (int64_t) node.outputs.size();
                std::vector<int64_t> sp   = node.attr.getints("split");
                if (sp.empty() && pwCoreInputs(node) > 1 && node.inputs[1] != kNoTensor)
                {
                    const RtTensor &S = ctx.t(node.inputs[1]);
                    sp.assign(S.host.i64(), S.host.i64() + S.elems());
                }
                // Collapse the input to a 3-D view [outer, X.shape[axis], inner] by folding every
                // dim before the split axis into `outer` and every dim after it into `inner`. The
                // split then partitions only the middle (axis) dimension; outer and inner are shared
                // by all outputs, so a copy is a contiguous run of `inner` elements per (outer, axis).
                int64_t outer = 1, inner = 1;
                for (int i = 0; i < axis; ++i)
                {
                    outer *= X.shape[i];
                }
                for (int i = (int) axis + 1; i < rank; ++i)
                {
                    inner *= X.shape[i];
                }
                // Split copies raw elements, so only the element WIDTH matters: int64 outputs are moved
                // through the i64 view, everything else (fp32 and any type aliased to it) through f32.
                bool    i64 = X.dtype == DType::Int64;
                int64_t off = 0; // running start of output k along the split axis; += seg after each output
                for (int64_t k = 0; k < nout; ++k)
                {
                    // A kNoTensor output slot has no destination tensor; skip it before `off` advances,
                    // so it consumes no span of the split axis (unused parts must have segment size 0).
                    if (node.outputs[k] == kNoTensor)
                    {
                        continue;
                    }
                    RtTensor &Y = ctx.t(node.outputs[k]);
                    // This output's extent along the split axis: its `split` entry, or an equal share
                    // of the runtime axis. The output shape is the runtime input shape with the split
                    // axis swapped for that extent.
                    int64_t seg = k < (int64_t) sp.size() ? sp[k] : X.shape[axis] / nout;
                    Shape   os  = X.shape;
                    os[axis]    = seg;
                    float   *yf = i64 ? nullptr : cpu::allocOut(Y, os);
                    int64_t *yi = i64 ? cpu::allocOutI64(Y, os) : nullptr;
                    // For each outer slab `o` and each position `s` within this output's segment, copy
                    // the contiguous `inner`-length row. Source axis index is `off + s` (this output's
                    // slice of the input axis), so the input is strided by its full X.shape[axis]; the
                    // output is dense, strided by its own `seg`. Row-major flat offsets = coord * inner.
                    for (int64_t o = 0; o < outer; ++o)
                    {
                        for (int64_t s = 0; s < seg; ++s)
                        {
                            int64_t srcBase = (o * X.shape[axis] + off + s) * inner;
                            int64_t dstBase = (o * seg + s) * inner;
                            if (i64)
                            {
                                const int64_t *x = X.host.i64();
                                for (int64_t j = 0; j < inner; ++j)
                                {
                                    yi[dstBase + j] = x[srcBase + j];
                                }
                            } else
                            {
                                const float *x = X.host.f32();
                                for (int64_t j = 0; j < inner; ++j)
                                {
                                    yf[dstBase + j] = x[srcBase + j];
                                }
                            }
                        }
                    }
                    off += seg;
                }
            }
        };
    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Split, SplitCpu);
} // namespace vknn
