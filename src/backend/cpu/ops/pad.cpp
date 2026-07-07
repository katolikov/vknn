// ONNX Pad, generic N-D, modes "constant" (default), "edge", and "reflect". Every output element is
// mapped back to a source coordinate per axis: an out-of-input coordinate is filled with `cval` under
// "constant", clamped to the nearest edge under "edge", or mirrored under "reflect". Only the leading
// `begin` half of `pads` shifts coordinates (out coord - begin[axis] = in coord); the trailing `end`
// half only enlarges the output shape, which derives from the RUNTIME input shape plus `pads` (never
// from the static graph desc, so the output follows a runtime shape that diverges from the plan).
// Iteration is over the output in row-major order, so the accumulation is a plain gather (no
// reduction, order-independent).
//
// pads = [begin_0..begin_{rank-1}, end_0..end_{rank-1}] (length 2*rank) from attr "pads" or input[1];
// constant fill value from attr "value" or input[2]. Data is fp32.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <algorithm>

namespace vknn {
    namespace {
        struct PadCpu: CpuOp {
            // Read an int64 vector either from attribute `attr` (opset-10 style) or, if that attribute
            // is absent, from input `idx` (opset-11+ moved `pads` and the pad value to inputs). An
            // input tensor is read from its raw int64 host storage; a missing input yields empty.
            static std::vector<int64_t> readI64(const Node &n, ExecContext &ctx, const char *attr, int idx) {
                const auto &a = n.attr.getints(attr);
                if (!a.empty())
                {
                    return a;
                }
                if (idx < (int) n.inputs.size() && n.inputs[idx] != kNoTensor)
                {
                    const RtTensor &t = ctx.t(n.inputs[idx]);
                    int64_t         m = t.elems();
                    return std::vector<int64_t>(t.host.i64(), t.host.i64() + m);
                }
                return {};
            }
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor      &X    = ctx.t(node.inputs[0]);
                RtTensor            &Y    = ctx.t(node.outputs[0]);
                int                  rank = (int) X.shape.size();
                std::vector<int64_t> pads = readI64(node, ctx, "pads", 1);
                std::string          mode = node.attr.gets("mode", "constant");
                float                cval = node.attr.getf("value", 0.f);
                if (node.inputs.size() > 2 && node.inputs[2] != kNoTensor)
                {
                    cval = ctx.t(node.inputs[2]).host.f32()[0];
                }
                // Output shape = runtime input shape + begin/end pads per axis, mirroring the Pad arm
                // of inferShapes; a pads vector shorter than 2*rank shifts and enlarges nothing. Build
                // row-major (contiguous, last-axis-fastest) strides for both input and output so a
                // per-axis coordinate can be reconstructed from, and folded back into, a flat index.
                const bool           padded = (int64_t) pads.size() >= 2 * (int64_t) rank;
                Shape                out    = X.shape;
                if (padded)
                {
                    for (int i = 0; i < rank; ++i)
                    {
                        out[i] = X.shape[i] + pads[i] + pads[i + rank];
                    }
                }
                std::vector<int64_t> inStride(rank, 1), outStride(rank, 1);
                for (int i = rank - 2; i >= 0; --i)
                {
                    inStride[i]  = inStride[i + 1] * X.shape[i + 1];
                    outStride[i] = outStride[i + 1] * out[i + 1];
                }
                int64_t      elems   = numElements(out);
                float       *y       = cpu::allocOut(Y, out);
                const float *x       = X.host.f32();
                // Map a possibly out-of-range coordinate `i` onto [0, n) by mirroring at both edges
                // WITHOUT repeating the boundary element (ONNX "reflect": ...2,1,[0,1,2,3],2,1...).
                // A degenerate axis (n == 1) has no interior to mirror, so every coordinate folds to 0.
                // Period p = 2*(n-1) is the length of one out-and-back sweep; `i` is reduced into
                // [0, p) with a floored modulo (the ((i%p)+p)%p guards negative i), then a value past
                // the last index (i >= n) reflects to p - i.
                auto         reflect = [](int64_t i, int64_t n) {
                    if (n == 1)
                    {
                        return (int64_t) 0;
                    }
                    int64_t p = 2 * (n - 1);
                    i         = ((i % p) + p) % p;
                    return i < n ? i : p - i;
                };
                // For each output element, decode its per-axis output coordinate `oc`, shift by the
                // axis's `begin` pad to the input coordinate `ic`, and accumulate the flat input offset
                // `inf`. Under "constant" an in-range axis simply advances; the first out-of-range axis
                // sets `oob` and short-circuits so the element takes `cval`. "edge"/"reflect" instead
                // remap `ic` back in-range and never set `oob`. A short `pads` means no shift on any axis.
                for (int64_t oi = 0; oi < elems; ++oi)
                {
                    int64_t rem = oi, inf = 0;
                    bool    oob = false;
                    for (int i = 0; i < rank; ++i)
                    {
                        int64_t oc = rem / outStride[i];
                        rem %= outStride[i];
                        int64_t ic = oc - (padded ? pads[i] : 0);
                        if (ic < 0 || ic >= X.shape[i])
                        {
                            if (mode == "edge")
                            {
                                ic = std::min<int64_t>(std::max<int64_t>(ic, 0), X.shape[i] - 1);
                            } else if (mode == "reflect")
                            {
                                ic = reflect(ic, X.shape[i]);
                            } else
                            {
                                oob = true;
                                break;
                            }
                        }
                        inf += ic * inStride[i];
                    }
                    y[oi] = oob ? cval : x[inf];
                }
            }
        };
    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Pad, PadCpu);
} // namespace vknn
