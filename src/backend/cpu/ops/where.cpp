// ONNX Where (cond ? X : Y) with full NumPy-style broadcasting over all three inputs. cond is bool/
// uint8 in ONNX but arrives here as fp32 (or int64); it is treated as "true" iff != 0. X and Y are
// the value operands. Output dtype follows X/Y, and the value operands are read in their native
// dtype: the dynamic-shape subgraph runs Where on INT64 shape vectors (e.g.
// Where(Equal(dim,-1), input_shape, target)), where reading int bytes as fp32 would corrupt them.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <algorithm>

namespace vknn {
    namespace {

        struct WhereCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &C   = ctx.t(node.inputs[0]);
                const RtTensor &X   = ctx.t(node.inputs[1]);
                const RtTensor &Yv  = ctx.t(node.inputs[2]);
                RtTensor       &Out = ctx.t(node.outputs[0]);
                const Shape    &sc = C.shape, &sx = X.shape, &sy = Yv.shape;
                size_t          rank = std::max(sc.size(), std::max(sx.size(), sy.size()));
                Shape           out(rank, 1);
                // Right-align the shorter operand: NumPy broadcasting matches axes from the trailing
                // end, so a shape shorter than `rank` reads as an implicit leading run of size-1 dims.
                auto            dimOf = [&](const Shape &s, size_t i) -> int64_t {
                    size_t off = rank - s.size();
                    return i < off ? 1 : s[i - off];
                };
                for (size_t i = 0; i < rank; ++i)
                {
                    int64_t dc = dimOf(sc, i), dx = dimOf(sx, i), dy = dimOf(sy, i);
                    out[i]     = (dc == 0 || dx == 0 || dy == 0) ? 0 : std::max(dc, std::max(dx, dy)); // a 0 dim broadcasts to 0 (NumPy), never to 1
                }
                int64_t              n = cpu::elemCount(out); // a rank-0 scalar result carries its one element
                // Per-axis input strides in row-major (C-contiguous) order, built right to left. A
                // broadcast axis (input dim 1, output dim > 1) gets stride 0 so every output index
                // along it re-reads the single source element; a non-broadcast axis carries the
                // running product of the trailing input dims. sC/sX/sY accumulate that product.
                std::vector<int64_t> oc(rank), ox(rank), oy(rank);
                int64_t              sC = 1, sX = 1, sY = 1;
                for (int i = (int) rank - 1; i >= 0; --i)
                {
                    oc[i] = (dimOf(sc, i) == 1) ? 0 : sC;
                    ox[i] = (dimOf(sx, i) == 1) ? 0 : sX;
                    oy[i] = (dimOf(sy, i) == 1) ? 0 : sY;
                    sC *= dimOf(sc, i);
                    sX *= dimOf(sx, i);
                    sY *= dimOf(sy, i);
                }
                // cond read through its native dtype (bool/uint8 imported as fp32; int64 shape masks possible).
                auto condTrue = [](const RtTensor &T, int64_t i) -> bool {
                    return T.dtype == DType::Int64 ? T.host.i64()[i] != 0 : T.host.f32()[i] != 0.0f;
                };
                // Broadcast-index helper: decodes a flat row-major output index `lin` back into its
                // per-axis coordinate `id` (the output stride along axis d is the product of the
                // trailing output dims), then projects `id` through the zero-collapsing strides
                // oc/ox/oy to yield each operand's source offset under broadcasting.
                auto offsets = [&](int64_t lin, int64_t &ic, int64_t &ix, int64_t &iy) {
                    ic = ix = iy = 0;
                    for (size_t d = 0; d < rank; ++d)
                    {
                        int64_t stride = 1;
                        for (size_t e = d + 1; e < rank; ++e)
                        {
                            stride *= out[e];
                        }
                        int64_t id = (lin / stride) % out[d];
                        ic += id * oc[d];
                        ix += id * ox[d];
                        iy += id * oy[d];
                    }
                };
                // Output type follows the value operands (int64 for the shape-arithmetic Where).
                bool i64 = X.dtype == DType::Int64 && Yv.dtype == DType::Int64;
                if (i64)
                {
                    int64_t       *o = cpu::allocOutI64(Out, out);
                    const int64_t *x = X.host.i64();
                    const int64_t *y = Yv.host.i64();
                    for (int64_t lin = 0; lin < n; ++lin)
                    {
                        int64_t ic, ix, iy;
                        offsets(lin, ic, ix, iy);
                        o[lin] = condTrue(C, ic) ? x[ix] : y[iy];
                    }
                } else
                {
                    float       *o = cpu::allocOut(Out, out);
                    const float *x = X.host.f32();
                    const float *y = Yv.host.f32();
                    for (int64_t lin = 0; lin < n; ++lin)
                    {
                        int64_t ic, ix, iy;
                        offsets(lin, ic, ix, iy);
                        o[lin] = condTrue(C, ic) ? x[ix] : y[iy];
                    }
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Where, WhereCpu);
} // namespace vknn
