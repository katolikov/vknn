// Elementwise binary family (Mul/Sub/Div/Max/Min/Pow) with NumPy-style broadcasting.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <algorithm>
#include <cmath>

namespace vknn {
    namespace {

        // Scalar float kernel for one broadcast element pair. Add is the fall-through default so any
        // unlisted BinaryType degrades to addition rather than an undefined value; Div follows IEEE-754
        // (x/0 yields +/-inf or NaN), matching ONNX's float-division semantics.
        static float binary(float a, float b, BinaryType op) {
            switch (op)
            {
                case BinaryType::Mul:
                    return a * b;
                case BinaryType::Sub:
                    return a - b;
                case BinaryType::Div:
                    return a / b;
                case BinaryType::Max:
                    return std::max(a, b);
                case BinaryType::Min:
                    return std::min(a, b);
                case BinaryType::Pow:
                    return std::pow(a, b);
                default:
                    break;
            }
            return a + b;
        }

        struct BinaryCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &A  = ctx.t(node.inputs[0]);
                const RtTensor &B  = ctx.t(node.inputs[1]);
                RtTensor       &Y  = ctx.t(node.outputs[0]);
                const Shape    &sa = A.shape, &sb = B.shape;
                size_t          rank = std::max(sa.size(), sb.size());
                Shape           out(rank, 1);
                // NumPy broadcasting right-aligns shapes: an operand of lower rank is padded on the
                // LEFT with size-1 axes. dimOf reads operand `s`'s extent at output axis `i`, returning
                // 1 for the padded prefix (`i < off`) so those axes broadcast freely.
                auto            dimOf = [&](const Shape &s, size_t i) -> int64_t {
                    size_t off = rank - s.size();
                    return i < off ? 1 : s[i - off];
                };
                for (size_t i = 0; i < rank; ++i)
                {
                    int64_t da = dimOf(sa, i), db = dimOf(sb, i);
                    out[i]     = (da == 0 || db == 0) ? 0 : std::max(da, db); // a 0 dim broadcasts to 0 (NumPy), never to 1
                }
                int64_t n       = cpu::elemCount(out); // a rank-0 scalar result carries its one element
                // Per-operand broadcast strides (row-major, built back-to-front). A stride of 0 on a
                // broadcast axis (operand extent 1 where the output extent is larger) makes every output
                // index along that axis map to the same source element, i.e. the operand is repeated.
                // Real strides accumulate the operand's OWN extents, so they address its dense storage.
                auto    strides = [&](std::vector<int64_t> &oa, std::vector<int64_t> &ob) {
                    int64_t sA = 1, sB = 1;
                    for (int i = (int) rank - 1; i >= 0; --i)
                    {
                        oa[i] = (dimOf(sa, i) == 1) ? 0 : sA;
                        ob[i] = (dimOf(sb, i) == 1) ? 0 : sB;
                        sA *= dimOf(sa, i);
                        sB *= dimOf(sb, i);
                    }
                };
                // Map a flat output offset `lin` to the source offsets `ia`/`ib`. Recovering each axis
                // coordinate `id` costs a suffix-product of trailing output extents (`stride`); dotting
                // those coordinates with the broadcast strides gives the operand element to read.
                auto idx = [&](const std::vector<int64_t> &oa, const std::vector<int64_t> &ob, int64_t lin, int64_t &ia, int64_t &ib) {
                    ia = ib = 0;
                    for (size_t d = 0; d < rank; ++d)
                    {
                        int64_t stride = 1;
                        for (size_t e = d + 1; e < rank; ++e)
                        {
                            stride *= out[e];
                        }
                        int64_t id = (lin / stride) % out[d];
                        ia += id * oa[d];
                        ib += id * ob[d];
                    }
                };
                // Shape arithmetic is int64 (Shape/Gather feed Add/Div/Mul to compute slice/reshape bounds).
                // Compute it in int64 so const-folding stays exact — reading those bytes as float corrupts
                // them.
                if (A.dtype == DType::Int64 || B.dtype == DType::Int64)
                {
                    int64_t             *y = cpu::allocOutI64(Y, out);
                    std::vector<int64_t> oa(rank), ob(rank);
                    strides(oa, ob);
                    // Read either operand as int64: a genuine Int64 tensor directly, a float tensor
                    // truncated toward zero. This lets an int64 shape operand mix with a float sibling
                    // while keeping the result in the exact integer domain.
                    auto val = [](const RtTensor &T, int64_t i) {
                        return T.dtype == DType::Int64 ? T.host.i64()[i] : (int64_t) T.host.f32()[i];
                    };
                    for (int64_t lin = 0; lin < n; ++lin)
                    {
                        int64_t ia, ib;
                        idx(oa, ob, lin, ia, ib);
                        int64_t av = val(A, ia), bv = val(B, ib);
                        switch ((BinaryType) node.subOp)
                        {
                            case BinaryType::Mul:
                                y[lin] = av * bv;
                                break;
                            case BinaryType::Sub:
                                y[lin] = av - bv;
                                break;
                            case BinaryType::Div:
                                // Integer division: guard the divisor so a zero yields 0 rather than a
                                // hardware trap (the float path relies on IEEE inf/NaN instead).
                                y[lin] = bv ? av / bv : 0;
                                break;
                            case BinaryType::Max:
                                y[lin] = std::max(av, bv);
                                break;
                            case BinaryType::Min:
                                y[lin] = std::min(av, bv);
                                break;
                            default:
                                y[lin] = av + bv;
                                break;
                        }
                    }
                    return;
                }
                // Float fast path: same broadcast-stride and unravel arithmetic as the strides/idx
                // lambdas above, inlined into the element loop so the hot case avoids a per-element
                // std::function-style indirection. Keep the two in step when editing either.
                float               *y = cpu::allocOut(Y, out);
                const float         *a = A.host.f32();
                const float         *b = B.host.f32();
                std::vector<int64_t> oa(rank), ob(rank);
                int64_t              sA = 1, sB = 1;
                for (int i = (int) rank - 1; i >= 0; --i)
                {
                    oa[i] = (dimOf(sa, i) == 1) ? 0 : sA;
                    ob[i] = (dimOf(sb, i) == 1) ? 0 : sB;
                    sA *= dimOf(sa, i);
                    sB *= dimOf(sb, i);
                }
                for (int64_t lin = 0; lin < n; ++lin)
                {
                    int64_t ia = 0, ib = 0;
                    for (size_t d = 0; d < rank; ++d)
                    {
                        int64_t stride = 1;
                        for (size_t e = d + 1; e < rank; ++e)
                        {
                            stride *= out[e];
                        }
                        int64_t id = (lin / stride) % out[d];
                        ia += id * oa[d];
                        ib += id * ob[d];
                    }
                    y[lin] = binary(a[ia], b[ib], (BinaryType) node.subOp);
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Binary, BinaryCpu);
} // namespace vknn
