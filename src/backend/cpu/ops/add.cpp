// Elementwise add with NumPy-style broadcasting. Equal-shape inputs (the residual connections)
// take a NEON fast path; broadcasting falls back to the general index walk.
#include "backend/cpu/broadcast.h"
#include "backend/cpu/cpu_backend.h"
#include "vknn/logging.h"
#include <algorithm>
#if defined(VKNN_ENABLE_NEON) && defined(__ARM_NEON)
#include <arm_neon.h>
#define VKNN_HAS_NEON 1
#endif

namespace vknn {
    namespace {

        struct AddCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &A  = ctx.t(node.inputs[0]);
                const RtTensor &B  = ctx.t(node.inputs[1]);
                RtTensor       &Y  = ctx.t(node.outputs[0]);
                const Shape    &sa = A.shape, &sb = B.shape;

                // int64 path: shape arithmetic (Shape/Gather + Add) const-folded for slice/reshape bounds.
                if (A.dtype == DType::Int64 || B.dtype == DType::Int64)
                {
                    size_t rank = std::max(sa.size(), sb.size());
                    Shape  out(rank, 1);
                    // Right-align the shorter operand: NumPy broadcasting matches axes from the trailing
                    // end, so a shape shorter than `rank` reads as an implicit leading run of size-1 dims.
                    auto   dimOf = [&](const Shape &s, size_t i) -> int64_t {
                        size_t off = rank - s.size();
                        return i < off ? 1 : s[i - off];
                    };
                    for (size_t i = 0; i < rank; ++i)
                    {
                        int64_t da = dimOf(sa, i), db = dimOf(sb, i);
                    out[i]     = (da == 0 || db == 0) ? 0 : std::max(da, db); // a 0 dim broadcasts to 0 (NumPy), never to 1
                    }
                    int64_t  n   = cpu::elemCount(out); // a rank-0 scalar result carries its one element
                    int64_t *y   = cpu::allocOutI64(Y, out);
                    // Read either operand as int64: a float operand (mixed-dtype shape arithmetic) is
                    // truncated toward zero, since this path only sees integral index/bound values.
                    auto     val = [](const RtTensor &T, int64_t i) {
                        return T.dtype == DType::Int64 ? T.host.i64()[i] : (int64_t) T.host.f32()[i];
                    };
                    // Per-axis input strides in row-major (C-contiguous) order, built right to left.
                    // A broadcast axis (input dim 1, output dim > 1) gets stride 0 so every output index
                    // along it re-reads the single source element; a non-broadcast axis carries the
                    // running product of the trailing input dims. sA/sB accumulate that product.
                    std::vector<int64_t> oa(rank), ob(rank);
                    int64_t              sA = 1, sB = 1;
                    for (int i = (int) rank - 1; i >= 0; --i)
                    {
                        oa[i] = (dimOf(sa, i) == 1) ? 0 : sA;
                        ob[i] = (dimOf(sb, i) == 1) ? 0 : sB;
                        sA *= dimOf(sa, i);
                        sB *= dimOf(sb, i);
                    }
                    // Walk the output in flat row-major order, carrying each operand's source offset
                    // through the zero-collapsing strides oa/ob by an odometer carry.
                    cpu::BroadcastWalk w(out, {oa.data(), ob.data()});
                    w.seek(0);
                    for (int64_t lin = 0; lin < n; ++lin, w.next())
                    {
                        y[lin] = val(A, w.offset(0)) + val(B, w.offset(1));
                    }
                    return;
                }

                if (sa == sb)
                { // residual add: same shape, vectorizable
                    int64_t      n = cpu::elemCount(sa); // a rank-0 scalar result carries its one element
                    float       *y = cpu::allocOut(Y, sa);
                    const float *a = A.host.f32();
                    const float *b = B.host.f32();
                    int64_t      i = 0;
#if defined(VKNN_HAS_NEON)
                    // Four lanes per iteration; the scalar loop below finishes the tail of up to 3
                    // elements. Both compute a[i]+b[i] in fp32, so the result is identical either way.
                    for (; i + 4 <= n; i += 4)
                    {
                        vst1q_f32(y + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
                    }
#endif
                    for (; i < n; ++i)
                    {
                        y[i] = a[i] + b[i];
                    }
                    cpu::applyAct(y, n, node.fusedAct, node.actLo, node.actHi); // fused Relu (e.g. ResNet)
                    return;
                }

                // General fp32 broadcast: the shapes differ, so build the broadcast output shape and
                // zero-collapsing per-axis input strides, then gather element pairs (same scheme as the
                // int64 path above).
                size_t rank = std::max(sa.size(), sb.size());
                Shape  out(rank, 1);
                // Right-align the shorter operand: shorter shapes read as leading size-1 dims.
                auto   dimOf = [&](const Shape &s, size_t i) -> int64_t {
                    size_t off = rank - s.size();
                    return i < off ? 1 : s[i - off];
                };
                for (size_t i = 0; i < rank; ++i)
                {
                    int64_t da = dimOf(sa, i), db = dimOf(sb, i);
                    out[i]     = (da == 0 || db == 0) ? 0 : std::max(da, db); // a 0 dim broadcasts to 0 (NumPy), never to 1
                }
                int64_t              n = cpu::elemCount(out); // a rank-0 scalar result carries its one element
                float               *y = cpu::allocOut(Y, out);
                const float         *a = A.host.f32();
                const float         *b = B.host.f32();
                // Row-major input strides, built right to left: stride 0 on a broadcast axis (input dim 1)
                // makes every output coordinate along it re-read the lone source element.
                std::vector<int64_t> oa(rank), ob(rank);
                int64_t              sA = 1, sB = 1;
                for (int i = (int) rank - 1; i >= 0; --i)
                {
                    oa[i] = (dimOf(sa, i) == 1) ? 0 : sA;
                    ob[i] = (dimOf(sb, i) == 1) ? 0 : sB;
                    sA *= dimOf(sa, i);
                    sB *= dimOf(sb, i);
                }
                // Walk the output row-major, projecting each position through oa/ob by odometer carry.
                cpu::BroadcastWalk w(out, {oa.data(), ob.data()});
                w.seek(0);
                for (int64_t lin = 0; lin < n; ++lin, w.next())
                {
                    y[lin] = a[w.offset(0)] + b[w.offset(1)];
                }
                cpu::applyAct(y, n, node.fusedAct, node.actLo, node.actHi); // a broadcast Add carries fusedAct too
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Add, AddCpu);
} // namespace vknn
