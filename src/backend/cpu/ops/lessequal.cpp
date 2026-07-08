// ONNX LessOrEqual (A <= B -> 1.0/0.0) with NumPy-style broadcasting. Output is canonical fp32
// (1.0/0.0) so it can feed a downstream Where over fp32 tensors, exactly like Equal. Inputs may be
// fp32 or int64 (comparisons often run on int64 shape tensors); each operand is read through its
// own dtype so the comparison is exact.
#include "backend/cpu/broadcast.h"
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <algorithm>

namespace vknn {
    namespace {

        /// CPU reference for ONNX LessOrEqual: elementwise `A <= B` with NumPy broadcasting,
        /// emitting the canonical fp32 mask (1.0 for true, 0.0 for false).
        struct LessEqualCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &A  = ctx.t(node.inputs[0]);
                const RtTensor &B  = ctx.t(node.inputs[1]);
                RtTensor       &Y  = ctx.t(node.outputs[0]);
                const Shape    &sa = A.shape, &sb = B.shape;
                // Broadcasting aligns shapes at the trailing axis, so the result rank is the larger of
                // the two and the shorter operand is treated as if left-padded with size-1 axes.
                size_t          rank = std::max(sa.size(), sb.size());
                Shape           out(rank, 1);
                // Size of axis `i` (in the common `rank`-axis frame) for shape `s`: axes ahead of `s`'s
                // first real axis are the implicit leading 1s that padding introduces.
                auto            dimOf = [&](const Shape &s, size_t i) -> int64_t {
                    size_t off = rank - s.size();
                    return i < off ? 1 : s[i - off];
                };
                for (size_t i = 0; i < rank; ++i)
                {
                    int64_t da = dimOf(sa, i), db = dimOf(sb, i);
                    out[i]     = (da == 0 || db == 0) ? 0 : std::max(da, db); // a 0 dim broadcasts to 0 (NumPy), never to 1
                }
                int64_t              n = cpu::elemCount(out); // a rank-0 scalar result carries its one element
                // Per-operand broadcast strides in the common frame, built by a right-to-left
                // row-major scan: sA/sB accumulate each operand's own row-major stride, while a size-1
                // (broadcast) axis is pinned to stride 0 so every output index along it rereads the
                // single source element.
                std::vector<int64_t> oa(rank), ob(rank);
                int64_t              sA = 1, sB = 1;
                for (int i = (int) rank - 1; i >= 0; --i)
                {
                    oa[i] = (dimOf(sa, i) == 1) ? 0 : sA;
                    ob[i] = (dimOf(sb, i) == 1) ? 0 : sB;
                    sA *= dimOf(sa, i);
                    sB *= dimOf(sb, i);
                }
                // Read element `i` widened to double: only Int64 goes through the integer view; every
                // other accepted dtype (Float32, Int32) is read as fp32 so the operands compare on a
                // single ordered scale.
                auto val = [](const RtTensor &T, int64_t i) -> double {
                    return T.dtype == DType::Int64 ? (double) T.host.i64()[i] : (double) T.host.f32()[i];
                };
                float *y = cpu::allocOut(Y, out); // canonical fp32 output (1.0 / 0.0)
                // Walk the output in row-major order, carrying each operand's broadcast source offset
                // through the zero-collapsing strides oa/ob by an odometer carry.
                cpu::BroadcastWalk w(out, {oa.data(), ob.data()});
                w.seek(0);
                for (int64_t lin = 0; lin < n; ++lin, w.next())
                {
                    y[lin] = (val(A, w.offset(0)) <= val(B, w.offset(1))) ? 1.0f : 0.0f;
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::LessEqual, LessEqualCpu);
} // namespace vknn
