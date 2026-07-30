// ONNX Greater (A > B -> 1.0/0.0) with NumPy-style broadcasting. Output is canonical fp32
// (1.0/0.0) so it can feed a downstream Where over fp32 tensors, exactly like Equal. Inputs may be
// fp32 or int64 (comparisons often run on int64 shape tensors); each operand is read through its
// own dtype so the comparison is exact.
#include "backend/cpu/broadcast.h"
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <algorithm>

namespace vknn {
    namespace {

        struct GreaterCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &A  = ctx.t(node.inputs[0]);
                const RtTensor &B  = ctx.t(node.inputs[1]);
                RtTensor       &Y  = ctx.t(node.outputs[0]);
                const Shape    &sa = A.shape, &sb = B.shape;
                size_t          rank = std::max(sa.size(), sb.size());
                Shape           out(rank, 1);
                auto            dimOf = [&](const Shape &s, size_t i) -> int64_t {
                    size_t off = rank - s.size();
                    return i < off ? 1 : s[i - off];
                };
                for (size_t i = 0; i < rank; ++i)
                {
                    int64_t da = dimOf(sa, i), db = dimOf(sb, i);
                    out[i] = (da == 0 || db == 0) ? 0 : std::max(da, db); // a 0 dim broadcasts to 0 (NumPy), never to 1
                }
                int64_t n = cpu::elemCount(out); // a rank-0 scalar result carries its one element
                // Per-operand strides into the *original* (un-broadcast) buffers, indexed by output
                // axis. Built right-to-left so each stride is the product of the trailing original
                // dims. A broadcast axis (original extent 1) gets stride 0 so every output index on
                // that axis re-reads the same element, which is exactly NumPy broadcasting.
                std::vector<int64_t> oa(rank), ob(rank);
                int64_t              sA = 1, sB = 1;
                for (int i = (int) rank - 1; i >= 0; --i)
                {
                    oa[i] = (dimOf(sa, i) == 1) ? 0 : sA;
                    ob[i] = (dimOf(sb, i) == 1) ? 0 : sB;
                    sA *= dimOf(sa, i);
                    sB *= dimOf(sb, i);
                }
                // Read an operand element as double: Int64 through the integer view, any other dtype
                // through the fp32 view. Widening to double keeps the comparison exact across the
                // int64 range that shape tensors use without a separate integer compare path.
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
                    y[lin] = (val(A, w.offset(0)) > val(B, w.offset(1))) ? 1.0f : 0.0f;
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Greater, GreaterCpu);
} // namespace vknn
