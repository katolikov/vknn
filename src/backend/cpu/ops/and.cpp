// ONNX And (elementwise boolean AND) with NumPy-style broadcasting. Both operands are bool, which
// arrives on the canonical host path as fp32 (or int64) and is treated as "true" iff != 0, exactly
// like the cond operand of Where and the outputs of the flat comparison ops. Output is canonical
// fp32 (1.0/0.0) so it can feed a downstream Where/And/Add over fp32 tensors.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <algorithm>

namespace vknn {
    namespace {

        struct AndCpu: CpuOp {
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
                    out[i]     = (da == 0 || db == 0) ? 0 : std::max(da, db); // a 0 dim broadcasts to 0 (NumPy), never to 1
                }
                int64_t              n = cpu::elemCount(out); // a rank-0 scalar result carries its one element
                // Per-axis element strides into each operand, built right-to-left (row-major). A size-1
                // axis gets stride 0 so every output index along that axis re-reads the single source
                // element -- the standard broadcast trick. sA/sB accumulate the true packed stride
                // using the operand's own (unbroadcast) extent via dimOf.
                std::vector<int64_t> oa(rank), ob(rank);
                int64_t              sA = 1, sB = 1;
                for (int i = (int) rank - 1; i >= 0; --i)
                {
                    oa[i] = (dimOf(sa, i) == 1) ? 0 : sA;
                    ob[i] = (dimOf(sb, i) == 1) ? 0 : sB;
                    sA *= dimOf(sa, i);
                    sB *= dimOf(sb, i);
                }
                // Read each operand as bool: true iff != 0. int64 masks stay exact; every other dtype
                // (bool/uint8 materialized on the canonical fp32 path) reads through f32().
                auto isTrue = [](const RtTensor &T, int64_t i) -> bool {
                    return T.dtype == DType::Int64 ? T.host.i64()[i] != 0 : T.host.f32()[i] != 0.0f;
                };
                float *y = cpu::allocOut(Y, out); // canonical fp32 output (1.0 / 0.0)
                for (int64_t lin = 0; lin < n; ++lin)
                {
                    // Unravel the row-major linear output index into per-axis coordinates, then map each
                    // coordinate through the broadcast strides oa/ob to reach the source elements.
                    // `stride` is the trailing product of output extents for axis d, so
                    // (lin / stride) % out[d] recovers that axis's index.
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
                    y[lin] = (isTrue(A, ia) && isTrue(B, ib)) ? 1.0f : 0.0f;
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::And, AndCpu);
} // namespace vknn
