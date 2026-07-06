/// ONNX Flatten: reshape the input into a 2D matrix `[d0, d1]` where the first `axis`
/// dimensions collapse into the row count and the remaining dimensions into the column count,
/// i.e. `d0 = prod(shape[0..axis))`, `d1 = prod(shape[axis..rank))`. This is a pure metadata
/// reshape: elements keep their row-major order and byte values, so no data is moved or recomputed.
#include "backend/cpu/cpu_backend.h"

namespace vknn {
    namespace {

        struct FlattenCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X    = ctx.t(node.inputs[0]);
                RtTensor       &Y    = ctx.t(node.outputs[0]);
                // Default axis is 1 (batch dim stays as rows). ONNX permits axis in [-rank, rank].
                int64_t         axis = node.attr.geti("axis", 1);
                int64_t         rank = (int64_t) X.shape.size();
                // Negative axis counts back from the end, matching Python/ONNX indexing.
                if (axis < 0)
                {
                    axis += rank;
                }
                // Split the extents at `axis`: dims before it multiply into d0 (rows), the rest into
                // d1 (columns). Both start at 1 so the empty-product cases stay correct: axis==0 gives
                // d0==1 (single row over all elements), axis==rank gives d1==1 (single column).
                int64_t d0 = 1, d1 = 1;
                for (int64_t i = 0; i < rank; ++i)
                {
                    (i < axis ? d0 : d1) *= X.shape[i];
                }
                cpu::copyAs(X, Y, {d0, d1});
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Flatten, FlattenCpu);
} // namespace vknn
