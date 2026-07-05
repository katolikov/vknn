// ONNX Shape: emit the input's dims as a 1-D int64 vector. The output has rank 1 and length equal
// to the input's rank; element i is the size of input axis i. Metadata-only: the result depends
// solely on `X.shape` (never on X's element values or dtype), so a rank-0 scalar yields an empty
// vector. This kernel returns the full shape (no `start`/`end` slicing attributes).
#include "backend/cpu/cpu_backend.h"

namespace vknn {
    namespace {

        struct ShapeCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X = ctx.t(node.inputs[0]);
                RtTensor       &Y = ctx.t(node.outputs[0]);
                int64_t         r = (int64_t) X.shape.size(); // rank = number of output elements
                // Allocate Y as an int64 vector of length r, then copy each dim in axis order.
                int64_t *y = cpu::allocOutI64(Y, {r});
                for (int64_t i = 0; i < r; ++i)
                {
                    y[i] = X.shape[i];
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Shape, ShapeCpu);
} // namespace vknn
