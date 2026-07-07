// EyeLike: output has the same 2-D shape as the input, with ones on a diagonal (offset by attr `k`)
// and zeros elsewhere. The input values are ignored — only its shape matters. Usually const-folded.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
    namespace {

        struct EyeLikeCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X = ctx.t(node.inputs[0]);
                RtTensor       &Y = ctx.t(node.outputs[0]);
                Shape           s = X.shape;
                // EyeLike is defined only for 2-D tensors. A non-2-D input is degenerate here, so
                // reshape it to a single column [n,1] (n = total element count) rather than reject
                // it; with cols==1 the diagonal test below still produces a well-formed matrix.
                int64_t rows, cols;
                if (s.size() == 2)
                {
                    rows = s[0];
                    cols = s[1];
                } else
                {
                    int64_t n = X.elems();
                    rows      = n;
                    cols      = n > 0 ? 1 : 0; // keep cols==0 for an empty input so the output stays empty
                    s         = {rows, cols};
                }
                // `k` selects which diagonal carries the ones: k=0 is the main diagonal, k>0 shifts
                // it up (into the super-diagonals), k<0 down (sub-diagonals). Default 0.
                int64_t k = node.attr.geti("k", 0);
                float  *y = cpu::allocOut(Y, s);
                // Row-major fill: element (i,j) lives at i*cols + j. The "on" cell of row i is the
                // column j = i + k, i.e. exactly where j - i == k, so a positive k moves that cell
                // rightward (upper diagonal) and a negative k leftward (lower diagonal). Any k that
                // falls outside [-rows+1, cols-1] leaves the matrix all-zero.
                for (int64_t i = 0; i < rows; ++i)
                {
                    for (int64_t j = 0; j < cols; ++j)
                    {
                        y[i * cols + j] = (j - i == k) ? 1.f : 0.f;
                    }
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::EyeLike, EyeLikeCpu);
} // namespace vknn
