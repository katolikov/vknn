// Einsum CPU reference for the three equations the YoNoSplat encoder uses:
//   "i,j->ij"            outer product (RoPE freq table = positions (x) inv_freq)   [102x]
//   "...ab,...b->...a"   batched mat-vec (intrinsics_inv @ ray coords)              [2x]
//   "bij,bnjk->bnik"     batched matmul w/ broadcast on n (SE3 pose transform)      [1x]
// A general N-operand einsum isn't needed; these three cover the model.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <string>

namespace vknn {
    namespace {

        /// Drop ASCII spaces and tabs so an equation is matched by its label content alone, letting
        /// "b i j , b n j k -> b n i k" compare equal to the canonical "bij,bnjk->bnik" below.
        static std::string stripw(const std::string &s) {
            std::string r;
            for (char c: s)
            {
                if (c != ' ' && c != '\t')
                {
                    r += c;
                }
            }
            return r;
        }

        /// @brief Reference Einsum for the three equations the YoNoSplat encoder emits (see file head).
        ///
        /// Both operands are read as dense row-major fp32; the equation string (whitespace-stripped)
        /// selects a hand-written index contraction. Any label letter that appears in both inputs but
        /// not the output is a summed (contracted) axis; the ellipsis "..." denotes leading batch axes
        /// carried through verbatim. Unrecognized equations fall through to an identity copy so the
        /// graph stays runnable.
        struct EinsumCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                std::string     eq = stripw(node.attr.gets("equation", ""));
                const RtTensor &A  = ctx.t(node.inputs[0]);
                const RtTensor &B  = ctx.t(node.inputs[1]);
                RtTensor       &Y  = ctx.t(node.outputs[0]);
                const float    *a  = A.host.f32();
                const float    *b  = B.host.f32();

                if (eq == "i,j->ij")
                {
                    // Outer product of two vectors: no axis is shared, so nothing is summed and the
                    // result is the full I-by-J grid y[i,j] = a[i]*b[j] in row-major order.
                    int64_t I = A.elems(), J = B.elems();
                    float  *y = cpu::allocOut(Y, {I, J});
                    for (int64_t i = 0; i < I; ++i)
                    {
                        for (int64_t j = 0; j < J; ++j)
                        {
                            y[i * J + j] = a[i] * b[j];
                        }
                    }
                    return;
                }
                if (eq == "...ab,...b->...a")
                {
                    // Batched matrix-times-vector: A is [...,a,b], B is [...,b], the shared label b is
                    // contracted, and the output is [...,a]. The trailing two axes of A form each matrix
                    // (aN rows x bN cols); the batch is the flattened product of all leading axes.
                    const Shape &as    = A.shape;
                    int          aRank = (int) as.size();
                    int64_t      aN = as[aRank - 2], bN = as[aRank - 1];
                    int64_t      batch = 1;
                    for (int k = 0; k < aRank - 2; ++k)
                    {
                        batch *= as[k];
                    }
                    // B's own batch size (all axes but the contracted last one). When it is 1 the single
                    // vector is broadcast across every matrix; otherwise bi wraps modulo bBatch so a
                    // shorter B batch tiles over A's, matching NumPy einsum broadcasting.
                    int64_t bBatch = 1;
                    for (int k = 0; k + 1 < (int) B.shape.size(); ++k)
                    {
                        bBatch *= B.shape[k];
                    }
                    Shape  out(as.begin(), as.end() - 1); // [..., a]
                    float *y = cpu::allocOut(Y, out);
                    for (int64_t bi = 0; bi < batch; ++bi)
                    {
                        // Row-major bases: matrix bi occupies aN*bN floats, its vector bN floats, its
                        // result aN floats.
                        const float *Ap = a + bi * aN * bN;
                        const float *Bp = b + (bBatch == 1 ? 0 : bi % bBatch) * bN;
                        float       *Yp = y + bi * aN;
                        for (int64_t ii = 0; ii < aN; ++ii)
                        {
                            // Dot row ii of the matrix with the vector, summing over the contracted axis b.
                            float s = 0;
                            for (int64_t jj = 0; jj < bN; ++jj)
                            {
                                s += Ap[ii * bN + jj] * Bp[jj];
                            }
                            Yp[ii] = s;
                        }
                    }
                    return;
                }
                if (eq == "bij,bnjk->bnik")
                {
                    // Batched matmul with an extra output axis n on the right operand only: A is [b,i,j],
                    // B is [b,n,j,k], j is contracted, and the result is [b,n,i,k]. A carries no n axis,
                    // so the same [i,j] matrix is reused (broadcast) across all N slices of that batch.
                    const Shape &as = A.shape;
                    const Shape &bs = B.shape;
                    int64_t      Bb = as[0], I = as[1], J = as[2], N = bs[1], K = bs[3];
                    float       *y = cpu::allocOut(Y, {Bb, N, I, K});
                    for (int64_t bb = 0; bb < Bb; ++bb)
                    {
                        for (int64_t n = 0; n < N; ++n)
                        {
                            for (int64_t i = 0; i < I; ++i)
                            {
                                for (int64_t k = 0; k < K; ++k)
                                {
                                    // Contract over j: A[bb,i,j] indexes without n (broadcast), B[bb,n,j,k]
                                    // and Y[bb,n,i,k] use the standard row-major offset of each dense shape.
                                    float s = 0;
                                    for (int64_t j = 0; j < J; ++j)
                                    {
                                        s += a[(bb * I + i) * J + j] * b[((bb * N + n) * J + j) * K + k];
                                    }
                                    y[((bb * N + n) * I + i) * K + k] = s;
                                }
                            }
                        }
                    }
                    return;
                }
                // Unhandled equation: pass input through (keeps the graph runnable; not hit by this model).
                int64_t n = A.elems();
                float  *y = cpu::allocOut(Y, A.shape);
                for (int64_t i = 0; i < n; ++i)
                {
                    y[i] = a[i];
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Einsum, EinsumCpu);
} // namespace vknn
