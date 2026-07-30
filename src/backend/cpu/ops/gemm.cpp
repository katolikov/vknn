// ONNX Gemm: Y = alpha*op(A)*op(B) + beta*C, where op(X) is X or X^T per transA/transB, A is MxK,
// op(B) is KxN, and Y is MxN. C is an optional bias, broadcast to MxN. The inner K-contraction gets
// a NEON path for the common classifier shape (transB=1, A not transposed); everything else uses the
// scalar fallback.
#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/parallel.h"
#if defined(VKNN_ENABLE_NEON) && defined(__ARM_NEON)
#include <arm_neon.h>
#define VKNN_HAS_NEON 1
#endif

namespace vknn {
    namespace {

        struct GemmCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &A  = ctx.t(node.inputs[0]);
                const RtTensor &Bt = ctx.t(node.inputs[1]);
                // Optional bias C is input[2]. Bound the read by pwCoreInputs, not inputs.size():
                // pointwise-chain fusion appends operands past the op's core inputs, and one of those
                // must never be misread as C.
                const bool      hasC = pwCoreInputs(node) > 2 && node.inputs[2] != kNoTensor;
                const RtTensor *C    = hasC ? &ctx.t(node.inputs[2]) : nullptr;
                RtTensor       &Y    = ctx.t(node.outputs[0]);

                // ONNX Gemm defaults: alpha=beta=1, transA=transB=0.
                float   alpha  = node.attr.getf("alpha", 1.f);
                float   beta   = node.attr.getf("beta", 1.f);
                int64_t transA = node.attr.geti("transA", 0);
                int64_t transB = node.attr.geti("transB", 0);

                // Contraction dims from the un-transposed op() shapes: op(A) is MxK, op(B) is KxN.
                // A stored transposed is KxM, so its rows/cols swap; likewise Bt stored transposed
                // (transB=1) is NxK, the classifier weight layout where each row is one output unit.
                int64_t M = transA ? A.shape[1] : A.shape[0];
                int64_t K = transA ? A.shape[0] : A.shape[1];
                int64_t N = transB ? Bt.shape[0] : Bt.shape[1];

                float       *y = cpu::allocOut(Y, {M, N});
                const float *a = A.host.f32();
                const float *b = Bt.host.f32();
                const float *c = C ? C->host.f32() : nullptr;
                // cN is the divisor for the C broadcast below: for a 1-D bias it is the row length
                // (one value per output column N); otherwise it is C's total element count so a
                // full/other-shaped C is wrapped element-wise.
                int64_t cN = C ? (C->shape.size() == 1 ? C->shape[0] : numElements(C->shape)) : 0;

                // Output row m depends on no other row and its K-contraction runs to completion inside
                // one iteration, so the M rows partition across threads with every dot product summed
                // in the same order (and by the same NEON or scalar path) as a serial run.
                cpu::parallelFor(cpu::threadCount(ctx.config), 0, M, cpu::minChunkForWork(N * K), [&](int64_t rowBegin, int64_t rowEnd) {
                    for (int64_t m = rowBegin; m < rowEnd; ++m)
                    {
                        for (int64_t n = 0; n < N; ++n)
                        {
                            float   acc = 0;
                            int64_t k   = 0;
#if defined(VKNN_HAS_NEON)
                            // transB && !transA makes both A row m and B row n contiguous over K, so the
                            // dot product is a plain vector fused-multiply-add; k advances 4 lanes at a
                            // time and the scalar loop below finishes any K-tail. The horizontal-add
                            // reduction changes float summation order versus the scalar path, so this
                            // branch is only taken for that exact layout.
                            if (transB && !transA)
                            {
                                const float *arow = a + m * K;
                                const float *brow = b + n * K;
                                float32x4_t  v    = vdupq_n_f32(0.f);
                                for (; k + 4 <= K; k += 4)
                                {
                                    v = vmlaq_f32(v, vld1q_f32(arow + k), vld1q_f32(brow + k));
                                }
                                acc = vaddvq_f32(v);
                            }
#endif
                            // Row-major indexing: A[m,k] is a[m*K+k] (or a[k*M+m] when stored KxM), and
                            // op(B)[k,n] is b[k*N+n] (or b[n*K+k] when Bt is stored NxK, transB=1).
                            for (; k < K; ++k)
                            {
                                float av = transA ? a[k * M + m] : a[m * K + k];
                                float bv = transB ? b[n * K + k] : b[k * N + n];
                                acc += av * bv;
                            }
                            acc *= alpha;
                            if (c)
                            {
                                // C broadcast: a 1-D bias (cN==N) contributes per-column c[n]; any other
                                // C is indexed by the flattened output position modulo its element count,
                                // wrapping a scalar or partial-shape C across Y.
                                acc += beta * (cN == N ? c[n] : c[(m * N + n) % cN]);
                            }
                            y[m * N + n] = acc;
                        }
                    }
                });
                cpu::applyAct(y, Y.elems(), node.fusedAct, node.actLo, node.actHi);
            }
        };

    } // namespace

    VKNN_REGISTER_CPU_OP(OpType::Gemm, GemmCpu);

} // namespace vknn
