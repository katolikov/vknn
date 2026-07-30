// ONNX MatMul: general batched N-D matmul with NumPy broadcasting on the leading batch dims.
//   A[...,M,K] @ B[...,K,N] -> [...,M,N].  A 1-D operand [K] is promoted to [1,K] (then the
//   prepended 1 is dropped from the result); a 1-D B [K] is promoted to [K,1] (and dropped).
// CPU reference: a straight triple loop over the broadcasted batch, then M,N,K. Host buffers are
// canonical NCHW fp32, so this is the oracle the host tests validate against.
#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/parallel.h"
#include "core/matmul_view.h"
#include "vknn/op.h"
#include <algorithm>
#include <vector>

namespace vknn {
    namespace {

        // One output row of one batch matrix, with per-operand element strides: the dense path
        // passes (aK=1, bK=N, bN=1), the view path its attr strides. One routine with FP
        // contraction pinned OFF: contract(on) merely permits fusion, and the optimizer clones
        // this function per call-site constants and fuses a*b+acc in some clones but not others —
        // two calls reading the same values in the same order then round differently. Only the
        // strict IEEE mul+add chain is bit-stable across every specialization (and across
        // compilers), so that is the oracle contract: fp32 products summed over ascending k.
        VKNN_NOINLINE void matmulRow(const float *am, const float *bm, float *ym, const float *bias, int64_t N, int64_t K, int64_t aK, int64_t bK, int64_t bN) {
#pragma clang fp contract(off)
            for (int64_t n = 0; n < N; ++n)
            {
                float acc = 0.f;
                for (int64_t k = 0; k < K; ++k)
                {
                    acc += am[k * aK] * bm[k * bK + n * bN];
                }
                ym[n] = bias ? acc + bias[n] : acc;
            }
        }

        struct MatMulCpu: CpuOp {
            // View-addressed variant (core/matmul_view.h): the foldMatMulViews pass rewired the
            // operands to their chain sources and the attrs carry the per-axis element strides, so
            // the operands are read through those strides instead of dense row-major. The same
            // ascending-k fp32 accumulation makes this bit-identical to the materialized chain.
            void runView(const Node &node, ExecContext &ctx) {
                const float *a = ctx.t(node.inputs[0]).host.f32();
                const float *b = ctx.t(node.inputs[1]).host.f32();
                RtTensor    &Y = ctx.t(node.outputs[0]);

                const std::vector<int64_t> &dims    = node.attr.getints(kMmViewDims);
                const std::vector<int64_t> &aStride = node.attr.getints(kMmViewAStride);
                const std::vector<int64_t> &bStride = node.attr.getints(kMmViewBStride);
                const int64_t               aK = node.attr.geti(kMmViewAK), bK = node.attr.geti(kMmViewBK);
                const int64_t               M = node.attr.geti(kMmViewM), N = node.attr.geti(kMmViewN), K = node.attr.geti(kMmViewK);
                const int64_t               batchRank = (int64_t) dims.size() - 2;

                float       *y    = cpu::allocOut(Y, ctx.graph->desc(node.outputs[0]).shape);
                const float *bias = node.fusedBias != kNoTensor ? ctx.t(node.fusedBias).host.f32() : nullptr;

                int64_t batchElems = 1;
                for (int64_t i = 0; i < batchRank; ++i)
                {
                    batchElems *= dims[i];
                }
                // Same row partitioning and ascending-k order as the dense path below; only the
                // operand addressing differs (attr strides for the batch/m axes, aK/bK per k step).
                cpu::parallelFor(cpu::threadCount(ctx.config), 0, batchElems * M, cpu::minChunkForWork(N * K), [&](int64_t rowBegin, int64_t rowEnd) {
                    for (int64_t row = rowBegin; row < rowEnd; ++row)
                    {
                        int64_t bi    = row / M;
                        int64_t m     = row % M;
                        int64_t aBase = m * aStride[batchRank], bBase = 0, rem = bi;
                        for (int64_t i = batchRank - 1; i >= 0; --i)
                        {
                            int64_t c = rem % dims[i];
                            rem /= dims[i];
                            aBase += c * aStride[i];
                            bBase += c * bStride[i];
                        }
                        matmulRow(a + aBase, b + bBase, y + (bi * M + m) * N, bias, N, K, aK, bK, bStride[batchRank + 1]);
                    }
                });
            }

            void run(const Node &node, ExecContext &ctx) override {
                if (node.attr.has(kMmView))
                {
                    runView(node, ctx);
                    return;
                }
                const RtTensor &A = ctx.t(node.inputs[0]);
                const RtTensor &B = ctx.t(node.inputs[1]);
                RtTensor       &Y = ctx.t(node.outputs[0]);

                // Promote 1-D operands: A[K] -> [1,K]; B[K] -> [K,1]. Track whether we prepended/appended a
                // dim so it can be stripped from the output (NumPy matmul semantics).
                Shape sa = A.shape, sb = B.shape;
                bool  aWas1D = sa.size() == 1, bWas1D = sb.size() == 1;
                if (aWas1D)
                {
                    sa = {1, sa[0]};
                }
                if (bWas1D)
                {
                    sb = {sb[0], 1};
                }

                int64_t M = sa[sa.size() - 2], K = sa[sa.size() - 1];
                int64_t Kb = sb[sb.size() - 2], N = sb[sb.size() - 1];
                (void) Kb; // K == Kb by construction of a valid graph

                // Broadcast the batch dims (everything before the trailing 2) to a common shape.
                int64_t aBatchRank = (int64_t) sa.size() - 2, bBatchRank = (int64_t) sb.size() - 2;
                int64_t batchRank = std::max(aBatchRank, bBatchRank);
                Shape   batch(batchRank, 1);
                auto    aDim = [&](int64_t i) -> int64_t { // i in [0,batchRank)
                    int64_t off = batchRank - aBatchRank;
                    return i < off ? 1 : sa[i - off];
                };
                auto bDim = [&](int64_t i) -> int64_t {
                    int64_t off = batchRank - bBatchRank;
                    return i < off ? 1 : sb[i - off];
                };
                int64_t batchElems = 1;
                for (int64_t i = 0; i < batchRank; ++i)
                {
                    batch[i] = std::max(aDim(i), bDim(i));
                    batchElems *= batch[i];
                }

                // Per-batch-dim element strides into A's and B's matrix stacks (0 on a broadcast dim).
                // sA/sB seed at one full matrix (M*K, K*N elements) and accumulate the trailing batch
                // dims' extents as the loop walks right to left, so aBatchStride[i] is the element step
                // between successive matrices along batch axis i.
                std::vector<int64_t> aBatchStride(batchRank, 0), bBatchStride(batchRank, 0);
                int64_t              sA = M * K, sB = K * N;
                for (int64_t i = batchRank - 1; i >= 0; --i)
                {
                    aBatchStride[i] = (aDim(i) == 1) ? 0 : sA;
                    bBatchStride[i] = (bDim(i) == 1) ? 0 : sB;
                    sA *= aDim(i);
                    sB *= bDim(i);
                }

                // Output shape = batch ++ [M,N], with the promoted dims stripped back out.
                Shape out = batch;
                if (!aWas1D)
                {
                    out.push_back(M);
                }
                out.push_back(N);
                if (bWas1D)
                {
                    out.pop_back();
                }
                if (out.empty())
                {
                    out.push_back(1); // scalar dot product -> [1]
                }

                float       *y = cpu::allocOut(Y, out);
                const float *a = A.host.f32();
                const float *b = B.host.f32();
                // A fused Linear bias (rank-1 [N]) added per output column, matching the GPU epilogue.
                const float *bias = node.fusedBias != kNoTensor ? ctx.t(node.fusedBias).host.f32() : nullptr;

                // Flatten (batch, row) into one index: each output row of each batch matrix is written by
                // exactly one iteration and its K-contraction completes inside that iteration, so the rows
                // partition across threads with every acc summed in the same ascending-k order.
                cpu::parallelFor(cpu::threadCount(ctx.config), 0, batchElems * M, cpu::minChunkForWork(N * K), [&](int64_t rowBegin, int64_t rowEnd) {
                    for (int64_t row = rowBegin; row < rowEnd; ++row)
                    {
                        int64_t bi = row / M;
                        int64_t m  = row % M;
                        // Decode the batch index into per-dim coords to find the A/B base offsets.
                        int64_t aBase = 0, bBase = 0, rem = bi;
                        for (int64_t i = batchRank - 1; i >= 0; --i)
                        {
                            int64_t c = rem % batch[i];
                            rem /= batch[i];
                            aBase += c * aBatchStride[i];
                            bBase += c * bBatchStride[i];
                        }
                        // Row-major single-matrix product through the shared row routine: A[m,k] is
                        // am[m*K+k], B[k,n] is bm[k*N+n], the result Y[m,n] is ym[m*N+n]. acc sums
                        // over k in ascending order in fp32; that accumulation order is the
                        // observable reference the host byte-compare validates.
                        matmulRow(a + aBase + m * K, b + bBase, y + (bi * M + m) * N, bias, N, K, 1, N, 1);
                    }
                });
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::MatMul, MatMulCpu);
} // namespace vknn
