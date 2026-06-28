// Batched N-D MatMul on the FLAT row-major GPU path: out[...,m,n] = sum_k A[...,m,k]*B[...,k,n]
// with NumPy broadcasting on the batch dims. One thread per output element walks K (see
// shaders/matmul.comp). Either operand may be an activation or a constant initializer (e.g. a
// Linear weight); an initializer is uploaded flat in prepare(). Mirrors the CPU oracle's
// broadcast/stride math byte-for-byte.
#include "flat_ops.h"
#include "vk_op_common.h"
#include "vknn/op.h"
#include <algorithm>
#include <vector>

namespace vknn {
    namespace {

        struct MatMulPC {
            int rank, total, M, N, K, aK, bK;
            int outDim[flat::kMaxRank];
            int aStride[flat::kMaxRank];
            int bStride[flat::kMaxRank];
        };

        struct MatMulOp: VulkanOp {
            std::unique_ptr<vk::ComputePipeline> pipe;
            MatMulPC                             pc {};
            std::shared_ptr<vk::Buffer>          constBuf[2]; // set when an operand is an initializer
            bool                                 useTiled = false;
            int                                  numBatch = 1;
            static constexpr int                 kTile    = 128; // must match TM/TN in matmul_tiled.comp

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g   = *env.graph;
                Shape        sa  = g.desc(node.inputs[0]).shape;
                Shape        sb  = g.desc(node.inputs[1]).shape;
                Shape        out = g.desc(node.outputs[0]).shape;

                // Promote 1-D operands (A[K]->[1,K], B[K]->[K,1]) to find M/N/K; the output rank already had
                // the promoted dim stripped by inferShapes, so we work the strides against `out` directly
                // below.
                bool aWas1D = sa.size() == 1, bWas1D = sb.size() == 1;
                if (aWas1D)
                {
                    sa = {1, sa[0]};
                }
                if (bWas1D)
                {
                    sb = {sb[0], 1};
                }

                int64_t M = sa[sa.size() - 2], K = sa[sa.size() - 1];
                int64_t N = sb[sb.size() - 1];

                int rank = (int) out.size();
                pc.rank  = rank;
                pc.total = (int) numElements(out);
                pc.M     = (int) M;
                pc.N     = (int) N;
                pc.K     = (int) K;
                pc.aK    = 1;       // A is [...,M,K] row-major -> stepping K moves by 1
                pc.bK    = (int) N; // B is [...,K,N] row-major -> stepping K moves by N
                for (int k = 0; k < rank; ++k)
                {
                    pc.outDim[k] = (int) out[k];
                }

                // The trailing output dims are the matrix dims. With 1-D promotion an axis may be absent:
                //   A 1-D  -> the M axis was dropped from the output; B 1-D -> the N axis was dropped.
                // Identify which output index (if any) is the M axis and which is the N axis.
                int nAxis = rank - 1; // N is the last output dim, unless B was 1-D (then absent)
                int mAxis = aWas1D ? -1 : (bWas1D ? rank - 1 : rank - 2);
                if (bWas1D)
                {
                    nAxis = -1; // N axis was stripped
                }
                // batch dims occupy output indices [0, firstMatAxis)
                int firstMatAxis = rank;
                if (mAxis >= 0)
                {
                    firstMatAxis = std::min(firstMatAxis, mAxis);
                }
                if (nAxis >= 0)
                {
                    firstMatAxis = std::min(firstMatAxis, nAxis);
                }
                int batchRank = firstMatAxis;

                // Per-operand batch shapes (everything before the trailing matrix dims), left-padded to
                // batchRank.
                int64_t aBatchRank = (int64_t) sa.size() - 2, bBatchRank = (int64_t) sb.size() - 2;
                auto    aDim = [&](int i) -> int64_t {
                    int off = batchRank - (int) aBatchRank;
                    return i < off ? 1 : sa[i - off];
                };
                auto bDim = [&](int i) -> int64_t {
                    int off = batchRank - (int) bBatchRank;
                    return i < off ? 1 : sb[i - off];
                };
                std::vector<int64_t> aBatchStride(batchRank, 0), bBatchStride(batchRank, 0);
                int64_t              sAcc = M * K, sBcc = K * N;
                for (int i = batchRank - 1; i >= 0; --i)
                {
                    aBatchStride[i] = (aDim(i) == 1) ? 0 : sAcc;
                    bBatchStride[i] = (bDim(i) == 1) ? 0 : sBcc;
                    sAcc *= aDim(i);
                    sBcc *= bDim(i);
                }
                for (int i = 0; i < batchRank; ++i)
                {
                    pc.aStride[i] = (int) aBatchStride[i];
                    pc.bStride[i] = (int) bBatchStride[i];
                }
                // Matrix-axis strides: A depends on m (row stride K) not n; B depends on n (col stride 1) not
                // m.
                if (mAxis >= 0)
                {
                    pc.aStride[mAxis] = (int) K;
                    pc.bStride[mAxis] = 0;
                }
                if (nAxis >= 0)
                {
                    pc.aStride[nAxis] = 0;
                    pc.bStride[nAxis] = 1;
                }

                // Upload a constant operand flat (row-major NCHW fp32 -> device, fp16 when half precision).
                // Direct fp16->fp16 passthrough when the stored weight already matches compute precision.
                auto maybeUpload = [&](TensorId t, int which, const Shape &s) {
                    if (!g.isInitializer(t))
                    {
                        return;
                    }
                    constBuf[which] = uploadInit(env, t, s);
                };
                maybeUpload(node.inputs[0], 0, g.desc(node.inputs[0]).shape);
                maybeUpload(node.inputs[1], 1, g.desc(node.inputs[1]).shape);

                // Use the register-blocked tiled GEMM for the standard (non-mat-vec) case with large
                // enough matrices; it assumes M at out[rank-2], N at out[rank-1], so the batch dims are
                // exactly out[0..rank-3] (true when neither operand was 1-D). Tiny / mat-vec / 1-D cases
                // keep the naive 1-thread/output kernel.
                useTiled = !aWas1D && !bWas1D && M >= 32 && N >= 32 && K >= 32;
                numBatch = (M > 0 && N > 0) ? pc.total / (int) (M * N) : 1;

                pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader(useTiled ? "matmul_tiled" : "matmul", env.useFp16), 3, sizeof(MatMulPC), std::vector<uint32_t> {},
                                                             env.cache->handle());
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                auto buf = [&](int e) {
                    return constBuf[e] ? constBuf[e].get() : env.devBuf(node.inputs[e]);
                };
                std::vector<VkBuffer> bufs {buf(0)->handle(), buf(1)->handle(), env.devBuf(node.outputs[0])->handle()};
                if (useTiled)
                {
                    uint32_t gx = (uint32_t) ((pc.N + kTile - 1) / kTile);
                    uint32_t gy = (uint32_t) ((pc.M + kTile - 1) / kTile);
                    pipe->dispatch(cmd, bufs, &pc, sizeof(pc), gx, gy, (uint32_t) numBatch);
                } else
                {
                    pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(pc.total, 256));
                }
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::MatMul, MatMulOp);
} // namespace vknn
