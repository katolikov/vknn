// Batched N-D MatMul on the FLAT row-major GPU path: out[...,m,n] = sum_k A[...,m,k]*B[...,k,n]
// with NumPy broadcasting on the batch dims. One thread per output element walks K (see
// shaders/matmul.comp). Either operand may be an activation or a constant initializer (e.g. a
// Linear weight); an initializer is uploaded flat in prepare(). Mirrors the CPU oracle's
// broadcast/stride math byte-for-byte.
#include "flat_ops.h"
#include "pw_plan.h"
#include "vk_op_common.h"
#include "core/matmul_tile.h"
#include "vknn/logging.h"
#include "vknn/op.h"
#include <algorithm>
#include <functional>
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
            std::shared_ptr<vk::ComputePipeline> pipe;
            MatMulPC                             pc {};
            std::shared_ptr<vk::Buffer>          constBuf[2]; // set when an operand is an initializer
            std::shared_ptr<vk::Buffer>          biasBuf;     // set when a rank-1 [N] bias is fused in
            bool                                 useTiled = false;
            int                                  numBatch = 1;
            MatMulTile                           tile     = kMatMulTiles[0]; // matmul_tiled spec constants (TM/TN/TK)

            // Set when a pointwise chain (fusePointwiseChains) is attached to this MatMul's store.
            PwEpi epi;

            // The tiled dispatch is 3-D, so it gets no runtime X-overflow rescue (the 1-D split in
            // ComputePipeline::dispatch only applies when gy==gz==1); a tile whose group counts
            // exceed the device limits must not be dispatched.
            bool tileFits(VkOpEnv &env, const MatMulTile &t) const {
                const auto &cap = env.ctx->caps();
                return (uint32_t) ((pc.N + t.tn - 1) / t.tn) <= cap.maxWorkGroupCount[0] &&
                       (uint32_t) ((pc.M + t.tm - 1) / t.tm) <= cap.maxWorkGroupCount[1] &&
                       (uint32_t) numBatch <= cap.maxWorkGroupCount[2];
            }

            // Pick the tiled-GEMM tile for this shape (an index into kMatMulTiles; 0 = the default
            // {128,128,16}). Tuning::None keeps the default; Fast/Heavy race the candidates
            // min-of-5 x 8 reps on scratch buffers and persist the winning index in the tune
            // table. Every candidate is bit-neutral (the per-output fp32 K chain is one
            // ascending-k sequence for any tile), so the race needs no anti-noise margin and the
            // choice never affects output bits.
            MatMulTile pickTile(VkOpEnv &env, bool hasBias) {
                if (env.tuning == Tuning::None || !env.runner)
                {
                    return kMatMulTiles[0];
                }
                char buf[112];
                snprintf(buf, sizeof(buf), "mm_%d_%d_%d_%d_%d", pc.M, pc.N, pc.K, numBatch, hasBias ? 1 : 0);
                std::string sig = env.gpuTag + "/" + buf;
                if (env.weights)
                {
                    // Decode guard: a stale/foreign index, or a cached tile whose dispatch no
                    // longer fits the device limits, re-races instead of decoding as garbage.
                    int cached = env.weights->tuned(sig, -1);
                    if (cached >= 0 && cached < kMatMulTileCount && tileFits(env, kMatMulTiles[cached]))
                    {
                        return kMatMulTiles[cached];
                    }
                }
                // Dedicated scratch operands sized like the real tensors (never the live
                // activation buffers). The timing batch is clamped: per-batch work is identical
                // across candidates, so the relative ranking is preserved while attention-scale
                // batch counts keep the scratch footprint bounded.
                size_t es       = env.useFp16 ? 2 : 4;
                size_t perBatch = ((size_t) pc.M * pc.K + (size_t) pc.K * pc.N + (size_t) pc.M * pc.N) * es;
                if (perBatch == 0 || perBatch > ((size_t) 256 << 20))
                {
                    return kMatMulTiles[0];
                }
                int gzT = (int) std::min<int64_t>({(int64_t) numBatch, 64, std::max<int64_t>(1, (int64_t) (((size_t) 128 << 20) / perBatch))});
                auto mk = [&](size_t bytes) {
                    return std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>(bytes, 16), vk::MemPref::kDeviceOnly);
                };
                auto sA = mk((size_t) gzT * pc.M * pc.K * es);
                auto sB = mk((size_t) gzT * pc.K * pc.N * es);
                auto sD = mk((size_t) gzT * pc.M * pc.N * es);
                std::shared_ptr<vk::Buffer> sBias = hasBias ? mk((size_t) pc.N * es) : nullptr;
                auto timeIt = [&](const std::function<void(VkCommandBuffer)> &rec) {
                    VkCommandBuffer cmd = env.runner->allocate();
                    env.runner->begin(cmd);
                    for (int r = 0; r < 8; ++r)
                    {
                        rec(cmd);
                    }
                    env.runner->end(cmd);
                    double ms = env.runner->submitAndWait(cmd);
                    vkFreeCommandBuffers(env.ctx->device(), env.runner->pool(), 1, &cmd);
                    return ms;
                };
                // Min over repeats: the fastest observed time is the least OS-perturbed estimate
                // (see tuneWino); winners persist, so the measurement earns the rigorous tier.
                auto bestOf = [&](const std::function<void(VkCommandBuffer)> &rec) {
                    double m = 1e30;
                    for (int k = 0; k < 5; ++k)
                    {
                        m = std::min(m, timeIt(rec));
                    }
                    return m;
                };
                const char *base   = hasBias ? "matmul_tiled_bias" : "matmul_tiled";
                uint32_t    nbuf   = hasBias ? 4 : 3;
                int         best   = 0;
                double      bestMs = 1e30;
                for (int ci = 0; ci < kMatMulTileCount; ++ci)
                {
                    const MatMulTile &t = kMatMulTiles[ci];
                    if (!tileFits(env, t))
                    {
                        continue;
                    }
                    auto     p   = env.pipeline(shader(base, env.useFp16), nbuf, sizeof(MatMulPC), {(uint32_t) t.tm, (uint32_t) t.tn, (uint32_t) t.tk});
                    uint32_t gxT = (uint32_t) ((pc.N + t.tn - 1) / t.tn);
                    uint32_t gyT = (uint32_t) ((pc.M + t.tm - 1) / t.tm);
                    double   ms  = bestOf([&](VkCommandBuffer cmd) {
                        std::vector<VkBuffer> bufs {sA->handle(), sB->handle(), sD->handle()};
                        if (sBias)
                        {
                            bufs.push_back(sBias->handle());
                        }
                        p->dispatch(cmd, bufs, &pc, sizeof(pc), gxT, gyT, (uint32_t) gzT);
                        vk::computeBarrier(cmd);
                    });
                    if (ms < bestMs)
                    {
                        bestMs = ms;
                        best   = ci;
                    }
                }
                VKNN_DEBUG << "autotune " << sig << " -> tile " << kMatMulTiles[best].tm << "x" << kMatMulTiles[best].tn << "x" << kMatMulTiles[best].tk;
                if (env.weights)
                {
                    env.weights->setTuned(sig, best);
                }
                return kMatMulTiles[best];
            }

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
                // keep the naive 1-thread/output kernel. fusePointwiseChains mirrors this predicate via
                // the same constant (core/matmul_tile.h).
                useTiled = !aWas1D && !bWas1D && M >= kTiledMatMulMin && N >= kTiledMatMulMin && K >= kTiledMatMulMin;
                numBatch = (M > 0 && N > 0) ? pc.total / (int) (M * N) : 1;
                if (useTiled && !tileFits(env, kMatMulTiles[0]))
                {
                    // A shape whose group counts exceed the device limits keeps the naive kernel:
                    // its 1-D dispatch gets the runtime X-overflow split, and it is byte-identical
                    // to the tiled kernel (same ascending-k fp32 chain per output).
                    useTiled = false;
                }
                if (useTiled)
                {
                    tile = pickTile(env, node.fusedBias != kNoTensor);
                }

                // A fused Linear bias (rank-1 [N]) is added in the fp32 accumulator by the _bias kernel
                // variant; upload it flat and bind it as a 4th buffer.
                const char *base = useTiled ? "matmul_tiled" : "matmul";
                std::string name = base;
                uint32_t    nbuf = 3;
                if (node.fusedBias != kNoTensor)
                {
                    biasBuf = uploadInit(env, node.fusedBias, g.desc(node.fusedBias).shape);
                    name += "_bias";
                    nbuf = 4;
                }

                // A pointwise chain (fusePointwiseChains) attached to this MatMul runs in the kernel's own
                // epilogue (shaders/pw_epilogue.glsl), appended at binding nbuf. The plan indexes the flat
                // row-major output world (MatMul's output is always row-major, never NC4HW4).
                epi.prepare(node, env, /*flat=*/true, out);
                name += epi.suffix();
                nbuf += epi.extraBufs();

                // The tiled kernel takes its TM/TN/TK tile as specialization constants 0/1/2.
                std::vector<uint32_t> spec;
                if (useTiled)
                {
                    spec = {(uint32_t) tile.tm, (uint32_t) tile.tn, (uint32_t) tile.tk};
                }
                pipe = env.pipeline(shader(name.c_str(), env.useFp16), nbuf, sizeof(MatMulPC), spec);
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                auto buf = [&](int e) {
                    return constBuf[e] ? constBuf[e].get() : env.devBuf(node.inputs[e]);
                };
                VkBuffer              dstHandle = env.devBuf(node.outputs[0])->handle();
                std::vector<VkBuffer> bufs {buf(0)->handle(), buf(1)->handle(), dstHandle};
                if (biasBuf)
                {
                    bufs.push_back(biasBuf->handle());
                }
                epi.append(bufs, node, env, dstHandle);
                if (useTiled)
                {
                    // Tiled GEMM contract: one workgroup per tm x tn output tile, dispatched as
                    // (ceil(N/tn), ceil(M/tm), numBatch). x/y cover the N/M matrix dims, z indexes the
                    // flattened batch dims (see matmul_tiled.comp). The tile is the pipeline's
                    // spec-constant geometry, so the dispatch math and the kernel cannot drift.
                    uint32_t gx = (uint32_t) ((pc.N + tile.tn - 1) / tile.tn);
                    uint32_t gy = (uint32_t) ((pc.M + tile.tm - 1) / tile.tm);
                    pipe->dispatch(cmd, bufs, &pc, sizeof(pc), gx, gy, (uint32_t) numBatch);
                } else
                {
                    // Naive kernel: one thread per output element over a flat 1-D grid of pc.total lanes.
                    // matmul.comp is local_size_x=256 == flat::kFlatLocalSize.
                    pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(pc.total, flat::kFlatLocalSize));
                }
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::MatMul, MatMulOp);
} // namespace vknn
