// Batched N-D MatMul on the FLAT row-major GPU path: out[...,m,n] = sum_k A[...,m,k]*B[...,k,n]
// with NumPy broadcasting on the batch dims. Three kernels serve it, by shape: the register-blocked
// tiled GEMM (shaders/matmul_tiled*.comp) for full matrices, the split-K mat-vec
// (shaders/matmul_gemv*.comp — vector-load when N % kGemvVec == 0, scalar otherwise) for a narrow
// M == 1 row, and otherwise one thread per output element walking K (shaders/matmul.comp). Either
// operand may be an activation or a constant initializer (e.g. a Linear weight); an initializer is
// uploaded flat in prepare(). The naive and tiled kernels mirror the CPU oracle's broadcast/stride
// math byte-for-byte; the split-K kernels regroup the k chain (deterministically, identically to each
// other) and so agree only to fp32 rounding.
#include "backend/vulkan/coopmat_check.h"
#include "core/lowp_gemm.h"
#include "core/matmul_tile.h"
#include "core/matmul_view.h"
#include "core/quant_weights.h"
#include "flat_ops.h"
#include "pw_plan.h"
#include "vk_op_common.h"
#include "vknn/hint.h"
#include "vknn/logging.h"
#include "vknn/op.h"
#include <algorithm>
#include <functional>
#include <vector>

namespace vknn {
    namespace {

        // Scalars only; the per-axis outDim/aStride/bStride geometry rides a content-deduped SSBO
        // (flat::uploadFlatGeom) bound after the operands (and after the fused bias when present), so
        // the decodable batch rank is unbounded and the push constant stays small.
        struct MatMulPC {
            int rank, total, M, N, K, aK, bK;
        };

        // Push constant for the packed-quantized-weight kernels (matmul_gemv_i4/i8 /
        // matmul_tiled_i4/i8). The weight is always a 2-D constant, so there is no geometry SSBO:
        // A/D batch offsets are dense whole-matrix steps, decoded in-kernel from the row/batch
        // workgroup id. rowWords is the packed row width in uint words — format-dependent
        // (core/quant_weights.h) but always word-valued, so every kernel strides identically.
        struct MatMulWqPC {
            int total, M, N, K, group, nOut, rowWords, nGroups;
        };

        struct MatMulOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            MatMulPC                             pc {};
            std::shared_ptr<vk::Buffer>          geom;        // outDim/aStride/bStride, deduped SSBO
            std::shared_ptr<vk::Buffer>          constBuf[2]; // set when an operand is an initializer
            std::shared_ptr<vk::Buffer>          biasBuf;     // set when a rank-1 [N] bias is fused in
            bool                                 useTiled = false;
            bool                                 useGemv  = false; // split-K mat-vec (shaders/matmul_gemv*.comp)
            int                                  numBatch = 1;
            MatMulTile                           tile     = kMatMulTiles[0]; // matmul_tiled spec constants (TM/TN/TK)

            // Set when a pointwise chain (fusePointwiseChains) is attached to this MatMul's store.
            PwEpi epi;

            // Packed-quantized-weight path (vknn_compile -Os; scheme in core/quant_weights.h). The
            // session's materialization keeps the kWq attrs only on Vulkan-assigned MatMuls whose
            // format has a native kernel, so their presence here guarantees the payload is packed
            // and this op must read it natively.
            bool                        useWq = false;
            MatMulWqPC                  wqPc {};
            std::shared_ptr<vk::Buffer> wqPacked, wqScales, wqOidx, wqOval, wqLut;

            // Cooperative-matrix path (Hint::CoopmatGemm; routing rule in core/lowp_gemm.h).
            // Fp16 binds the operands directly; the opt-in low-precision kinds add a per-run
            // A-quantization prelude (absmax -> quant) and a host-quantized weight operand.
            CoopmatGemmKind                      coopKind = CoopmatGemmKind::None;
            std::shared_ptr<vk::ComputePipeline> coopAbsmaxPipe, coopQuantPipe;
            std::shared_ptr<vk::Buffer>          coopWeights; // e4m3/int8 codes of the B initializer
            std::shared_ptr<vk::Buffer>          coopScales;  // [0] sA (device-written), [1] sB (host-written)
            std::shared_ptr<vk::Buffer>          coopQuantA;  // per-run quantized A operand
            struct CoopAbsmaxPC {
                int   total;
                float divisor;
            };
            struct CoopGemmPC {
                int M, N, K;
            };
            CoopGemmPC coopPc {};

            // Build the coopmat pipelines and (for the opt-in low-precision kinds) the quantized
            // weight + scale + scratch buffers. The routing rule already established: dense 2-D
            // batch-1 fp16 GEMM, no bias/epilogue, M,N multiples of 32, K of 16, caps present and
            // the self-check passed; for Fp8/Int8 the B operand is an initializer.
            void prepareCoopmat(const Node &node, VkOpEnv &env) {
                const Graph &g = *env.graph;
                useTiled       = false;
                useGemv        = false;
                numBatch       = 1;
                coopPc         = {pc.M, pc.N, pc.K};
                if (coopKind == CoopmatGemmKind::Fp16)
                {
                    pipe = env.pipeline("coopmat_gemm", 3, sizeof(CoopGemmPC), {}, /*requiredSubgroupSize=*/32);
                    return;
                }
                const bool  fp8     = coopKind == CoopmatGemmKind::Fp8;
                const float divisor = fp8 ? 448.f : 127.f;

                // Host-quantized weights: per-tensor symmetric scale sB = absmax / divisor, codes
                // uploaded device-only. The dequantization factor sA * sB rides the scales SSBO.
                std::vector<float> weightFloats = initFloats(g, node.inputs[1]);
                float              absmax       = 0.f;
                for (float w: weightFloats)
                {
                    absmax = std::max(absmax, std::fabs(w));
                }
                const float scaleB = absmax > 0.f ? absmax / divisor : 0.f;
                coopWeights        = env.acquireWeight(node.name + (fp8 ? "#cmf8" : "#cmi8"), env.useFp16, [&] {
                    std::vector<uint8_t> codes(weightFloats.size());
                    for (size_t i = 0; i < weightFloats.size(); ++i)
                    {
                        if (fp8)
                        {
                            codes[i] = encodeFp8E4M3(scaleB > 0.f ? weightFloats[i] / scaleB : 0.f);
                        } else
                        {
                            codes[i] = (uint8_t) encodeInt8Symmetric(weightFloats[i], scaleB);
                        }
                    }
                    return env.uploadWeightDeviceOnly(codes.data(), codes.size(), codes.size());
                });

                // scales[0] = sA, written by the absmax dispatch each run; scales[1] = sB, host-set
                // once. zeroInit keeps scales[0] deterministic before the first absmax write.
                coopScales = std::make_shared<vk::Buffer>(*env.ctx, 2 * sizeof(float), vk::MemPref::kAuto, 0, /*zeroInit=*/true);
                coopScales->upload(&scaleB, sizeof(float), sizeof(float));
                coopQuantA = std::make_shared<vk::Buffer>(*env.ctx, (size_t) pc.M * (size_t) pc.K, vk::MemPref::kDeviceOnly);

                // No ternary composition here: these kernels host no pointwise epilogue, and the
                // epi-sync checker treats literal-bearing ternaries in this file as epi stems.
                const char *quantShader = "lowp_quant_i8";
                const char *gemmShader  = "coopmat_gemm_i8";
                if (fp8)
                {
                    quantShader = "lowp_quant_fp8";
                    gemmShader  = "coopmat_gemm_fp8";
                }
                coopAbsmaxPipe = env.pipeline("lowp_absmax", 2, sizeof(CoopAbsmaxPC));
                coopQuantPipe  = env.pipeline(quantShader, 3, sizeof(int));
                pipe           = env.pipeline(gemmShader, 4, sizeof(CoopGemmPC), {}, /*requiredSubgroupSize=*/32);
            }

            void recordCoopmat(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) {
                vk::Buffer    *srcA      = constBuf[0] ? constBuf[0].get() : env.devBuf(node.inputs[0]);
                VkBuffer       dstHandle = env.devBuf(node.outputs[0])->handle();
                const uint32_t gx        = (uint32_t) (pc.N / 32);
                const uint32_t gy        = (uint32_t) (pc.M / 32);
                if (coopKind == CoopmatGemmKind::Fp16)
                {
                    vk::Buffer *srcB = constBuf[1] ? constBuf[1].get() : env.devBuf(node.inputs[1]);
                    pipe->dispatch(cmd, {srcA->handle(), srcB->handle(), dstHandle}, &coopPc, sizeof(coopPc), gx, gy);
                    return;
                }
                // Opt-in low-precision: per-tensor A absmax -> quantize A -> GEMM. The scales and
                // quantized-A buffers are op-private, so the two internal barriers order only this
                // node's prelude against its GEMM.
                CoopAbsmaxPC absmaxPc {pc.M * pc.K, coopKind == CoopmatGemmKind::Fp8 ? 448.f : 127.f};
                coopAbsmaxPipe->dispatch(cmd, {srcA->handle(), coopScales->handle()}, &absmaxPc, sizeof(absmaxPc), 1);
                vk::computeBarrier(*env.ctx, cmd);
                const int quantTotal = pc.M * pc.K;
                coopQuantPipe->dispatch(cmd, {srcA->handle(), coopQuantA->handle(), coopScales->handle()}, &quantTotal, sizeof(quantTotal), groups(quantTotal, 256));
                vk::computeBarrier(*env.ctx, cmd);
                pipe->dispatch(cmd, {coopQuantA->handle(), coopWeights->handle(), dstHandle, coopScales->handle()}, &coopPc, sizeof(coopPc), gx, gy);
            }

            // Prepare the packed-weight dispatch: raw packed/index payloads upload unconverted, the
            // fp16 scale/outlier tensors upload at compute precision (uploadInit's passthrough), and
            // the kernel is the split-K GEMV for a single output row or the {128,128,16} tile
            // otherwise, in the wrapper variant matching the weight's format.
            void prepareWq(const Node &node, VkOpEnv &env) {
                const Graph  &g      = *env.graph;
                const Shape  &sa     = g.desc(node.inputs[0]).shape;
                const Shape  &out    = g.desc(node.outputs[0]).shape;
                const int64_t M      = sa.size() >= 2 ? sa[sa.size() - 2] : 1;
                const int     format = weightQuantFormat(node);
                const bool    int8Fmt = format == kWqFormatInt8;
                const bool    lutFmt  = format == kWqFormatLut4;

                wqPc.total = (int) numElements(out);
                wqPc.M     = (int) M;
                wqPc.N     = (int) node.attr.geti(kWqN, 0);
                wqPc.K     = (int) node.attr.geti(kWqK, 0);
                wqPc.group = (int) node.attr.geti(kWqGroup, 1);
                wqPc.nOut  = (int) node.attr.geti(kWqNOut, 0);
                // Packed row width in uint words (int4RowBytes / int8RowBytes over 4): one word per
                // 8-column block at 4 bits, two at 8 bits.
                wqPc.rowWords = int8Fmt ? (int) (2 * ((wqPc.N + 7) / 8)) : (int) ((wqPc.N + 7) / 8);
                wqPc.nGroups  = (int) int4GroupCount(wqPc.K, wqPc.group);
                numBatch      = wqPc.N > 0 && M > 0 ? wqPc.total / (int) (M * (int64_t) wqPc.N) : 1;

                const TensorId scaleId = (TensorId) node.attr.geti(kWqScales, kNoTensor);
                const TensorId oidxId  = (TensorId) node.attr.geti(kWqOidx, kNoTensor);
                const TensorId ovalId  = (TensorId) node.attr.geti(kWqOval, kNoTensor);
                // Weight-pool tags, per format so a raw upload never aliases across payload layouts.
                const char *packedTag = "i4w";
                const char *oidxTag   = "i4i";
                if (int8Fmt)
                {
                    packedTag = "i8w";
                    oidxTag   = "i8i";
                }
                wqPacked = uploadInitRaw(env, node.inputs[1], packedTag);
                wqScales = uploadInit(env, scaleId, g.desc(scaleId).shape);
                if (wqPc.nOut > 0)
                {
                    wqOidx = uploadInitRaw(env, oidxId, oidxTag);
                    wqOval = uploadInit(env, ovalId, g.desc(ovalId).shape);
                } else
                {
                    // Descriptor sets bind every declared buffer; a zero-outlier weight binds a
                    // shared dummy word the kernel's nOut == 0 loop never reads.
                    wqOidx = env.acquireWeight("i4#dummy", env.useFp16, [&] {
                        const uint32_t zero[4] = {0, 0, 0, 0};
                        return env.uploadWeightDeviceOnly(zero, sizeof zero, sizeof zero);
                    });
                    wqOval = wqOidx;
                }

                useGemv = M == 1;
                const char *base;
                if (lutFmt)
                {
                    base = useGemv ? "matmul_gemv_lut4" : "matmul_tiled_lut4";
                } else if (int8Fmt)
                {
                    base = useGemv ? "matmul_gemv_i8" : "matmul_tiled_i8";
                } else
                {
                    base = useGemv ? "matmul_gemv_i4" : "matmul_tiled_i4";
                }
                std::string name = base;
                uint32_t    nbuf = 6; // A, packed, D, scales, oidx, oval
                if (node.fusedBias != kNoTensor)
                {
                    biasBuf = uploadInit(env, node.fusedBias, g.desc(node.fusedBias).shape);
                    name += "_bias";
                    nbuf = 7;
                }
                if (lutFmt)
                {
                    // The 16-entry codebook binds after the bias (matmul_*_lut4 declare it at 6, the
                    // _bias twins at 7).
                    const TensorId lutId = (TensorId) node.attr.geti(kWqLut, kNoTensor);
                    wqLut                = uploadInit(env, lutId, g.desc(lutId).shape);
                    nbuf += 1;
                }
                epi.prepare(node, env, /*flat=*/true, out);
                name += epi.suffix();
                nbuf += epi.extraBufs();
                pipe = env.pipeline(shader(name.c_str(), env.useFp16), nbuf, sizeof(MatMulWqPC), {});
            }

            void recordWq(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) {
                vk::Buffer           *src       = constBuf[0] ? constBuf[0].get() : env.devBuf(node.inputs[0]);
                VkBuffer              dstHandle = env.devBuf(node.outputs[0])->handle();
                std::vector<VkBuffer> bufs {src->handle(), wqPacked->handle(), dstHandle, wqScales->handle(), wqOidx->handle(), wqOval->handle()};
                if (biasBuf)
                {
                    bufs.push_back(biasBuf->handle());
                }
                if (wqLut)
                {
                    bufs.push_back(wqLut->handle());
                }
                epi.append(bufs, node, env, dstHandle);
                if (useGemv)
                {
                    // (ceil(N/64) output blocks, one output row per y group): a lane's 8 outputs
                    // stay word-aligned within their row, so a ragged N only pads the last block.
                    const uint32_t rows = (uint32_t) (wqPc.N > 0 ? wqPc.total / wqPc.N : 1);
                    pipe->dispatch(cmd, bufs, &wqPc, sizeof(wqPc), groups(wqPc.N, 64), rows);
                } else
                {
                    const uint32_t gx = (uint32_t) ((wqPc.N + 127) / 128);
                    const uint32_t gy = (uint32_t) ((wqPc.M + 127) / 128);
                    pipe->dispatch(cmd, bufs, &wqPc, sizeof(wqPc), gx, gy, (uint32_t) numBatch);
                }
            }

            // The tiled dispatch is 3-D, so it gets no runtime X-overflow rescue (the 1-D split in
            // ComputePipeline::dispatch only applies when gy==gz==1); a tile whose group counts
            // exceed the device limits must not be dispatched.
            bool tileFits(VkOpEnv &env, const MatMulTile &t) const {
                const auto &cap = env.ctx->caps();
                return (uint32_t) ((pc.N + t.tn - 1) / t.tn) <= cap.maxWorkGroupCount[0] && (uint32_t) ((pc.M + t.tm - 1) / t.tm) <= cap.maxWorkGroupCount[1] && (uint32_t) numBatch <= cap.maxWorkGroupCount[2];
            }

            // Pick the tiled-GEMM tile for this shape (an index into kMatMulTiles; 0 = the default
            // {128,128,16}). Tuning::None keeps the default; Fast/Heavy race the candidates
            // min-of-5 x 8 reps on scratch buffers and persist the winning index in the tune
            // table. Every candidate is bit-neutral (the per-output fp32 K chain is one
            // ascending-k sequence for any tile), so the race needs no anti-noise margin and the
            // choice never affects output bits.
            MatMulTile pickTile(VkOpEnv &env, bool hasBias) {
                char buf[112];
                snprintf(buf, sizeof(buf), "mm_%d_%d_%d_%d_%d", pc.M, pc.N, pc.K, numBatch, hasBias ? 1 : 0);
                std::string sig = env.gpuTag + "/" + buf;
                int         reuse;
                // Decode guard: a stale/foreign index, or a cached tile whose dispatch no longer fits the
                // device limits, re-races instead of decoding as garbage.
                if (env.reuseTuned(sig, reuse) && reuse >= 0 && reuse < kMatMulTileCount && tileFits(env, kMatMulTiles[reuse]))
                {
                    return kMatMulTiles[reuse];
                }
                if (env.tuning == Tuning::None || !env.runner)
                {
                    return kMatMulTiles[0];
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
                int  gzT = (int) std::min<int64_t>({(int64_t) numBatch, 64, std::max<int64_t>(1, (int64_t) (((size_t) 128 << 20) / perBatch))});
                auto mk  = [&](size_t bytes) {
                    return std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>(bytes, 16), vk::MemPref::kDeviceOnly);
                };
                auto                        sA     = mk((size_t) gzT * pc.M * pc.K * es);
                auto                        sB     = mk((size_t) gzT * pc.K * pc.N * es);
                auto                        sD     = mk((size_t) gzT * pc.M * pc.N * es);
                std::shared_ptr<vk::Buffer> sBias  = hasBias ? mk((size_t) pc.N * es) : nullptr;
                auto                        timeIt = [&](const std::function<void(VkCommandBuffer)> &rec) {
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
                // Time each candidate with the kernel it will actually dispatch: the default tile
                // runs the compile-time _fast kernel (what prepare() picks for it), every other tile
                // runs the spec-constant kernel. Timing the default with the spec-constant kernel
                // would under-rank it against its own real (faster) dispatch.
                const char *specBase = hasBias ? "matmul_tiled_bias" : "matmul_tiled";
                const char *fastBase = hasBias ? "matmul_tiled_fast_bias" : "matmul_tiled_fast";
                uint32_t    nbuf     = hasBias ? 5 : 4; // + the geometry SSBO bound after the operands/bias
                int         best     = 0;
                double      bestMs   = 1e30;
                for (int ci = 0; ci < kMatMulTileCount; ++ci)
                {
                    const MatMulTile &t = kMatMulTiles[ci];
                    if (!tileFits(env, t))
                    {
                        continue;
                    }
                    bool                  fast = isDefaultMatMulTile(t);
                    std::vector<uint32_t> spec = fast ? std::vector<uint32_t> {} : std::vector<uint32_t> {(uint32_t) t.tm, (uint32_t) t.tn, (uint32_t) t.tk};
                    auto                  p    = env.pipeline(shader(fast ? fastBase : specBase, env.useFp16), nbuf, sizeof(MatMulPC), spec);
                    uint32_t              gxT  = (uint32_t) ((pc.N + t.tn - 1) / t.tn);
                    uint32_t              gyT  = (uint32_t) ((pc.M + t.tm - 1) / t.tm);
                    double                ms   = bestOf([&](VkCommandBuffer cmd) {
                        std::vector<VkBuffer> bufs {sA->handle(), sB->handle(), sD->handle()};
                        if (sBias)
                        {
                            bufs.push_back(sBias->handle());
                        }
                        bufs.push_back(geom->handle()); // geometry SSBO (matches the real dispatch's binding count)
                        p->dispatch(cmd, bufs, &pc, sizeof(pc), gxT, gyT, (uint32_t) gzT);
                        vk::computeBarrier(*env.ctx, cmd);
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
                    env.weights->setTuned(sig, best, (int) env.tuning);
                }
                return kMatMulTiles[best];
            }

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g = *env.graph;
                if (node.attr.has(kWq))
                {
                    // A packed quantized weight takes its own kernels; the constant A operand (if
                    // any) still uploads flat below the branch's needs.
                    if (g.isInitializer(node.inputs[0]))
                    {
                        constBuf[0] = uploadInit(env, node.inputs[0], g.desc(node.inputs[0]).shape);
                    }
                    useWq = true;
                    prepareWq(node, env);
                    return;
                }
                Shape sa  = g.desc(node.inputs[0]).shape;
                Shape sb  = g.desc(node.inputs[1]).shape;
                Shape out = g.desc(node.outputs[0]).shape;

                const bool hasView = node.attr.has(kMmView);
                bool       aWas1D = false, bWas1D = false;
                int64_t    M, N, K;
                int        rank;
                // gemv4 loads B as vec4 along n; a view keeps that legal only when its n stride is 1
                // and every other B offset term is 4-aligned.
                bool                 viewGemv4Ok = false;
                std::vector<int32_t> outDim, aStride, bStride;
                if (hasView)
                {
                    // View-addressed operands (core/matmul_view.h): the foldMatMulViews pass rewired
                    // the inputs to their chain sources and precomputed the whole geometry — dims may
                    // split a batch axis (GQA head groups), so every array below is authoritative and
                    // the dense derivation in the else-branch does not apply.
                    const std::vector<int64_t> &dims = node.attr.getints(kMmViewDims);
                    const std::vector<int64_t> &vas  = node.attr.getints(kMmViewAStride);
                    const std::vector<int64_t> &vbs  = node.attr.getints(kMmViewBStride);
                    rank                             = (int) dims.size();
                    M                                = node.attr.geti(kMmViewM);
                    N                                = node.attr.geti(kMmViewN);
                    K                                = node.attr.geti(kMmViewK);
                    pc.aK                            = (int) node.attr.geti(kMmViewAK);
                    pc.bK                            = (int) node.attr.geti(kMmViewBK);
                    outDim.resize(rank);
                    aStride.resize(rank);
                    bStride.resize(rank);
                    viewGemv4Ok = pc.bK % kGemvVec == 0;
                    for (int i = 0; i < rank; ++i)
                    {
                        outDim[i]  = (int) dims[i];
                        aStride[i] = (int) vas[i];
                        bStride[i] = (int) vbs[i];
                        if (i < rank - 1 && bStride[i] % kGemvVec != 0)
                        {
                            viewGemv4Ok = false;
                        }
                    }
                    viewGemv4Ok = viewGemv4Ok && bStride[rank - 1] == 1;
                } else
                {
                    // Promote 1-D operands (A[K]->[1,K], B[K]->[K,1]) to find M/N/K; the output rank already had
                    // the promoted dim stripped by inferShapes, so we work the strides against `out` directly
                    // below.
                    aWas1D = sa.size() == 1;
                    bWas1D = sb.size() == 1;
                    if (aWas1D)
                    {
                        sa = {1, sa[0]};
                    }
                    if (bWas1D)
                    {
                        sb = {sb[0], 1};
                    }

                    M = sa[sa.size() - 2];
                    K = sa[sa.size() - 1];
                    N = sb[sb.size() - 1];

                    rank  = (int) out.size();
                    pc.aK = 1;       // A is [...,M,K] row-major -> stepping K moves by 1
                    pc.bK = (int) N; // B is [...,K,N] row-major -> stepping K moves by N
                    outDim.assign(rank, 0);
                    aStride.assign(rank, 0);
                    bStride.assign(rank, 0);
                    for (int k = 0; k < rank; ++k)
                    {
                        outDim[k] = (int) out[k];
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
                        aStride[i] = (int) aBatchStride[i];
                        bStride[i] = (int) bBatchStride[i];
                    }
                    // Matrix-axis strides: A depends on m (row stride K) not n; B depends on n (col stride 1) not
                    // m.
                    if (mAxis >= 0)
                    {
                        aStride[mAxis] = (int) K;
                        bStride[mAxis] = 0;
                    }
                    if (nAxis >= 0)
                    {
                        aStride[nAxis] = 0;
                        bStride[nAxis] = 1;
                    }
                }
                pc.rank  = rank;
                pc.total = (int) numElements(out);
                pc.M     = (int) M;
                pc.N     = (int) N;
                pc.K     = (int) K;
                geom     = flat::uploadFlatGeom(env, {outDim, aStride, bStride});

                // ---- cooperative-matrix routing (deterministic capability + shape rule) ----
                // The route never races: the coopmat kernels regroup the K reduction relative to
                // the SSBO kernels, so the choice is a pure function of device caps, the
                // Hint::CoopmatGemm value and the shape (core/lowp_gemm.h). A one-time on-device
                // exact self-check guards the kernel's fragment mapping before the first use.
                {
                    const auto     &cap = env.ctx->caps();
                    CoopmatGemmCaps cmCaps;
                    cmCaps.coopmatFp16Fp32Row16 = cap.hasCoopmatShape(16, 16, 16, (uint32_t) VK_COMPONENT_TYPE_FLOAT16_KHR, (uint32_t) VK_COMPONENT_TYPE_FLOAT32_KHR);
                    cmCaps.coopmatFp8Fp32Row16  = cap.shaderFloat8CoopMat && cap.hasCoopmatShape(16, 16, 16, (uint32_t) VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT, (uint32_t) VK_COMPONENT_TYPE_FLOAT32_KHR);
                    cmCaps.coopmatI8I32Row16    = cap.hasCoopmatShape(16, 16, 16, (uint32_t) VK_COMPONENT_TYPE_SINT8_KHR, (uint32_t) VK_COMPONENT_TYPE_SINT32_KHR);
                    cmCaps.wave32Pinnable       = cap.subgroupSizeControl && cap.requiredSubgroupSizeCompute && cap.minSubgroupSize <= 32u && 32u <= cap.maxSubgroupSize;
                    cmCaps.vulkanMemoryModel    = cap.vulkanMemoryModel;
                    cmCaps.selfCheckPassed      = true; // provisionally; the on-device check runs below only when the rule matches
                    const bool denseRank2Batch1 = !hasView && !aWas1D && !bWas1D && M > 0 && N > 0 && pc.total == (int) (M * N);
                    const bool hasBiasOrEpi     = node.fusedBias != kNoTensor || node.attr.has("pw_steps");
                    const int  hintValue        = env.config ? env.config->hint(Hint::CoopmatGemm, (int) Mode::Auto) : (int) Mode::Auto;
                    coopKind                    = coopmatGemmRoute(cmCaps, hintValue, env.useFp16, denseRank2Batch1, hasBiasOrEpi, M, N, K, g.isInitializer(node.inputs[1]));
                    // The 2-D coopmat dispatch has no runtime X-overflow rescue; a grid past the
                    // device limit keeps the SSBO kernels.
                    if (coopKind != CoopmatGemmKind::None && ((uint32_t) (N / 32) > cap.maxWorkGroupCount[0] || (uint32_t) (M / 32) > cap.maxWorkGroupCount[1]))
                    {
                        coopKind = CoopmatGemmKind::None;
                    }
                    if (coopKind != CoopmatGemmKind::None && !coopmatGemmSelfCheckPassed(env))
                    {
                        coopKind = CoopmatGemmKind::None;
                    }
                }

                // Upload a constant operand flat (row-major NCHW fp32 -> device, fp16 when half precision).
                // Direct fp16->fp16 passthrough when the stored weight already matches compute precision.
                // A low-precision coopmat route replaces the B upload with host-quantized codes below.
                auto maybeUpload = [&](TensorId t, int which, const Shape &s) {
                    if (!g.isInitializer(t))
                    {
                        return;
                    }
                    constBuf[which] = uploadInit(env, t, s);
                };
                maybeUpload(node.inputs[0], 0, g.desc(node.inputs[0]).shape);
                if (coopKind != CoopmatGemmKind::Fp8 && coopKind != CoopmatGemmKind::Int8)
                {
                    maybeUpload(node.inputs[1], 1, g.desc(node.inputs[1]).shape);
                }

                if (coopKind != CoopmatGemmKind::None)
                {
                    prepareCoopmat(node, env);
                    return;
                }

                // Use the register-blocked tiled GEMM for the standard (non-mat-vec) case with large
                // enough matrices; it assumes M at out[rank-2], N at out[rank-1], so the batch dims are
                // exactly out[0..rank-3] (true when neither operand was 1-D). Tiny / mat-vec / 1-D cases
                // keep the naive 1-thread/output kernel. fusePointwiseChains mirrors this predicate via
                // the same constant (core/matmul_tile.h). A view never tiles: the tiled kernels hardcode
                // dense row-major panels, and the fold pass only claims non-tiled-class shapes anyway.
                useTiled = !hasView && !aWas1D && !bWas1D && M >= kTiledMatMulMin && N >= kTiledMatMulMin && K >= kTiledMatMulMin;
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

                // A mat-vec too narrow to fill the naive grid takes the split-K kernel. B must keep its
                // n axis last (bWas1D strips it), both so the GEMV_NX lanes of a row stay coalesced and
                // so the _bias variant's `gid % N` indexes the bias.
                bool gemvShape = !useTiled && !bWas1D && M == 1 && K >= kGemvMinK && N < kGemvMaxN;

                // The two mat-vec kernels spend a different number of invocations on the same kGemvNx
                // outputs, and a workgroup above maxComputeWorkGroupInvocations fails pipeline creation
                // rather than degrading. Vulkan guarantees only 128, so each kernel is gated on the limit
                // it needs: matmul_gemv4 packs kGemvVec adjacent n into one lane and needs
                // kGemvNx/kGemvVec x kGemvKs invocations; matmul_gemv gives every n its own lane and needs
                // kGemvNx x kGemvKs. An N indivisible by kGemvVec on a device too small for the scalar
                // kernel keeps matmul.comp -- correct everywhere, merely slower.
                const uint32_t maxInv = env.ctx->caps().maxWorkGroupInvocations;

                // An N divisible by kGemvVec lets a mat-vec lane pull its kGemvVec adjacent n as one
                // vector element of B (a quarter of the lanes, four times the bytes per load
                // instruction, same kGemvNx outputs per workgroup). The two mat-vec kernels are
                // bit-identical, so an indivisible N simply falls back to the scalar one. A view keeps
                // the vector loads only when its B addressing stays contiguous and 4-aligned along n.
                bool useGemvVec = gemvShape && N % kGemvVec == 0 && maxInv >= (uint32_t) (kGemvNx / kGemvVec * kGemvKs) && (!hasView || viewGemv4Ok);

                useGemv = useGemvVec || (gemvShape && maxInv >= (uint32_t) (kGemvNx * kGemvKs));

                // The default {128,128,16} tile runs the compile-time matmul_tiled_fast kernel (full
                // inner-loop unroll, register-resident micro-tile — main's fast literal geometry);
                // only a non-default raced tile runs the spec-constant matmul_tiled kernel. The
                // tiled kernels are byte-identical at the default geometry (see core/matmul_tile.h).
                bool useFastTiled = useTiled && isDefaultMatMulTile(tile);

                // A fused Linear bias (rank-1 [N]) is added in the fp32 accumulator by the _bias kernel
                // variant; upload it flat and bind it as a 4th buffer. The geometry SSBO follows the
                // operands (and the bias when present): binding 3 without bias, binding 4 with it.
                const char *gemvBase = useGemvVec ? "matmul_gemv4" : "matmul_gemv";
                const char *base     = useFastTiled ? "matmul_tiled_fast" : (useTiled ? "matmul_tiled" : (useGemv ? gemvBase : "matmul"));
                std::string name     = base;
                uint32_t    nbuf     = 4; // A, B, D, geometry
                if (node.fusedBias != kNoTensor)
                {
                    biasBuf = uploadInit(env, node.fusedBias, g.desc(node.fusedBias).shape);
                    name += "_bias";
                    nbuf = 5; // A, B, D, bias, geometry
                }

                // A pointwise chain (fusePointwiseChains) attached to this MatMul runs in the kernel's own
                // epilogue (shaders/pw_epilogue.glsl), appended at binding nbuf. The plan indexes the flat
                // row-major output world (MatMul's output is always row-major, never NC4HW4).
                epi.prepare(node, env, /*flat=*/true, out);
                name += epi.suffix();
                nbuf += epi.extraBufs();

                // The spec-constant tiled kernel takes its TM/TN/TK tile as specialization constants
                // 0/1/2; the fast kernel bakes {128,128,16} in as literal #defines (no spec words).
                std::vector<uint32_t> spec;
                if (useTiled && !useFastTiled)
                {
                    spec = {(uint32_t) tile.tm, (uint32_t) tile.tn, (uint32_t) tile.tk};
                }
                pipe = env.pipeline(shader(name.c_str(), env.useFp16), nbuf, sizeof(MatMulPC), spec);
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                if (useWq)
                {
                    recordWq(cmd, node, env);
                    return;
                }
                if (coopKind != CoopmatGemmKind::None)
                {
                    recordCoopmat(cmd, node, env);
                    return;
                }
                auto buf = [&](int e) {
                    return constBuf[e] ? constBuf[e].get() : env.devBuf(node.inputs[e]);
                };
                VkBuffer              dstHandle = env.devBuf(node.outputs[0])->handle();
                std::vector<VkBuffer> bufs {buf(0)->handle(), buf(1)->handle(), dstHandle};
                if (biasBuf)
                {
                    bufs.push_back(biasBuf->handle());
                }
                bufs.push_back(geom->handle()); // geometry SSBO, after the operands/bias and before the epilogue
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
                } else if (useGemv)
                {
                    // Split-K mat-vec: a workgroup covers kGemvNx output elements along the flat grid and
                    // spends its local y lanes on the k reduction, so only x counts groups. The 1-D grid
                    // keeps the dispatch's runtime X-overflow rescue (it spills into y, which the kernel
                    // folds back through gl_WorkGroupID.y).
                    pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(pc.total, (uint32_t) kGemvNx));
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
