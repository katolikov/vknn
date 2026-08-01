// Batched N-D MatMul on the FLAT row-major GPU path: out[...,m,n] = sum_k A[...,m,k]*B[...,k,n]
// with NumPy broadcasting on the batch dims. Three kernels serve it, by shape: the register-blocked
// tiled GEMM (shaders/matmul_tiled*.comp) for full matrices — the default tile in its f16vec4-load
// form (matmul_tiled_fast_v4*, byte-identical, plus a zero-padded "#wv4" weight repack for an
// unaligned N) when the matmulVec4Route alignment rule holds — the split-K mat-vec
// (shaders/matmul_gemv*.comp — vector-load when N % kGemvVec == 0, scalar otherwise) for a narrow
// M == 1 row, and otherwise one thread per output element walking K (shaders/matmul.comp). Either
// operand may be an activation or a constant initializer (e.g. a Linear weight); an initializer is
// uploaded flat in prepare(). The naive and tiled kernels mirror the CPU oracle's broadcast/stride
// math byte-for-byte; the split-K kernels regroup the k chain (deterministically, identically to each
// other) and so agree only to fp32 rounding.
#include "backend/vulkan/coopmat_check.h"
#include "backend/vulkan/vk_tune_model.h"
#include "backend/vulkan/vk_tune_race.h"
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
        // the decodable batch rank is unbounded and the push constant stays small. aKp/bNp are the
        // operands' physical row strides: K and N for the packed operands every scalar kernel reads,
        // the zero-padded Np of a "#wv4" repacked weight, or the padded last axis of a virtualized
        // activation (VkOpEnv::rowPad). Tail fields: only the vec4-load twins
        // (matmul_tiled_fast_v4*_fp16.comp) declare and consume them; every other MatMulPC kernel
        // declares the shared 28-byte prefix (a pipeline's push range may exceed the shader's
        // block). 36 bytes total, well under the 128-byte guaranteed push-constant minimum.
        struct MatMulPC {
            int rank, total, M, N, K, aK, bK, bNp, aKp;
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
                coopAbsmaxPipe = env.pipeline("lowp_absmax", 2, sizeof(CoopAbsmaxPC), std::vector<uint32_t> {flat::laneWidthPow2For(env.ctx->caps(), flat::kFlatLocalSize)});
                coopQuantPipe = env.pipeline(quantShader, 3, sizeof(int));
                pipe          = env.pipeline(gemmShader, 4, sizeof(CoopGemmPC), {}, /*requiredSubgroupSize=*/32);
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
                const Graph  &g       = *env.graph;
                const Shape  &sa      = g.desc(node.inputs[0]).shape;
                const Shape  &out     = g.desc(node.outputs[0]).shape;
                const int64_t M       = sa.size() >= 2 ? sa[sa.size() - 2] : 1;
                const int     format  = weightQuantFormat(node);
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
            // {128,128,16}). Tuning::None keeps the default; Fast/Heavy race the candidates on
            // scratch buffers through vk::raceCandidates (interleaved submits, order reversed every
            // other round, median per candidate) and persist the winning index in the tune table.
            // Every candidate is bit-neutral (the per-output fp32 K chain is one ascending-k
            // sequence for any tile), so the choice never affects output bits.
            // Bit-neutrality is numeric safety, not measurement safety: the device throttles
            // several-fold under sustained load, so a challenger must still clear the incumbent by
            // kTuneRaceMargin before it displaces a proven pick in the persisted tune table (the
            // conv races carry the same margin for the same reason).
            MatMulTile pickTile(VkOpEnv &env, bool hasBias, bool v4Default) {
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
                // The B scratch is sized by pc.bNp (== N unless the default candidate times the v4
                // kernel over a padded weight layout), so a v4 timing read at the padded row
                // stride stays in-bounds.
                size_t es       = env.useFp16 ? 2 : 4;
                size_t perBatch = ((size_t) pc.M * pc.K + (size_t) pc.K * pc.bNp + (size_t) pc.M * pc.N) * es;
                if (perBatch == 0 || perBatch > ((size_t) 256 << 20))
                {
                    return kMatMulTiles[0];
                }
                int  gzT = (int) std::min<int64_t>({(int64_t) numBatch, 64, std::max<int64_t>(1, (int64_t) (((size_t) 128 << 20) / perBatch))});
                auto mk  = [&](size_t bytes) {
                    return std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>(bytes, 16), vk::MemPref::kDeviceOnly);
                };
                auto                        sA    = mk((size_t) gzT * pc.M * pc.K * es);
                auto                        sB    = mk((size_t) gzT * pc.K * pc.bNp * es);
                auto                        sD    = mk((size_t) gzT * pc.M * pc.N * es);
                std::shared_ptr<vk::Buffer> sBias = hasBias ? mk((size_t) pc.N * es) : nullptr;
                vk::TuneTimer               timer(env);
                // Time each candidate with the kernel it will actually dispatch: the default tile
                // runs the compile-time _fast kernel — the vec4-load twin when prepare()'s v4
                // routing holds for this shape — and every other tile runs the spec-constant
                // kernel. Timing the default with any other kernel would mis-rank it against its
                // own real dispatch.
                const char *specBase = hasBias ? "matmul_tiled_bias" : "matmul_tiled";
                const char *fastBase = hasBias ? "matmul_tiled_fast_bias" : "matmul_tiled_fast";
                if (v4Default)
                {
                    fastBase = hasBias ? "matmul_tiled_fast_v4_bias" : "matmul_tiled_fast_v4";
                }
                uint32_t nbuf = hasBias ? 5 : 4; // + the geometry SSBO bound after the operands/bias
                // Entrants in kMatMulTiles order, minus the tiles whose dispatch does not fit; index
                // 0 is always the default tile (tileFits holds for it whenever the op dispatches).
                struct Entrant {
                    int      tileIndex;
                    uint32_t gxT, gyT;
                };
                std::vector<Entrant>        entrants;
                std::vector<vk::KernelCost> costs;
                // Cooperative panel loads: one workgroup stages TM*TK of A and TK*TN of B per K
                // step, so it issues K*(TM+TN) elements over the whole reduction and stores TM*TN.
                // The compulsory footprint is the three operand panels, identical for every tile.
                // Threads per tiled-GEMM workgroup (shaders/matmul_tiled.comp dispatches TILE x TILE).
                constexpr double kMatMulTiledThreads = 256.0;
                const double     elemsPerVec4        = 4.0;
                const double     streamFootprint     = (double) gzT * ((double) pc.M * pc.K + (double) pc.M * pc.N) / elemsPerVec4;
                const double     residentFootprint   = (double) gzT * (double) pc.K * (double) pc.N / elemsPerVec4;
                for (int ci = 0; ci < kMatMulTileCount; ++ci)
                {
                    const MatMulTile &t = kMatMulTiles[ci];
                    if (!tileFits(env, t))
                    {
                        continue;
                    }
                    const uint32_t gxT = (uint32_t) ((pc.N + t.tn - 1) / t.tn), gyT = (uint32_t) ((pc.M + t.tm - 1) / t.tm);
                    entrants.push_back({ci, gxT, gyT});
                    const double   wgroups = (double) gxT * (double) gyT * (double) gzT;
                    vk::KernelCost cost;
                    // The A panel and the output are the activation side; the B panel is the
                    // weight side, shared across the workgroups of one output column.
                    cost.streamVec4            = (wgroups * (double) pc.K * (double) t.tm + (double) gzT * (double) pc.M * (double) pc.N) / elemsPerVec4;
                    cost.residentVec4          = wgroups * (double) pc.K * (double) t.tn / elemsPerVec4;
                    cost.streamFootprintVec4   = streamFootprint;
                    cost.residentFootprintVec4 = residentFootprint;
                    cost.waves                 = wgroups * (double) kMatMulTiledThreads / 64.0;
                    costs.push_back(cost);
                }
                if (entrants.empty())
                {
                    return kMatMulTiles[0];
                }
                std::vector<double> ms = vk::racePruned(costs, vk::deviceTuneModel(env), [&](int index) {
                    const Entrant    &entrant = entrants[(size_t) index];
                    const MatMulTile &t       = kMatMulTiles[entrant.tileIndex];
                    // Built inside the timed lambda so a candidate the analytical prefilter dropped
                    // never compiles its spec-constant variant (env.pipeline memoises).
                    const bool            fast = isDefaultMatMulTile(t);
                    std::vector<uint32_t> spec = fast ? std::vector<uint32_t> {} : std::vector<uint32_t> {(uint32_t) t.tm, (uint32_t) t.tn, (uint32_t) t.tk};
                    auto                  pipe = env.pipeline(shader(fast ? fastBase : specBase, env.useFp16), nbuf, sizeof(MatMulPC), spec);
                    return timer.time([&](VkCommandBuffer cmd) {
                        std::vector<VkBuffer> bufs {sA->handle(), sB->handle(), sD->handle()};
                        if (sBias)
                        {
                            bufs.push_back(sBias->handle());
                        }
                        bufs.push_back(geom->handle()); // geometry SSBO (matches the real dispatch's binding count)
                        pipe->dispatch(cmd, bufs, &pc, sizeof(pc), entrant.gxT, entrant.gyT, (uint32_t) gzT);
                    });
                });
                // The default tile is the incumbent, so every other candidate is a challenger and
                // must clear the margin; a tie keeps the incumbent, and so does a noise-width win
                // the analytical model does not corroborate.
                const std::vector<double> model  = vk::modelEstimates(costs, vk::deviceTuneModel(env));
                int                       best   = entrants[0].tileIndex;
                double                    bestMs = ms[0];
                for (size_t ei = 1; ei < entrants.size(); ++ei)
                {
                    // Measured against the incumbent's own time rather than a running best, so the
                    // margin cannot compound and the outcome does not depend on list order. The
                    // margin is waived for a tile the model also ranks cheaper (see kTuneRaceMargin).
                    const double need = ms[0] * (model[ei] < model[0] ? 1.0 : vk::kTuneRaceMargin);
                    if (ms[ei] < need && ms[ei] < bestMs)
                    {
                        bestMs = ms[ei];
                        best   = entrants[ei].tileIndex;
                    }
                }
                VKNN_DEBUG << "autotune " << sig << " -> tile " << kMatMulTiles[best].tm << "x" << kMatMulTiles[best].tn << "x" << kMatMulTiles[best].tk << vk::raceTimes(ms);
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
                // Physical last-axis extents of the operand buffers (0 = packed) and the resolved
                // row strides the kernels index with. A view-addressed MatMul carries its own
                // geometry and is never handed a virtualized operand.
                int64_t aRowPad = 0, bRowPad = 0, aRow = 0, bRow = 0;
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
                    aRow        = K;
                    bRow        = N;
                } else
                {
                    // Dense derivation (core/matmul_tile.h): 1-D promotion, broadcast batch strides,
                    // matrix-axis strides. An operand whose activation buffer the segment allocated
                    // with a virtualized last axis (env.rowPad) enters with that PHYSICAL row
                    // stride, which scales every batch stride stepping over a whole matrix — the
                    // segment's padding rule predicted this same geometry before it sized the
                    // buffer, so the layout and the kernel can never disagree.
                    aRowPad          = env.rowPad ? env.rowPad(node.inputs[0]) : 0;
                    bRowPad          = env.rowPad ? env.rowPad(node.inputs[1]) : 0;
                    MatMulFlatGeom d = matmulFlatGeom(sa, sb, out, aRowPad, bRowPad);
                    aWas1D           = d.aWas1D;
                    bWas1D           = d.bWas1D;
                    M                = d.M;
                    N                = d.N;
                    K                = d.K;
                    aRow             = d.aRow;
                    bRow             = d.bRow;
                    rank             = d.rank;
                    pc.aK            = 1;            // A is [...,M,K] row-major -> stepping K moves by 1
                    pc.bK            = (int) d.bRow; // B is [...,K,N] row-major -> stepping K moves by one physical row
                    outDim           = d.outDim;
                    aStride          = d.aStride;
                    bStride          = d.bStride;
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
                    cmCaps.coopmatFp8Fp32Row16 = cap.shaderFloat8CoopMat && cap.hasCoopmatShape(16, 16, 16, (uint32_t) VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT, (uint32_t) VK_COMPONENT_TYPE_FLOAT32_KHR);
                    cmCaps.coopmatI8I32Row16 = cap.hasCoopmatShape(16, 16, 16, (uint32_t) VK_COMPONENT_TYPE_SINT8_KHR, (uint32_t) VK_COMPONENT_TYPE_SINT32_KHR);
                    cmCaps.wave32Pinnable = cap.subgroupSizeControl && cap.requiredSubgroupSizeCompute && cap.minSubgroupSize <= 32u && 32u <= cap.maxSubgroupSize;
                    cmCaps.vulkanMemoryModel = cap.vulkanMemoryModel;
                    cmCaps.selfCheckPassed   = true; // provisionally; the on-device check runs below only when the rule matches
                    // A virtualized operand keeps the SSBO kernels: the coopmat kernels load dense
                    // packed panels and have no physical-row-stride parameter.
                    const bool denseRank2Batch1 = !hasView && !aWas1D && !bWas1D && aRowPad == 0 && bRowPad == 0 && M > 0 && N > 0 && pc.total == (int) (M * N);
                    const bool hasBiasOrEpi     = node.fusedBias != kNoTensor || node.attr.has("pw_steps");
                    const int  hintValue        = env.config ? env.config->hint(Hint::CoopmatGemm, (int) Mode::Auto) : (int) Mode::Auto;
                    coopKind = coopmatGemmRoute(cmCaps, hintValue, env.useFp16, denseRank2Batch1, hasBiasOrEpi, M, N, K, g.isInitializer(node.inputs[1]));
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
                // A low-precision coopmat route replaces the B upload with host-quantized codes. On the
                // SSBO paths B's upload waits for the kernel choice below: the vec4 route may take the
                // "#wv4" repacked layout instead of the flat copy, and the flat upload releases the host
                // payload the repack computes from.
                auto maybeUpload = [&](TensorId t, int which, const Shape &s) {
                    if (!g.isInitializer(t))
                    {
                        return;
                    }
                    constBuf[which] = uploadInit(env, t, s);
                };
                maybeUpload(node.inputs[0], 0, g.desc(node.inputs[0]).shape);

                if (coopKind != CoopmatGemmKind::None)
                {
                    if (coopKind != CoopmatGemmKind::Fp8 && coopKind != CoopmatGemmKind::Int8)
                    {
                        maybeUpload(node.inputs[1], 1, g.desc(node.inputs[1]).shape);
                    }
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

                // ---- vec4-load kernel routing (deterministic alignment rule) ----
                // The default {128,128,16} tile dispatches the f16vec4-load twin of the fast kernel
                // when every global element index it would form is 4-aligned (matmulVec4Route,
                // core/matmul_tile.h): a 4-aligned physical A row stride, a 4-aligned physical B row
                // stride (N itself, the zero-padded Np of a repacked weight, or a virtualized
                // activation's padded last axis), and 4-aligned batch strides on both operands. The
                // twin is byte-identical to the scalar kernel, so the route trades load width only,
                // never bits. Precision-high (fp32) nodes and non-default raced tiles keep the
                // scalar kernels. The rule runs after A's upload above, so a payload shared with A
                // (already uploaded flat and released) refuses the repack.
                MatMulVec4Route v4Route;
                if (useTiled)
                {
                    const bool bIsInit          = g.isInitializer(node.inputs[1]);
                    bool       bPayloadResident = false;
                    if (bIsInit)
                    {
                        const HostBuffer &hb       = g.initializers.at(node.inputs[1]);
                        const int64_t     elemSize = g.desc(node.inputs[1]).dtype == DType::Float16 ? 2 : 4;
                        bPayloadResident           = (int64_t) hb.bytes.size() >= numElements(g.desc(node.inputs[1]).shape) * elemSize;
                    }
                    const int            batchRank = rank - 2;
                    std::vector<int32_t> aBatch(aStride.begin(), aStride.begin() + batchRank);
                    std::vector<int32_t> bBatch(bStride.begin(), bStride.begin() + batchRank);
                    v4Route = matmulVec4Route(env.useFp16, N, K, aBatch, bBatch, bIsInit, bPayloadResident, aRow, bRow);
                }
                // Set before pickTile: the race times the v4 kernel for the default candidate when
                // the route holds, and that kernel reads pc.aKp/pc.bNp.
                pc.aKp = (int) (v4Route.eligible ? v4Route.aKp : aRow);
                pc.bNp = (int) (v4Route.eligible ? v4Route.bNp : bRow);
                if (useTiled)
                {
                    // A virtualized operand pins the default tile: its padded layout exists only so
                    // the v4 twin can read it, and the segment's rule already proved the route
                    // holds, so there is nothing left to race.
                    tile = (aRowPad != 0 || bRowPad != 0) ? kMatMulTiles[0] : pickTile(env, node.fusedBias != kNoTensor, v4Route.eligible);
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
                // inner-loop unroll, register-resident micro-tile — main's fast literal geometry),
                // in its vec4-load form when the alignment route holds; only a non-default raced
                // tile runs the spec-constant matmul_tiled kernel. The tiled kernels are
                // byte-identical at the default geometry (see core/matmul_tile.h).
                bool useFastTiled = useTiled && isDefaultMatMulTile(tile);
                bool useV4        = useFastTiled && v4Route.eligible;
                if ((aRowPad != 0 || bRowPad != 0) && !useV4)
                {
                    // Contract backstop. A virtualized operand's buffer holds padded rows whatever
                    // kernel runs, and only the v4 twin and the fully stride-driven naive kernel
                    // read a physical row stride — the tiled scalar kernels and the split-K mat-vec
                    // assume packed rows. The segment's rule and the routing above agree by
                    // construction; if a future shape class ever splits them, drop to the naive
                    // kernel (correct at any stride, merely slower) instead of reading the pad as
                    // data.
                    useTiled     = false;
                    useFastTiled = false;
                    useGemvVec   = false;
                    useGemv      = false;
                }

                // B's upload, in the layout the chosen kernel reads. The vec4 route with an
                // unaligned N repacks the weight to the zero-padded row stride bNp through the
                // weight cache (key suffix "#wv4", so a warm start reuses the packed blob); every
                // other route uploads the flat packed copy the scalar kernels read.
                if (useV4 && v4Route.padB)
                {
                    const TensorId bId   = node.inputs[1];
                    const int64_t  bRows = numElements(g.desc(bId).shape) / N; // == K: the repack requires broadcast (size-1) batch dims
                    constBuf[1]          = uploadCached(env, node.name + "#wv4", [&] {
                        return padMatMulRowsVec4(initFloats(g, bId), bRows, N, v4Route.bNp);
                    });
                } else
                {
                    maybeUpload(node.inputs[1], 1, g.desc(node.inputs[1]).shape);
                }

                // A fused Linear bias (rank-1 [N]) is added in the fp32 accumulator by the _bias kernel
                // variant; upload it flat and bind it as a 4th buffer. The geometry SSBO follows the
                // operands (and the bias when present): binding 3 without bias, binding 4 with it.
                const char *gemvBase = useGemvVec ? "matmul_gemv4" : "matmul_gemv";
                const char *fastBase = useV4 ? "matmul_tiled_fast_v4" : "matmul_tiled_fast";
                const char *base     = useFastTiled ? fastBase : (useTiled ? "matmul_tiled" : (useGemv ? gemvBase : "matmul"));
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
                if (!useFastTiled && !useTiled && !useGemv)
                {
                    spec = {env.flatLocalSize}; // the naive kernel's workgroup width (spec 0), resolved at load
                }
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
                    pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(pc.total, env.flatLocalSize));
                }
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::MatMul, MatMulOp);
} // namespace vknn
