// ConvGemm: a Conv lowered by lowerConv to one implicit-GEMM kernel over the receptive field
// (out[m, oc] = sum_k patch(m, k) * Wt[k, oc], k channel-fastest). The weights arrive repacked
// [K][Cout] from convert time, both panels stage through LDS, and the fp32 K reduction runs in a
// fixed chunked order — deterministic, fp16-floor equivalent to the direct Conv kernel (the
// summation order differs). Serves the KxK shapes the Winograd path rejects (strided, dilated,
// shallow, non-square).
//
// The M tile is a specialization constant (16/32/64): a shape heuristic picks it under
// Tuning::None (bit-neutral — the tile only remaps threads to outputs), and Tuning::Fast/Heavy
// race the variants per shape. Tiny-M/deep-K shapes additionally race a split-K pair
// (conv_gemm_ksplit fp32 partials + conv_gemm_kreduce finish); split-K changes the summation
// order, so it is only ever selected by the race, never by default. Winners persist in the tune
// cache so warm runs are stable.
#include "backend/vulkan/vk_tune_race.h"
#include "core/conv_gemm_route.h"
#include "core/conv_geom.h"
#include "pw_plan.h"
#include "vk_op_common.h"
#include "vknn/logging.h"
#include <algorithm>
#include <functional>

namespace vknn {
    namespace {

        // Split-K reorders the fp32 summation (fp16-floor legal, not byte-identical), so it is never
        // auto-selected by a timing race - that would let thermal state change the output bits. The
        // split-K kernels stay available for an explicit request; pickVariant never chooses them.

        // Local workgroup size along x for the split-K reduce pass; matches local_size_x in
        // shaders/conv_gemm_kreduce.comp.
        constexpr uint32_t kKreduceLocalSize = 64;

        struct ConvGemmOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            // wt is the packed [K][Cout] panel the split-K partial pass reads; wtV4 is what the
            // single-pass kernel binds — the same buffer when the panel is already 4-aligned on
            // Cout, else the zero-padded [K][coutP] repack the vec4-weight twin needs.
            std::shared_ptr<vk::Buffer>          wt, wtV4, bs;
            ConvGemmPC                           pc {};
            PwEpi                                epi;
            uint32_t                             gx = 0, gy = 0, gz = 0;
            bool                                 wv4 = false; // the single pass runs conv_gemm_wv4

            // --- split-K state (partial pass + reduce pass) ---
            bool                                 ksplit = false;
            std::shared_ptr<vk::ComputePipeline> ksPipe, krPipe;
            std::shared_ptr<vk::Buffer>          partBuf;
            ConvGemmKsPC                         kspc {};
            ConvGemmKrPC                         krpc {};
            int64_t                              krGroups = 0;

            // Deterministic split geometry: S chunks of chunkK (a TK multiple, so the in-chunk
            // stepping matches the single-pass kernel's), recomputed from K so a cached choice
            // reproduces the same partial layout on every load.
            static void splitGeom(int64_t K, int &S, int &chunk) {
                int64_t s = std::min<int64_t>(std::max<int64_t>(K / 1024, 2), 8);
                int64_t c = (((K + s - 1) / s + kConvGemmTileK - 1) / kConvGemmTileK) * kConvGemmTileK;
                S         = (int) ((K + c - 1) / c);
                chunk     = (int) c;
            }

            // Pick the M tile (16/32/64) for this shape. The M tile is bit-neutral (it only remaps
            // threads to outputs), so Fast/Heavy race it for throughput and cache the winner; None takes
            // the shape heuristic. Split-K (value 1) is NEVER auto-selected: it reorders the K summation
            // (fp16-floor, not byte-identical), so letting a timing race pick it would make the output
            // depend on thermal state and tuning level. A stale cached split-K value is ignored.
            int pickVariant(VkOpEnv &env, NCHW x, NCHW y, int64_t M, int64_t K, int64_t Cout, const char *raceStem, vk::Buffer &raceWeights) {
                (void) K;
                int  heur = convGemmTileM(M);
                char buf[112];
                // The vec4-weight route gets its own signature namespace: a tile raced on one load
                // width must never be reused for the other (bit-neutral either way, but the winner
                // is a throughput measurement of the kernel it was raced on).
                snprintf(buf, sizeof(buf), "cgemm%s_%d_%d_%d", wv4 ? "w" : "", (int) M, (int) K, (int) Cout);
                std::string sig = env.gpuTag + "/" + buf;
                int         reuse;
                if (env.reuseTuned(sig, reuse) && (reuse == 16 || reuse == 32 || reuse == 64))
                {
                    return reuse;
                }
                if (env.tuning == Tuning::None || !env.runner)
                {
                    return heur;
                }
                int64_t Cinb = cBlocks(x.c), Coutb = cBlocks(Cout);
                int     es   = env.useFp16 ? 2 : 4;
                auto    mk   = [&](size_t bytes) {
                    return std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>(bytes, 16), vk::MemPref::kDeviceOnly);
                };
                auto sSrc   = mk((size_t) x.n * Cinb * x.h * x.w * 4 * es);
                auto sDst   = mk((size_t) y.n * Coutb * y.h * y.w * 4 * es);
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
                uint32_t gxT    = (uint32_t) ((Cout + kConvGemmTileN - 1) / kConvGemmTileN);
                // Entrants with the shape heuristic's tile first: it is what Tuning::None dispatches,
                // so seeding the race with it keeps a race that resolves nothing on the default. The
                // whole list is timed interleaved (vk::raceCandidates), so no tile is measured on a
                // systematically warmer GPU than another.
                struct Entrant {
                    int                                  tm;
                    std::shared_ptr<vk::ComputePipeline> pipe;
                    uint32_t                             gyT;
                };
                std::vector<Entrant> entrants;
                for (int tm: {heur, 16, 32, 64})
                {
                    int64_t gyT = (M + tm - 1) / tm;
                    if (gyT > 65535)
                    {
                        continue; // the Y axis has no runtime split
                    }
                    if (std::any_of(entrants.begin(), entrants.end(), [&](const Entrant &e) {
                            return e.tm == tm;
                        }))
                    {
                        continue; // heur repeats one of the three tiles
                    }
                    entrants.push_back({tm, env.pipeline(shader(raceStem, env.useFp16), 4, sizeof(ConvGemmPC), {(uint32_t) tm}), (uint32_t) gyT});
                }
                if (entrants.empty())
                {
                    return heur;
                }
                std::vector<double> ms     = vk::raceCandidates((int) entrants.size(), [&](int index) {
                    const Entrant &entrant = entrants[(size_t) index];
                    return timeIt([&](VkCommandBuffer cmd) {
                        entrant.pipe->dispatch(cmd, {sSrc->handle(), raceWeights.handle(), pc.hasBias ? bs->handle() : sDst->handle(), sDst->handle()}, &pc, sizeof(pc), gxT,
                                               entrant.gyT, (uint32_t) x.n);
                        vk::computeBarrier(*env.ctx, cmd);
                    });
                });
                int                 best   = entrants[0].tm;
                double              bestMs = ms[0];
                for (size_t ei = 1; ei < entrants.size(); ++ei)
                {
                    if (ms[ei] < bestMs)
                    {
                        bestMs = ms[ei];
                        best   = entrants[ei].tm;
                    }
                }
                VKNN_DEBUG << "autotune " << sig << " -> tm=" << best;
                if (env.weights)
                {
                    env.weights->setTuned(sig, best, (int) env.tuning);
                }
                return best;
            }

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g   = *env.graph;
                NCHW         x   = NCHW::from(g.desc(node.inputs[0]).shape);
                Shape        out = g.desc(node.outputs[0]).shape;
                NCHW         y   = NCHW::from(out);
                auto         a   = [&](const char *k, std::vector<int64_t> d) {
                    const auto &v = node.attr.getints(k);
                    return v.empty() ? d : v;
                };
                auto k = a("kernel_shape", {1, 1}), st = a("strides", {1, 1});
                auto dl = a("dilations", {1, 1});
                // Shared forward geometry (core/conv_geom.h): resolves auto_pad into begin/end pads.
                auto p = convGeom(x.h, x.w, k[0], k[1], node.attr).pads();

                // Bias presence bounds by pwCoreInputs: inputs appended past it are fused-unit
                // operands, and reading one as the bias would double-apply it.
                bool hasBias = pwCoreInputs(node) > 2 && node.inputs[2] != kNoTensor;
                pc = {(int) x.c,  (int) x.h,  (int) x.w,  (int) y.c,  (int) y.h,  (int) y.w,
                      (int) k[0], (int) k[1], (int) st[0], (int) st[1], (int) p[0], (int) p[1],
                      (int) dl[0], (int) dl[1], (int) node.fusedAct, hasBias ? 1 : 0,
                      node.actLo, node.actHi};

                wt = uploadInit(env, node.inputs[1], g.desc(node.inputs[1]).shape);
                if (pc.hasBias)
                {
                    bs = uploadInit(env, node.inputs[2], g.desc(node.inputs[2]).shape);
                }
                epi.prepare(node, env, false, out);

                int64_t M = y.h * y.w;
                int64_t K = x.c * k[0] * k[1];

                // ---- vec4-weight kernel routing (deterministic alignment rule) ----
                // The single-pass kernel dispatches the STORE4-load twin conv_gemm_wv4 when the
                // weight panel's physical row stride is 4-aligned (convGemmWVec4Route,
                // core/conv_gemm_route.h). A Cout indivisible by four routes only when the lowered
                // [K][Cout] initializer's payload is still readable, since the twin then needs a
                // zero-padded [K][coutP] repack. The twin is byte-identical to conv_gemm — it
                // widens the weight load and nothing else — so the rule needs no race and no
                // accuracy gate.
                const TensorId wtId       = node.inputs[1];
                const int64_t  wtElemSize = g.desc(wtId).dtype == DType::Float16 ? 2 : 4;
                const bool     wtResident = (int64_t) g.initializers.at(wtId).bytes.size() >= numElements(g.desc(wtId).shape) * wtElemSize;
                const ConvGemmWVec4Route wRoute = convGemmWVec4Route(y.c, wtResident);
                pc.Coutp                        = (int) (wRoute.eligible ? wRoute.coutP : y.c);
                wv4                             = wRoute.eligible;
                // The padded repack goes through the weight cache (key suffix "#gemmwp", so a warm
                // start reuses the packed blob and never collides with the flat upload's entry).
                wtV4 = wt;
                if (wRoute.padW)
                {
                    wtV4 = uploadCached(env, g.desc(wtId).name + "#gemmwp", [&] {
                        return padConvGemmWeightVec4(initFloats(g, wtId), K, y.c, wRoute.coutP);
                    });
                }

                // The M-tile race times the kernel this node will actually dispatch, so a routed
                // node picks its tile under its own load width.
                const char *raceStem = wv4 ? "conv_gemm_wv4" : "conv_gemm";
                int         choice   = pickVariant(env, x, y, M, K, y.c, raceStem, wv4 ? *wtV4 : *wt);
                ksplit               = (choice == 1);
                int tm               = ksplit ? convGemmTileM(M) : choice;
                if (ksplit)
                {
                    wv4 = false; // the split-K partial pass has no vec4-weight twin
                }

                gx = (uint32_t) ((y.c + kConvGemmTileN - 1) / kConvGemmTileN);
                gy = (uint32_t) ((M + tm - 1) / tm);
                gz = (uint32_t) y.n;
                if (ksplit)
                {
                    int S = 0, chunk = 0;
                    splitGeom(K, S, chunk);
                    int64_t Coutb = cBlocks(y.c);
                    kspc = {pc.C, pc.H, pc.W, pc.Cout, pc.OH, pc.OW, pc.KH, pc.KW, pc.SH, pc.SW, pc.PT, pc.PL, pc.DH, pc.DW, S, chunk};
                    krpc = {(int) y.n, pc.Cout, (int) M, S, pc.act, pc.hasBias, pc.actLo, pc.actHi};
                    partBuf  = std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>((size_t) y.n * S * Coutb * M * 4 * 4, 16), vk::MemPref::kDeviceOnly);
                    krGroups = groups(y.n * Coutb * M, kKreduceLocalSize);
                    gz       = (uint32_t) (y.n * S);
                    ksPipe = env.pipeline(shader("conv_gemm_ksplit", env.useFp16), 3, sizeof(ConvGemmKsPC), {(uint32_t) tm});
                    krPipe = env.pipeline(shader((std::string("conv_gemm_kreduce") + epi.suffix()).c_str(), env.useFp16), 3 + epi.extraBufs(), sizeof(ConvGemmKrPC));
                } else
                {
                    pipe = env.pipeline(shader((std::string(wv4 ? "conv_gemm_wv4" : "conv_gemm") + epi.suffix()).c_str(), env.useFp16), 4 + epi.extraBufs(), sizeof(ConvGemmPC), {(uint32_t) tm});
                }
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *dst = env.devBuf(node.outputs[0]);
                if (ksplit)
                {
                    // partial pass (no bias/act/epilogue) -> reduce pass (finish + epilogue).
                    ksPipe->dispatch(cmd, {env.devBuf(node.inputs[0])->handle(), wt->handle(), partBuf->handle()}, &kspc, sizeof(kspc), gx, gy, gz);
                    vk::computeBarrier(*env.ctx, cmd);
                    std::vector<VkBuffer> rb {partBuf->handle(), pc.hasBias ? bs->handle() : dst->handle(), dst->handle()};
                    epi.append(rb, node, env, dst->handle());
                    krPipe->dispatch(cmd, rb, &krpc, sizeof(krpc), (uint32_t) krGroups);
                    return;
                }
                std::vector<VkBuffer> bufs {env.devBuf(node.inputs[0])->handle(), (wv4 ? wtV4 : wt)->handle(), pc.hasBias ? bs->handle() : dst->handle(), dst->handle()};
                epi.append(bufs, node, env, dst->handle());
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), gx, gy, gz);
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::ConvGemm, ConvGemmOp);
} // namespace vknn
