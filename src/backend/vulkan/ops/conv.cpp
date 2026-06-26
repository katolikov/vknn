// Conv2D on the GPU. One op handles both the group==1 case (the "conv" shader, which also
// covers 1x1 pointwise) and the depthwise case (the "dwconv" shader). Weights are repacked to
// NC4HW4 on the host and uploaded once. For the group==1 path we also autotune the workgroup
// size the first time we see a given shape and cache the winner.
#include "vk_op_common.h"
#include "vknn/logging.h"
#include <cstdlib>
#include <functional>

namespace vknn {
    namespace {

        // Read an advanced kernel hint from the session Config (no env vars; see include/vknn/config.h).
        inline int cfgHint(const VkOpEnv &env, Hint h) {
            return env.config ? env.config->hint(h) : 0;
        }

        struct ConvOp: VulkanOp {
            static constexpr int                 kTile     = 4; // output pixels per thread in the 1x1 kernel (matches shader)
            bool                                 depthwise = false;
            bool                                 pointwise = false;
            bool                                 winograd  = false;
            bool                                 splitk    = false;
            bool                                 reg       = false; // register-tiled implicit-im2col general conv (WTILE pixels/thread)
            bool                                 lds       = false; // LDS input-halo 3x3 (8x8 tile/workgroup)
            int64_t                              ldsGroups = 0;
            bool                                 hasRes    = false; // residual Add fused into the epilogue (out = act(conv + residual))
            std::unique_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          wbuf, bbuf;
            ConvPC                               pc {};
            DwPC                                 dpc {};
            int64_t                              total     = 0;
            uint32_t                             localSize = 64;

            // --- split-K 1x1 (deep, low-parallelism convs): partial pass + reduce pass ---
            std::unique_ptr<vk::ComputePipeline> skPipe, skRed;
            std::shared_ptr<vk::Buffer>          partBuf;
            SplitKPC                             skPC {};
            ReducePC                             skRedPC {};
            int64_t                              skGroups = 0, skRedGroups = 0;

            void prepareSplitK(const Node &node, VkOpEnv &env, NCHW x, NCHW y, int64_t Cout, int64_t Coutb) {
                int64_t Cinb = cBlocks(x.c), HW = y.h * y.w;
                // pick KPARTS so the partial pass has enough threads to fill the GPU (~8192), capped by Cinb.
                int64_t kparts = (8192 + Coutb * HW - 1) / (Coutb * HW);
                kparts         = std::max<int64_t>(2, std::min<int64_t>({kparts, Cinb, 16}));
                int64_t chunk  = (Cinb + kparts - 1) / kparts;
                skPC           = {(int) x.c, (int) Cout, (int) HW, (int) kparts, (int) chunk};
                skRedPC        = {(int) Cout, (int) HW, (int) kparts, (int) node.fusedAct, node.actLo, node.actHi};
                partBuf        = std::make_shared<vk::Buffer>(*env.ctx, (size_t) kparts * Coutb * HW * 4 * 2, vk::MemPref::kDeviceOnly);
                skGroups       = groups(kparts * Coutb * HW, 64);
                skRedGroups    = groups(Coutb * HW, 64);
                skPipe = std::make_unique<vk::ComputePipeline>(*env.ctx, "conv1x1_splitk_fp16", 3, sizeof(SplitKPC), std::vector<uint32_t> {}, env.cache->handle());
                skRed = std::make_unique<vk::ComputePipeline>(*env.ctx, "conv1x1_reduce_fp16", hasRes ? 4 : 3, sizeof(ReducePC), std::vector<uint32_t> {(uint32_t) (hasRes ? 1 : 0)},
                                                              env.cache->handle());
            }

            // --- Winograd F(2x2,3x3) state (3x3, stride 1, pad 1, group 1, fp16) ---
            // Three structural variants were implemented and verified (cosine 1.0 on ResNet-50) but ALL
            // regress vs the direct 3x3 kernel (~11.2 ms GPU) on this driver — Winograd's 2.25x FLOP saving
            // can't overcome the driver's punishment of memory/register/LDS pressure. DO NOT RE-TRY:
            //   - 2-pass (VKNN_WINOGRAD): input transform -> V in GLOBAL, then fused matmul+output transform
            //     with 16 register accumulators/thread. ~15 ms. MEMORY-bound: V is ~4x the input for F(2,3)
            //     and round-trips to DRAM. Splitting the 16 accumulators across 4 cooperating threads
            //     (VKNN_WINO2, wino_fused2) does NOT help -> confirms it's bandwidth, not occupancy.
            //   - fully-fused (VKNN_WINOFULL, wino_full): one kernel, V staged in LDS (no global round-trip).
            //     ~88 ms. OCCUPANCY collapse: the 16 KB static sV LDS array caps workgroups/SM to ~2-3 and
            //     starves the GPU. This is the "production fused-LDS Winograd" approach; it dies here.
            // The variants are kept behind env flags (default OFF) as a documented negative result.
            // Two passes: input transform -> V (global), then a FUSED matmul+output-transform kernel that
            // keeps the 16 transform-domain accumulators in registers (no M global buffer).
            std::unique_ptr<vk::ComputePipeline> wInPipe, wFusedPipe, wFullPipe;
            std::shared_ptr<vk::Buffer>          ubuf, vbuf;
            WinoInPC                             wInPC {};
            WinoFusedPC                          wFusedPC {};
            int64_t                              wInGroups = 0, wFusedGroups = 0, wFullGroups = 0;
            bool                                 wino2    = false; // occupancy-friendly 2-pass matmul (4 threads/tile, M split across quads)
            bool                                 winofull = false; // single fully-fused kernel: V stays in LDS, no global round-trip
            // 3-pass with a TILED batched GEMM for the transform-domain multiply (MNN's structure).
            bool                                 winogemm = false;
            int                                  winoUnit = 2; // output-tile edge: 2 = F(2,3) (16 pts), 4 = F(4,3) (36 pts, 0.56x V/M traffic)
            std::unique_ptr<vk::ComputePipeline> wGemmPipe, wOutPipe;
            std::shared_ptr<vk::Buffer>          mbuf;
            WinoGemmPC                           wGemmPC {};
            int64_t                              wGemmGX = 0, wGemmGY = 0, wGemmGZ = 0;

            void prepareWinograd(const Node &node, VkOpEnv &env, NCHW x, NCHW y, int64_t Cout, int64_t Coutb) {
                const Graph       &g   = *env.graph;
                int64_t            Cin = x.c, Cinb = cBlocks(x.c);
                const int          U_ = winoUnit, A_ = U_ + 2; // output edge, transform-domain edge (4 or 6)
                const int          nPos = A_ * A_;             // 16 (F2,3) or 36 (F4,3)
                int64_t            nTH = (y.h + U_ - 1) / U_, nTW = (y.w + U_ - 1) / U_, nT = x.n * nTH * nTW;
                std::vector<float> wsrcv = initFloats(g, node.inputs[1]);
                const float       *wsrc  = wsrcv.data();

                // Filter transform matrix G (A_ x 3): F(2,3) and F(4,3).
                static const float G2[4][3] = {{1, 0, 0}, {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0, 0, 1}};
                static const float G4[6][3] = {
                    {0.25f, 0, 0}, {-1.f / 6, -1.f / 6, -1.f / 6}, {-1.f / 6, 1.f / 6, -1.f / 6}, {1.f / 24, 1.f / 12, 1.f / 6}, {1.f / 24, -1.f / 12, 1.f / 6},
                    {0, 0, 1}};
                // Host weight transform U = G g G^T, packed [pos][oc][icb] vec4(4 ic). Cached on disk.
                ubuf = uploadCached(env, node.name + "#wino" + std::to_string(U_), [&] {
                    std::vector<float> U((size_t) nPos * Cout * Cinb * 4, 0.f);
                    for (int64_t oc = 0; oc < Cout; ++oc)
                    {
                        for (int64_t ic = 0; ic < Cin; ++ic)
                        {
                            const float *gk = wsrc + (oc * Cin + ic) * 9; // 3x3
                            float        Gg[6][3];
                            for (int i = 0; i < A_; ++i)
                            {
                                for (int j = 0; j < 3; ++j)
                                {
                                    const float *Gi = (U_ == 2) ? G2[i] : G4[i];
                                    Gg[i][j]        = Gi[0] * gk[j] + Gi[1] * gk[3 + j] + Gi[2] * gk[6 + j];
                                }
                            }
                            int64_t icb = ic / 4, lane = ic % 4;
                            for (int i = 0; i < A_; ++i)
                            {
                                for (int j = 0; j < A_; ++j)
                                {
                                    const float *Gj                                  = (U_ == 2) ? G2[j] : G4[j];
                                    float        u                                   = Gg[i][0] * Gj[0] + Gg[i][1] * Gj[1] + Gg[i][2] * Gj[2];
                                    int          pos                                 = i * A_ + j;
                                    U[(((pos * Cout + oc) * Cinb + icb) * 4) + lane] = u;
                                }
                            }
                        }
                    }
                    return U;
                });

                wFusedPC = {(int) x.n, (int) Cin, (int) Cout, (int) y.h, (int) y.w, (int) nTH, (int) nTW, (int) node.fusedAct, node.actLo, node.actHi};
                int wvar = cfgHint(env, Hint::kWinogradVariant); // 0=tiled-GEMM 1=fused 2=fused-split 3=full
                winofull = (wvar == 3);
                if (winofull)
                {
                    // Single fused kernel: one workgroup per (tile, ocb-group of 16). No V buffer, no input pass.
                    int64_t ocbGroups = (Coutb + 15) / 16;
                    wFullGroups       = nT * ocbGroups;
                    wFullPipe = std::make_unique<vk::ComputePipeline>(*env.ctx, "wino_full_fp16", 4, sizeof(WinoFusedPC), std::vector<uint32_t> {}, env.cache->handle());
                    return;
                }
                int el    = 2; // fp16
                vbuf      = std::make_shared<vk::Buffer>(*env.ctx, (size_t) nPos * Cinb * nT * 4 * el, vk::MemPref::kDeviceOnly);
                wInPC     = {(int) x.n, (int) x.c, (int) x.h, (int) x.w, (int) y.h, (int) y.w, (int) nTH, (int) nTW};
                wInGroups = groups(Cinb * nT, 64);

                // The tiled-GEMM 3-pass is the production Winograd kernel (default, variant 0). The fused /
                // fused-split variants are kept only as Hint-gated experiments (documented regressions above).
                winogemm = (wvar == 0);
                if (winogemm)
                {
                    // 3-pass: input transform -> V, TILED batched GEMM -> M, output transform -> dst.
                    mbuf             = std::make_shared<vk::Buffer>(*env.ctx, (size_t) nPos * nT * Coutb * 4 * el, vk::MemPref::kDeviceOnly);
                    wGemmPC          = {(int) Cin, (int) Cout, (int) nT};
                    const int TILE_M = 32, TILE_NB = 8; // must match wino_gemm_fp16.comp (MDIM*RM, NDIM)
                    wGemmGX = groups(nT, TILE_M);       // workgroups over M (tiles)
                    wGemmGY = groups(Coutb, TILE_NB);   // workgroups over N (ocb)
                    wGemmGZ = nPos;                     // one GEMM per transform position (16 or 36)
                    wInPipe = std::make_unique<vk::ComputePipeline>(*env.ctx, U_ == 2 ? "wino_input_fp16" : "wino_input4_fp16", 2, sizeof(WinoInPC), std::vector<uint32_t> {},
                                                                    env.cache->handle());
                    wGemmPipe = std::make_unique<vk::ComputePipeline>(*env.ctx, "wino_gemm_fp16", 3, sizeof(WinoGemmPC), std::vector<uint32_t> {}, env.cache->handle());
                    wOutPipe = std::make_unique<vk::ComputePipeline>(*env.ctx, U_ == 2 ? "wino_out_fp16" : "wino_out4_fp16", 3, sizeof(WinoFusedPC), std::vector<uint32_t> {},
                                                                     env.cache->handle());
                    return;
                }
                wino2 = (wvar == 2);
                // wino2: 4 threads cooperate per (ocb,tile) -> 4x the threads, dispatched 64/workgroup.
                wFusedGroups = wino2 ? groups(Coutb * nT * 4, 64) : groups(Coutb * nT, 64);
                wInPipe = std::make_unique<vk::ComputePipeline>(*env.ctx, "wino_input_fp16", 2, sizeof(WinoInPC), std::vector<uint32_t> {}, env.cache->handle());
                wFusedPipe = std::make_unique<vk::ComputePipeline>(*env.ctx, wino2 ? "wino_fused2_fp16" : "wino_fused_fp16", 4, sizeof(WinoFusedPC), std::vector<uint32_t> {},
                                                                   env.cache->handle());
            }

            // Try a few workgroup sizes for this exact shape, keep the fastest, and remember it so the
            // next session (or run) skips the measurement. Only the group==1 conv shader is tunable.
            // The timing dispatches run on dedicated SCRATCH buffers, never the real activation buffers -
            // tuning must not write into the data path (doing so raced the first real run and corrupted it).
            uint32_t pickLocalSize(VkOpEnv &env) {
                if (env.tuning == TuningLevel::kOff)
                {
                    return 64;
                }
                char buf[96];
                snprintf(buf, sizeof(buf), "convls_%d_%d_%d_%d_%d_%d_%d_%d", pc.Cin, pc.H, pc.W, pc.Cout, pc.OH, pc.OW, pc.KH, pc.SH);
                std::string sig = buf;
                if (env.weights)
                {
                    int cached = env.weights->tuned(sig, 0);
                    if (cached > 0)
                    {
                        return (uint32_t) cached;
                    }
                }
                uint32_t best = 64;
                if (env.tuning != TuningLevel::kOff && env.runner)
                {
                    int    es       = env.useFp16 ? 2 : 4;
                    size_t srcBytes = (size_t) pc.N * cBlocks(pc.Cin) * pc.H * pc.W * 4 * es;
                    size_t dstBytes = (size_t) pc.N * cBlocks(pc.Cout) * pc.OH * pc.OW * 4 * es;
                    auto   sSrc     = std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>(srcBytes, 16), vk::MemPref::kDeviceOnly);
                    auto   sDst     = std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>(dstBytes, 16), vk::MemPref::kDeviceOnly);
                    std::vector<uint32_t> cands = (env.tuning == TuningLevel::kThorough) ? std::vector<uint32_t> {32, 64, 128, 256} : std::vector<uint32_t> {64, 128, 256};
                    double bestMs = 1e30;
                    for (uint32_t ls: cands)
                    {
                        vk::ComputePipeline p(*env.ctx, shader("conv", env.useFp16), 4, sizeof(ConvPC), {ls}, env.cache->handle());
                        VkCommandBuffer     cmd = env.runner->allocate();
                        env.runner->begin(cmd);
                        for (int rep = 0; rep < 8; ++rep)
                        {
                            p.dispatch(cmd, {sSrc->handle(), wbuf->handle(), bbuf->handle(), sDst->handle()}, &pc, sizeof(pc), groups(total, ls));
                        }
                        env.runner->end(cmd);
                        double ms = env.runner->submitAndWait(cmd);
                        vkFreeCommandBuffers(env.ctx->device(), env.runner->pool(), 1, &cmd);
                        if (ms < bestMs)
                        {
                            bestMs = ms;
                            best   = ls;
                        }
                    }
                    VKNN_DEBUG << "autotune " << sig << " -> local_size_x=" << best;
                }
                if (env.weights)
                {
                    env.weights->setTuned(sig, (int) best);
                }
                return best;
            }

            // Autotune the 3x3 conv kernel for THIS shape: 0 = direct, 1 = F(2,3) Winograd, 2 = F(4,3)
            // Winograd. Winograd wins big on deep, square 3x3 but loses on small-channel / spatially-large 3x3,
            // so the choice is measured per-shape on scratch buffers and cached like the local-size tune.
            // F(4,3) (0.56x the V/M traffic, 4x FLOP saving) is only considered when fp16-safe (allowF4).
            int tuneWino(VkOpEnv &env, NCHW x, NCHW y, int64_t Cin, int64_t Cout, int act) {
                if (env.winograd == WinogradMode::kOff)
                {
                    return 0;
                }
                if (cfgHint(env, Hint::kWinogradUnit) == 4)
                {
                    return 2; // force F(4,3) (research; numerically fine but register-heavy transforms)
                }
                bool forceOn = (env.winograd == WinogradMode::kOn);
                if (env.tuning == TuningLevel::kOff || !env.runner)
                {
                    return forceOn ? 1 : 0;
                }
                int64_t Cinb = cBlocks(Cin), Coutb = cBlocks(Cout);
                int64_t nTH = (y.h + 1) / 2, nTW = (y.w + 1) / 2, nT = x.n * nTH * nTW;
                char    buf[128];
                snprintf(buf, sizeof(buf), "wino_%d_%d_%d_%d", (int) Cin, (int) Cout, (int) y.h, (int) y.w);
                std::string sig = buf;
                if (env.weights)
                {
                    int c = env.weights->tuned(sig, -1);
                    if (c >= 0)
                    {
                        return c;
                    }
                }
                auto mk = [&](size_t bytes) {
                    return std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>(bytes, 16), vk::MemPref::kDeviceOnly);
                };
                auto                sSrc  = mk((size_t) x.n * Cinb * x.h * x.w * 8);
                auto                sU    = mk((size_t) 16 * Cout * Cinb * 8);
                auto                sV    = mk((size_t) 16 * Cinb * nT * 8);
                auto                sM    = mk((size_t) 16 * nT * Coutb * 8);
                auto                sBias = mk((size_t) Coutb * 8);
                auto                sWt   = mk((size_t) Cout * Cinb * 9 * 8);
                auto                sDst  = mk((size_t) x.n * Coutb * y.h * y.w * 8);
                ConvPC              dpc = {(int) x.n, (int) Cin, (int) x.h, (int) x.w, (int) Cout, (int) y.h, (int) y.w, 3, 3, 1, 1, 1, 1, 1, 1, act, 0.f, 0.f};
                WinoInPC            ipc = {(int) x.n, (int) Cin, (int) x.h, (int) x.w, (int) y.h, (int) y.w, (int) nTH, (int) nTW};
                WinoGemmPC          gpc = {(int) Cin, (int) Cout, (int) nT};
                WinoFusedPC         opc = {(int) x.n, (int) Cin, (int) Cout, (int) y.h, (int) y.w, (int) nTH, (int) nTW, act, 0.f, 0.f};
                vk::ComputePipeline inPipe(*env.ctx, "wino_input_fp16", 2, sizeof(WinoInPC), {}, env.cache->handle());
                vk::ComputePipeline gPipe(*env.ctx, "wino_gemm_fp16", 3, sizeof(WinoGemmPC), {}, env.cache->handle());
                vk::ComputePipeline oPipe(*env.ctx, "wino_out_fp16", 3, sizeof(WinoFusedPC), {}, env.cache->handle());
                uint32_t            gx = groups(nT, 32), gy = groups(Coutb, 8);
                auto                timeIt = [&](const std::function<void(VkCommandBuffer)> &rec) {
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
                double dms = 1e30;
                for (uint32_t ls: {64u, 128u, 256u})
                { // compare against the direct kernel's best local size
                    vk::ComputePipeline dPipe(*env.ctx, "conv_fp16", 4, sizeof(ConvPC), {ls}, env.cache->handle());
                    uint32_t            dg = groups(x.n * Coutb * y.h * y.w, ls);
                    dms                    = std::min(dms, timeIt([&](VkCommandBuffer cmd) {
                                       dPipe.dispatch(cmd, {sSrc->handle(), sWt->handle(), sBias->handle(), sDst->handle()}, &dpc, sizeof(dpc), dg);
                                       vk::computeBarrier(cmd);
                                                      }));
                }
                double wms    = timeIt([&](VkCommandBuffer cmd) {
                    inPipe.dispatch(cmd, {sSrc->handle(), sV->handle()}, &ipc, sizeof(ipc), groups(Cinb * nT, 64));
                    vk::computeBarrier(cmd);
                    gPipe.dispatch(cmd, {sV->handle(), sU->handle(), sM->handle()}, &gpc, sizeof(gpc), gx, gy, 16);
                    vk::computeBarrier(cmd);
                    oPipe.dispatch(cmd, {sM->handle(), sBias->handle(), sDst->handle()}, &opc, sizeof(opc), groups(Coutb * nT, 64));
                    vk::computeBarrier(cmd);
                });
                int    choice = (forceOn || wms < dms) ? 1 : 0;
                VKNN_DEBUG << "tuneWino " << sig << " direct=" << dms << " wino=" << wms << " -> " << choice;
                if (env.weights)
                {
                    env.weights->setTuned(sig, choice);
                }
                return choice;
            }

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g    = *env.graph;
                NCHW         x    = NCHW::from(g.desc(node.inputs[0]).shape);
                NCHW         y    = NCHW::from(g.desc(node.outputs[0]).shape);
                const Shape &ws   = g.desc(node.inputs[1]).shape; // [Cout, Cin/group, KH, KW]
                int64_t      Cout = ws[0], inCg = ws[1], KH = ws[2], KW = ws[3];
                auto         st    = attrInts(node, "strides", {1, 1});
                auto         pad   = attrInts(node, "pads", {0, 0, 0, 0});
                auto         dil   = attrInts(node, "dilations", {1, 1});
                int64_t      group = node.attr.geti("group", 1);
                hasRes             = (node.fusedResidual != kNoTensor); // set by the residual-Add fusion pass (1x1 only)
                depthwise          = (group == x.c && group == Cout && inCg == 1);
                pointwise = (!depthwise && group == 1 && KH == 1 && KW == 1 && st[0] == 1 && st[1] == 1 && pad[0] == 0 && pad[1] == 0 && pad[2] == 0 && pad[3] == 0);
                // Winograd F(2,3) via a tiled GEMM (the structure MNN's OpenCL backend uses). It beats the
                // direct kernel on deep, square 3x3 (ResNet-50: 12.6 -> 10.5 ms, matching MNN's tuned OpenCL)
                // but loses on small-channel / spatially-large 3x3, so tuneWino picks per-shape and caches.
                // Controlled by Config::winograd (kAuto default); NOT an env var.
                bool winoShape = (env.useFp16 && !depthwise && group == 1 && KH == 3 && KW == 3 && st[0] == 1 && st[1] == 1 && pad[0] == 1 && pad[1] == 1 && pad[2] == 1 && pad[3] == 1 && x.c >= 32 && Cout >= 32);
                int wchoice = winoShape ? tuneWino(env, x, y, x.c, Cout, (int) node.fusedAct) : 0;
                winograd    = (wchoice > 0);
                winoUnit    = (wchoice == 2) ? 4 : 2;

                std::vector<float> wsrcv = initFloats(g, node.inputs[1]);
                const float       *wsrc  = wsrcv.data();
                int64_t            Coutb = cBlocks(Cout);

                // bias, padded out to a multiple of 4 so the kernel can read whole vec4s
                bbuf = uploadCached(env, node.name + "#b", [&] {
                    std::vector<float> bias(Coutb * 4, 0.f);
                    // inputs[2] is bias unless it's the appended residual (no-bias + fused-residual case)
                    if (node.inputs.size() > 2 && node.inputs[2] != kNoTensor && node.inputs[2] != node.fusedResidual)
                    {
                        std::vector<float> bsrcv = initFloats(g, node.inputs[2]);
                        const float       *bsrc  = bsrcv.data();
                        for (int64_t i = 0; i < Cout; ++i)
                        {
                            bias[i] = bsrc[i];
                        }
                    }
                    return bias;
                });

                if (winograd)
                {
                    prepareWinograd(node, env, x, y, Cout, Coutb);
                    return;
                }

                if (depthwise)
                {
                    int64_t Cb = cBlocks(x.c);
                    wbuf       = uploadCached(env, node.name + "#w", [&] {
                        // [C,1,KH,KW] -> [Cb][KH][KW][4]
                        std::vector<float> wp(Cb * KH * KW * 4, 0.f);
                        for (int64_t c = 0; c < x.c; ++c)
                        {
                            int64_t cb = c / 4, l = c % 4;
                            for (int64_t ky = 0; ky < KH; ++ky)
                            {
                                for (int64_t kx = 0; kx < KW; ++kx)
                                {
                                    wp[(((cb * KH + ky) * KW + kx) * 4) + l] = wsrc[((c * KH + ky) * KW + kx)];
                                }
                            }
                        }
                        return wp;
                    });
                    dpc        = {(int) x.n,   (int) x.c,    (int) x.h,    (int) x.w,    (int) y.h,    (int) y.w,           (int) KH, (int) KW,   (int) st[0],
                                  (int) st[1], (int) pad[0], (int) pad[1], (int) dil[0], (int) dil[1], (int) node.fusedAct, 0,        node.actLo, node.actHi};
                    total      = x.n * Cb * y.h * y.w;
                    pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("dwconv", env.useFp16), 4, sizeof(DwPC), std::vector<uint32_t> {}, env.cache->handle());
                } else
                {
                    int64_t Cinb = cBlocks(x.c);
                    pc    = {(int) x.n,   (int) x.c,   (int) x.h,    (int) x.w,    (int) Cout,   (int) y.h,    (int) y.w,           (int) KH,   (int) KW,
                             (int) st[0], (int) st[1], (int) pad[0], (int) pad[1], (int) dil[0], (int) dil[1], (int) node.fusedAct, node.actLo, node.actHi};
                    total = x.n * Coutb * y.h * y.w;
                    wbuf  = uploadCached(env, node.name + "#w", [&] {
                        // [Cout,Cin,KH,KW] -> [Cout][Cinb][KH][KW][4]
                        std::vector<float> wp((size_t) Cout * Cinb * KH * KW * 4, 0.f);
                        for (int64_t oc = 0; oc < Cout; ++oc)
                        {
                            for (int64_t ic = 0; ic < x.c; ++ic)
                            {
                                int64_t icb = ic / 4, l = ic % 4;
                                for (int64_t ky = 0; ky < KH; ++ky)
                                {
                                    for (int64_t kx = 0; kx < KW; ++kx)
                                    {
                                        wp[(((((oc * Cinb + icb) * KH + ky) * KW + kx) * 4) + l)] = wsrc[(((oc * x.c + ic) * KH + ky) * KW + kx)];
                                    }
                                }
                            }
                        }
                        return wp;
                    });
                    if (pointwise)
                    {
                        // Deep, small-spatial 1x1 convs have too few threads for the register-tiled kernel; use
                        // split-K there (parallelize the channel reduction). Threshold = standard thread count.
                        int64_t HW         = y.h * y.w;
                        int64_t stdThreads = Coutb * ((HW + kTile - 1) / kTile);
                        splitk             = (env.useFp16 && x.n == 1 && x.c >= 32 && stdThreads < 2048);
                        if (splitk)
                        {
                            prepareSplitK(node, env, x, y, Cout, Coutb);
                        } else
                        {
                            int64_t nTiles = (HW + kTile - 1) / kTile;
                            total          = x.n * Coutb * nTiles;
                            pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("conv1x1", env.useFp16), hasRes ? 5 : 4, sizeof(ConvPC), std::vector<uint32_t> {(uint32_t) (hasRes ? 1 : 0)},
                                                                         env.cache->handle());
                        }
                    } else if (cfgHint(env, Hint::kDirectConv3x3) == 2 && env.useFp16 && KH == 3 && KW == 3 && st[0] == 1 && st[1] == 1 && pad[0] == 1 && pad[1] == 1 && pad[2] == 1 && pad[3] == 1 && dil[0] == 1 && dil[1] == 1 && y.h >= 14 && y.w >= 14)
                    {
                        // LDS input-halo 3x3 for the larger-spatial layers (input reuse via shared memory). 7x7
                        // layer4 stays on the direct kernel (tile barely fills, halo overhead dominates).
                        lds         = true;
                        int64_t nTX = (y.w + 7) / 8, nTY = (y.h + 7) / 8;
                        ldsGroups = x.n * Coutb * nTY * nTX;
                        pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, "conv3x3_lds_fp16", 4, sizeof(ConvPC), std::vector<uint32_t> {}, env.cache->handle());
                    } else if (cfgHint(env, Hint::kDirectConv3x3) == 1)
                    {
                        // register-tiled implicit-im2col (opt-in): regresses 3x3 on this GPU (small weight tensors
                        // already cache well; WTILE overhead + extra input loads dominate). Kept for experiments.
                        reg        = true;
                        int64_t HW = y.h * y.w;
                        total      = x.n * Coutb * ((HW + kTile - 1) / kTile);
                        pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("conv_reg", env.useFp16), 4, sizeof(ConvPC), std::vector<uint32_t> {},
                                                                     env.cache->handle());
                    } else
                    {
                        // autotuned 1-pixel-per-thread direct kernel (fastest 3x3 path on the target GPU so far)
                        total     = x.n * Coutb * y.h * y.w;
                        localSize = pickLocalSize(env);
                        pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("conv", env.useFp16), 4, sizeof(ConvPC), std::vector<uint32_t> {localSize},
                                                                     env.cache->handle());
                    }
                }
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *src = env.devBuf(node.inputs[0]);
                vk::Buffer *dst = env.devBuf(node.outputs[0]);
                if (winograd)
                {
                    if (winofull)
                    {
                        // single fully-fused dispatch: U, raw input src, bias, dst. V stays in LDS.
                        wFullPipe->dispatch(cmd, {ubuf->handle(), src->handle(), bbuf->handle(), dst->handle()}, &wFusedPC, sizeof(wFusedPC), (uint32_t) wFullGroups);
                        return;
                    }
                    if (winogemm)
                    {
                        // 3-pass: input transform -> V, tiled batched GEMM -> M, output transform -> dst.
                        wInPipe->dispatch(cmd, {src->handle(), vbuf->handle()}, &wInPC, sizeof(wInPC), (uint32_t) wInGroups);
                        vk::computeBarrier(cmd);
                        wGemmPipe->dispatch(cmd, {vbuf->handle(), ubuf->handle(), mbuf->handle()}, &wGemmPC, sizeof(wGemmPC), (uint32_t) wGemmGX, (uint32_t) wGemmGY, (uint32_t) wGemmGZ);
                        vk::computeBarrier(cmd);
                        wOutPipe->dispatch(cmd, {mbuf->handle(), bbuf->handle(), dst->handle()}, &wFusedPC, sizeof(wFusedPC),
                                           (uint32_t) groups(cBlocks(wFusedPC.Cout) * wInPC.nTH * wInPC.nTW * wFusedPC.N, 64));
                        return;
                    }
                    // 2 stages: input transform -> V, then fused matmul + output transform -> dst.
                    wInPipe->dispatch(cmd, {src->handle(), vbuf->handle()}, &wInPC, sizeof(wInPC), (uint32_t) wInGroups);
                    vk::computeBarrier(cmd);
                    wFusedPipe->dispatch(cmd, {ubuf->handle(), vbuf->handle(), bbuf->handle(), dst->handle()}, &wFusedPC, sizeof(wFusedPC), (uint32_t) wFusedGroups);
                    return;
                }
                // fused residual (1x1 path only); bias is a harmless dummy when not fused (shader won't read
                // it)
                VkBuffer res = (hasRes ? env.devBuf(node.fusedResidual) : bbuf.get())->handle();
                if (splitk)
                {
                    // partial pass (K-parallel) -> reduce pass (+bias [+residual] +act).
                    skPipe->dispatch(cmd, {src->handle(), wbuf->handle(), partBuf->handle()}, &skPC, sizeof(skPC), (uint32_t) skGroups);
                    vk::computeBarrier(cmd);
                    std::vector<VkBuffer> rb = {partBuf->handle(), bbuf->handle(), dst->handle()};
                    if (hasRes)
                    {
                        rb.push_back(res);
                    }
                    skRed->dispatch(cmd, rb, &skRedPC, sizeof(skRedPC), (uint32_t) skRedGroups);
                    return;
                }
                std::vector<VkBuffer> bufs = {src->handle(), wbuf->handle(), bbuf->handle(), dst->handle()};
                if (depthwise)
                {
                    pipe->dispatch(cmd, bufs, &dpc, sizeof(dpc), groups(total, 64));
                } else if (pointwise)
                {
                    if (hasRes)
                    {
                        bufs.push_back(res);
                    }
                    pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(total, 64));
                } else if (lds)
                {
                    pipe->dispatch(cmd, bufs, &pc, sizeof(pc), (uint32_t) ldsGroups);
                } else
                {
                    pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(total, reg ? 64 : localSize));
                }
            }
        };

    } // namespace

    VKNN_REGISTER_VK_OP(OpType::kConv, ConvOp);

} // namespace vknn
