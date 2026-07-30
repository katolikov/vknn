// Conv2D on the GPU. One op handles both the group==1 case (the "conv" shader, which also
// covers 1x1 pointwise) and the depthwise case (the "dwconv" shader). Weights are repacked to
// NC4HW4 on the host and uploaded once. For the group==1 path we also autotune the workgroup
// size the first time we see a given shape and cache the winner.
#include "backend/vulkan/vk_tune_model.h"
#include "backend/vulkan/vk_tune_race.h"
#include "core/conv_gemm_route.h"
#include "core/conv_geom.h"
#include "core/wino_f63.h"
#include "pw_plan.h"
#include "pw_splitk_rule.h"
#include "vk_op_common.h"
#include "vknn/logging.h"
#include <cstdlib>
#include <functional>

namespace vknn {
    namespace {

        // Deterministic Winograd-vs-direct threshold: Winograd is chosen for a fp16 3x3 conv when
        // Cin*Cout <= this, otherwise the direct kernel. The two kernels round fp16 differently, so the
        // choice is made from the shape (not a timing race) to keep the output bits identical across
        // runs and tuning levels; Winograd's fp16 transform-domain intermediates grow with Cin*Cout and
        // make it memory-bound above this point. Calibrated against the measured conv-suite winners.
        constexpr int64_t kWinoMaxCinCout = 32768;
        // Large-Cin*Cout shapes still take Winograd when the output map supplies enough tiles to keep
        // the GEMM's M dimension fed. Measured on the primary device with the register-tile GEMM in
        // the race (single-conv probes, Conv GPU-total, min of 3 cooled rounds): 256x256 @ 14x14
        // (196 px) is +9% SLOWER on Winograd (tile-starved: 49 F(2,3) tiles), while 256x256 @ 20x20
        // (400 px) is -38%, 192x192 @ 35x35 is -42% and 512x512 @ 28x28 is -69% FASTER. The floor
        // sits at the smallest measured winner; between 196 and 400 output pixels is unmeasured and
        // stays direct.
        constexpr int64_t kWinoLargeCMinPixels = 400;

        // Workgroup size of the group==1 direct conv shader when no measurement applies: the value
        // Tuning::None dispatches and the incumbent every local-size race is seeded with.
        // The direct kernel's Tuning::None width and race incumbent is env.convLocalSize (the
        // device's one-subgroup width, resolved at load); only the race's narrow-candidate floor
        // stays a constant.
        constexpr uint32_t kConvRaceMinWidth = 32;
        // Output pixels per thread of the 1x1 kernels when no measurement applies: the value
        // Tuning::None dispatches and the incumbent pickWTile's race is seeded with.
        constexpr uint32_t kConv1x1DefaultWTile = 4;

        // Depthwise 2x2-tile occupancy floor: the tile kernel's dispatch (N * Cb * ceil(OH/2) *
        // ceil(OW/2) threads) must keep at least this many threads or the candidate never races
        // (see pickDwTile for the measured basis).
        constexpr int64_t kDwTileMinThreads = 4096;

        // Deterministic implicit-GEMM-vs-direct threshold: the conv_gemm kernel is chosen only when the
        // GEMM tiling has enough parallelism to amortize its setup - at least kGemmMinCoutTiles output-
        // channel tiles (of kConvGemmTileN each) on the N axis and kGemmMinM output rows on the M axis.
        // Same rationale as Winograd: a deterministic shape rule replaces a timing race so the K-reduction
        // order (and thus the output bits) never changes with thermal state or tuning level.
        // The per-thread conv family's lane width is env.convLocalSize (laneWidthFor over
        // flat::kConvFamilyLaneWidth, resolved at load). The kSplitK*/kGemmMin* thresholds below
        // stay fixed shape rules on purpose: they steer summation order (byte-affecting), so no
        // measured value may move them.
        constexpr int64_t kGemmMinCoutTiles = 8;
        constexpr int64_t kGemmMinM         = 128;

        // Deterministic general-split-K rule: a non-pointwise conv whose standard dispatch is too
        // small to hide its K-loop latency (deep reduction, small output extent - a stride-2 3x3
        // into a 7x7 map, an 8x8-map Inception branch conv) runs the split-K partial+reduce pair
        // instead of the register-tiled kernels. Splitting the K sum changes the fp32 summation
        // order, so like the Winograd and implicit-GEMM choices this is a shape rule, never a
        // timing race. Calibrated against per-shape cooled A/B winners with the compensated
        // partial pass: deep-tap tiny-map convs gain 22-32%; larger maps and shallow-tap strided
        // 1x1 downsamples LOSE (the extra partial traffic and compensation adds outweigh the
        // latency win), hence the taps floor and the output-extent ceiling.
        constexpr int64_t kSplitKGenMinTaps = 320; // Cinb*KH*KW: minimum per-thread reduction depth
        // A multi-tap kernel amortizes the partial-buffer round-trip over KH*KW input reads per
        // output, so its taps floor sits lower: the 8x8-map 3x1/1x3 Inception branches (taps 288)
        // gain like the 3x3s, while the single-tap strided 1x1 downsamples at the same tap count
        // keep losing and stay excluded by the higher single-tap floor.
        constexpr int64_t kSplitKGenMinTapsMulti = 256;   // taps floor when KH*KW >= 3
        constexpr int64_t kSplitKGenMaxOHW       = 64;    // output pixels: above this the standard kernels win
        constexpr int64_t kSplitKGenMaxThreads   = 16384; // Coutb*ceil(OHW/4): below this the GPU is starved

        // Read an advanced kernel hint from the session Config (see include/vknn/config.h).
        inline int cfgHint(const VkOpEnv &env, Hint h) {
            return env.config ? env.config->hint(h) : 0;
        }

        struct ConvOp: VulkanOp {
            static constexpr int kTile   = 4; // 1x1 threshold math default; the real per-shape tile is wTile
            uint32_t             wTile   = 4; // output pixels per thread in the 1x1 kernels (autotuned spec constant)
            uint32_t             ocbTile = 1; // output channel-blocks per thread in the 1x1 kernels (autotuned spec constant)
            // wino_gemm workgroup output tile, in lockstep with shaders/wino_gemm_fp16.comp: N is
            // NDIM=8 channel-blocks; M is MDIM=8 threads x the RM spec constant (4 or 8, raced by
            // tuneWino), so the real dispatch and the timing dispatch derive the tile from one
            // helper and cannot diverge. The subgroup variant (wino_gemm_sg) keeps its fixed
            // 32-tile geometry.
            static constexpr int kWinoGemmTileNB = 8;  // output channel-blocks (N) per workgroup
            static constexpr int kWinoSgTileM    = 32; // wino_gemm_sg's fixed M tile
            static int           winoGemmTileM(int rm) {
                return 8 * rm; // MDIM * RM
            }
            int                                  winoRm         = 4;     // wino_gemm tiles per thread (spec constant 0)
            int                                  winoAcc16      = 0;     // wino_gemm fp16 accumulation (spec constant 1; race-selected only)
            bool                                 winoRegGemm    = false; // the no-LDS register-tile GEMM body (wino_gemm_reg) won the race
            bool                                 depthwise      = false;
            bool                                 pointwise      = false;
            bool                                 winograd       = false;
            bool                                 splitk         = false;
            bool                                 reg            = false; // register-tiled implicit-im2col general conv (WTILE pixels/thread)
            bool                                 lds            = false; // LDS input-halo 3x3 (8x8 tile/workgroup)
            bool                                 gemm           = false; // implicit-GEMM kernel (conv_gemm.comp) won the plan-time race
            int64_t                              ldsGroups      = 0;
            bool                                 pwS2           = false; // strided 1x1 (downsample) on the register-tiled kernel
            bool                                 hasRes         = false; // residual Add fused into the epilogue (out = act(conv + residual))
            int                                  ocSplitParts   = 1;     // conv_reg OC-split: record() replays this many flat-gid slice dispatches
            int64_t                              ocSliceThreads = 0;     // threads per OC-split slice (64-aligned; see ocSplitSliceThreads)
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          wbuf, bbuf;
            ConvPC                               pc {};
            DwPC                                 dpc {};
            int64_t                              total     = 0;
            uint32_t                             localSize = 64;

            // Attached pointwise-chain epilogue (fusePointwiseChains); applies at whichever
            // variant's final store runs (direct/dw/1x1/lds/reg, split-K reduce, Winograd output).
            PwEpi epi;

            // --- implicit-GEMM path (conv_gemm.comp) for the general dense KxK branch ---
            std::shared_ptr<vk::Buffer> gwbuf; // weights repacked [K][Cout], k = (ky*KW+kx)*Cin+ic
            ConvGemmPC                  gpc {};
            uint32_t                    ggx = 0, ggy = 0, ggz = 0;

            // --- split-K 1x1 (deep, low-parallelism convs): partial pass + reduce pass ---
            std::shared_ptr<vk::ComputePipeline> skPipe, skRed;
            std::shared_ptr<vk::Buffer>          partBuf;
            SplitKPC                             skPC {};
            SplitKGenPC                          skGenPC {};
            ReducePC                             skRedPC {};
            int64_t                              skGroups = 0, skRedGroups = 0;
            bool                                 splitkGen = false; // general KxK/strided split-K (conv_splitk.comp)

            // Shared split-K geometry: KPARTS targets kPwSplitKTargetThreads partial-pass threads,
            // capped by Cinb (pw_splitk_rule.h, shared with the fused depthwise+project op).
            static int64_t splitKParts(int64_t Cinb, int64_t Coutb, int64_t OHW) {
                return pwSplitKParts(Cinb, Coutb, OHW);
            }

            void prepareSplitKShared(const Node &node, VkOpEnv &env, int64_t Cout, int64_t Coutb, int64_t OHW, int64_t kparts) {
                skRedPC     = {(int) Cout, (int) OHW, (int) kparts, (int) node.fusedAct, node.actLo, node.actHi};
                partBuf     = std::make_shared<vk::Buffer>(*env.ctx, (size_t) kparts * Coutb * OHW * 4 * 4, vk::MemPref::kDeviceOnly); // fp32 partials (vec4)
                skGroups    = groups(kparts * Coutb * OHW, 64);
                skRedGroups = groups(Coutb * OHW, 64);
                skRed = env.pipeline((std::string("conv1x1_reduce") + epi.suffix() + "_fp16").c_str(), epi.active ? 4 + epi.extraBufs() : (hasRes ? 4u : 3u), sizeof(ReducePC), std::vector<uint32_t> {(uint32_t) (hasRes ? 1 : 0)});
            }

            void prepareSplitK(const Node &node, VkOpEnv &env, NCHW x, NCHW y, int64_t Cout, int64_t Coutb) {
                int64_t Cinb = cBlocks(x.c), HW = y.h * y.w;
                int64_t kparts = splitKParts(Cinb, Coutb, HW);
                int64_t chunk  = (Cinb + kparts - 1) / kparts;
                skPC           = {(int) x.c, (int) Cout, (int) HW, (int) kparts, (int) chunk};
                prepareSplitKShared(node, env, Cout, Coutb, HW, kparts);
                skPipe = env.pipeline("conv1x1_splitk_fp16", 3, sizeof(SplitKPC), std::vector<uint32_t> {});
            }

            void prepareSplitKGeneral(const Node &node, VkOpEnv &env, NCHW x, NCHW y, int64_t Cout, int64_t Coutb, int64_t KH, int64_t KW, const std::vector<int64_t> &st, const std::vector<int64_t> &pad, const std::vector<int64_t> &dil) {
                int64_t Cinb = cBlocks(x.c), OHW = y.h * y.w;
                int64_t kparts = splitKParts(Cinb, Coutb, OHW);
                int64_t chunk  = (Cinb + kparts - 1) / kparts;
                skGenPC        = {(int) x.c,   (int) x.h,   (int) x.w,    (int) Cout,   (int) y.h,    (int) y.w,    (int) KH,     (int) KW,
                                  (int) st[0], (int) st[1], (int) pad[0], (int) pad[1], (int) dil[0], (int) dil[1], (int) kparts, (int) chunk};
                prepareSplitKShared(node, env, Cout, Coutb, OHW, kparts);
                skPipe = env.pipeline("conv_splitk_fp16", 3, sizeof(SplitKGenPC), std::vector<uint32_t> {});
            }

            // --- Winograd F(2x2,3x3) state (3x3, stride 1, pad 1, group 1, fp16) ---
            // The default Winograd kernel is the 3-pass tiled GEMM (wino_input -> wino_gemm -> wino_out);
            // tuneWino picks it per shape against the direct 3x3. The non-GEMM matmul variants all
            // regress on this GPU and are gated behind Hint::WinogradVariant as documented negative
            // results: a 2-pass naive-matmul (memory-bound on the global V round-trip), the same split
            // 4 ways (wino_fused2, bandwidth- not occupancy-bound), a fully-fused kernel keeping V in LDS
            // (wino_full, the static LDS array collapses occupancy), and a subgroup-shuffle GEMM
            // (wino_gemm_sg, shuffle costs more than LDS here).
            std::shared_ptr<vk::ComputePipeline> wInPipe, wFusedPipe, wFullPipe;
            std::shared_ptr<vk::Buffer>          ubuf, vbuf;
            WinoInPC                             wInPC {};
            WinoFusedPC                          wFusedPC {};
            int64_t                              wInGroups = 0, wFusedGroups = 0, wFullGroups = 0;
            bool                                 wino2    = false; // occupancy-friendly 2-pass matmul (4 threads/tile, M split across quads)
            bool                                 winofull = false; // single fully-fused kernel: V stays in LDS, no global round-trip
            // 3-pass with a TILED batched GEMM for the transform-domain multiply (MNN's structure).
            bool winogemm     = false;
            bool gemmSubgroup = false; // subgroup-shuffle GEMM (no LDS); U is [pos][icb][oc]
            // Output-tile edge: 2 = F(2,3) (16 pts), 4 = F(4,3) (36 pts, 0.56x V/M traffic),
            // 6 = F(6,3) (64 pts, 2.25x fewer tiles; explicit WinogradUnit hint only — see tuneWino).
            int                                  winoUnit = 2;
            std::shared_ptr<vk::ComputePipeline> wGemmPipe, wOutPipe;
            std::shared_ptr<vk::Buffer>          mbuf;
            WinoGemmPC                           wGemmPC {};
            int64_t                              wGemmGX = 0, wGemmGY = 0, wGemmGZ = 0, wOutGroups = 0;

            void prepareWinograd(const Node &node, VkOpEnv &env, NCHW x, NCHW y, int64_t Cout, int64_t Coutb) {
                const Graph       &g   = *env.graph;
                int64_t            Cin = x.c, Cinb = cBlocks(x.c);
                const int          U_ = winoUnit, A_ = U_ + 2; // output edge, transform-domain edge (4, 6 or 8)
                const int          nPos = A_ * A_;             // 16 (F2,3), 36 (F4,3) or 64 (F6,3)
                int64_t            nTH = (y.h + U_ - 1) / U_, nTW = (y.w + U_ - 1) / U_, nT = x.n * nTH * nTW;
                std::vector<float> wsrcv = initFloats(g, node.inputs[1]);
                const float       *wsrc  = wsrcv.data();

                int wvar     = cfgHint(env, Hint::WinogradVariant); // 0=tiled-GEMM 1=fused 2=split 3=full 4=subgroup-GEMM
                gemmSubgroup = (wvar == 4);

                // Filter transform matrix G (A_ x 3): F(2,3) and F(4,3) inline; F(6,3) from
                // core/wino_f63.h (kWinoF63G, derived + scored by tools/wino_f63_points.cpp).
                static const float G2[4][3] = {{1, 0, 0}, {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0, 0, 1}};
                static const float G4[6][3] = {
                    {0.25f, 0, 0}, {-1.f / 6, -1.f / 6, -1.f / 6}, {-1.f / 6, 1.f / 6, -1.f / 6}, {1.f / 24, 1.f / 12, 1.f / 6}, {1.f / 24, -1.f / 12, 1.f / 6},
                    {0, 0, 1}};
                // Host weight transform U = G g G^T, vec4(4 ic). Default packed [pos][oc][icb]; the
                // subgroup GEMM needs [pos][icb][oc] (coalesced per-K output-channel loads). Cached on disk.
                ubuf = uploadCached(env, node.name + (gemmSubgroup ? "#winosg" : "#wino") + std::to_string(U_), [&] {
                    std::vector<float> U((size_t) nPos * Cout * Cinb * 4, 0.f);
                    for (int64_t oc = 0; oc < Cout; ++oc)
                    {
                        for (int64_t ic = 0; ic < Cin; ++ic)
                        {
                            const float *gk = wsrc + (oc * Cin + ic) * 9; // 3x3
                            float        Gg[8][3];                        // A_ <= 8 rows across the F-unit family
                            for (int i = 0; i < A_; ++i)
                            {
                                for (int j = 0; j < 3; ++j)
                                {
                                    const float *Gi = (U_ == 2) ? G2[i] : (U_ == 4) ? G4[i] : kWinoF63G[i];
                                    Gg[i][j]        = Gi[0] * gk[j] + Gi[1] * gk[3 + j] + Gi[2] * gk[6 + j];
                                }
                            }
                            int64_t icb = ic / 4, lane = ic % 4;
                            for (int i = 0; i < A_; ++i)
                            {
                                for (int j = 0; j < A_; ++j)
                                {
                                    const float *Gj    = (U_ == 2) ? G2[j] : (U_ == 4) ? G4[j] : kWinoF63G[j];
                                    float        u     = Gg[i][0] * Gj[0] + Gg[i][1] * Gj[1] + Gg[i][2] * Gj[2];
                                    int          pos   = i * A_ + j;
                                    int64_t      uidx  = gemmSubgroup ? ((pos * Cinb + icb) * Cout + oc) : ((pos * Cout + oc) * Cinb + icb);
                                    U[uidx * 4 + lane] = u;
                                }
                            }
                        }
                    }
                    return U;
                });

                wFusedPC = {(int) x.n, (int) Cin, (int) Cout, (int) y.h, (int) y.w, (int) nTH, (int) nTW, (int) node.fusedAct, node.actLo, node.actHi};
                winofull = (wvar == 3);
                if (winofull)
                {
                    // Single fused kernel: one workgroup per (tile, ocb-group of 16). No V buffer, no input pass.
                    int64_t ocbGroups = (Coutb + 15) / 16;
                    wFullGroups       = nT * ocbGroups;
                    wFullPipe = env.pipeline((std::string("wino_full") + epi.suffix() + "_fp16").c_str(), 4 + epi.extraBufs(), sizeof(WinoFusedPC), std::vector<uint32_t> {});
                    return;
                }
                int el = 2; // fp16
                // Transformed-input V: one vec4 (4 packed channels, the NC4HW4 lane count) per
                // (transform position, input channel-block, tile), stored fp16. Sizing = nPos * Cinb * nT
                // vec4s * el bytes/element.
                vbuf  = std::make_shared<vk::Buffer>(*env.ctx, (size_t) nPos * Cinb * nT * 4 * el, vk::MemPref::kDeviceOnly);
                wInPC = {(int) x.n, (int) x.c, (int) x.h, (int) x.w, (int) y.h, (int) y.w, (int) nTH, (int) nTW};
                // wino_input / wino_input4 run one thread per (icb, tile); wino_input6's separable
                // two-stage transform runs kWinoF63TransformLanes cooperating threads per unit.
                wInGroups = groups(Cinb * nT * (U_ == 6 ? kWinoF63TransformLanes : 1), 64);

                // The tiled-GEMM 3-pass is the default Winograd kernel (variant 0). Variant 4 is the same
                // 3-pass with the subgroup-shuffle GEMM (no LDS). The fused / fused-split variants are
                // Hint-gated regressions (see above).
                winogemm = (wvar == 0 || wvar == 4);
                if (winogemm)
                {
                    // 3-pass: input transform -> V, TILED batched GEMM -> M, output transform -> dst.
                    // M: one vec4 (4 packed output channels) per (transform position, tile, output
                    // channel-block), fp16; sizing = nPos * nT * Coutb vec4s * el bytes/element.
                    mbuf    = std::make_shared<vk::Buffer>(*env.ctx, (size_t) nPos * nT * Coutb * 4 * el, vk::MemPref::kDeviceOnly);
                    wGemmPC = {(int) Cin, (int) Cout, (int) nT};
                    wGemmGX = groups(nT, gemmSubgroup ? kWinoSgTileM : winoGemmTileM(winoRm)); // workgroups over M (tiles)
                    wGemmGY = groups(Coutb, kWinoGemmTileNB);                                  // workgroups over N (ocb)
                    wGemmGZ = nPos;                                                            // one GEMM per transform position (16, 36 or 64)
                    wInPipe = env.pipeline(U_ == 2 ? "wino_input_fp16" : (U_ == 4 ? "wino_input4_fp16" : "wino_input6_fp16"), 2, sizeof(WinoInPC), std::vector<uint32_t> {});
                    // GEMM body: the LDS-staged kernel by default, the no-LDS register-tile twin
                    // when tuneWino's bit-neutral race picked it (bit 4), or the Hint-gated
                    // subgroup variant. The register kernel takes RM alone (no ACC16 body).
                    wGemmPipe = env.pipeline(gemmSubgroup ? "wino_gemm_sg_fp16" : (winoRegGemm ? "wino_gemm_reg_fp16" : "wino_gemm_fp16"), 3, sizeof(WinoGemmPC), gemmSubgroup ? std::vector<uint32_t> {} : (winoRegGemm ? std::vector<uint32_t> {(uint32_t) winoRm} : std::vector<uint32_t> {(uint32_t) winoRm, (uint32_t) winoAcc16}));
                    // wino_out / wino_out4 run one thread per (ocb, tile); wino_out6 runs
                    // kWinoF63TransformLanes cooperating threads per unit (record() dispatches
                    // wOutGroups). Each arm spells "<stem>" + epi.suffix() so the
                    // tools/check_epi_sync.py stem derivation sees every hosting kernel.
                    std::string outName = (U_ == 2) ? std::string("wino_out") + epi.suffix() :
                                          (U_ == 4) ? std::string("wino_out4") + epi.suffix() :
                                                      std::string("wino_out6") + epi.suffix();
                    wOutGroups          = groups(Coutb * nT * (U_ == 6 ? kWinoF63TransformLanes : 1), 64);
                    wOutPipe            = env.pipeline((outName + "_fp16").c_str(), 3 + epi.extraBufs(), sizeof(WinoFusedPC), std::vector<uint32_t> {});
                    return;
                }
                wino2 = (wvar == 2);
                // wino2: 4 threads cooperate per (ocb,tile) -> 4x the threads, dispatched 64/workgroup.
                wFusedGroups = wino2 ? groups(Coutb * nT * 4, 64) : groups(Coutb * nT, 64);
                wInPipe      = env.pipeline("wino_input_fp16", 2, sizeof(WinoInPC), std::vector<uint32_t> {});
                wFusedPipe = env.pipeline((std::string(wino2 ? "wino_fused2" : "wino_fused") + epi.suffix() + "_fp16").c_str(), 4 + epi.extraBufs(), sizeof(WinoFusedPC), std::vector<uint32_t> {});
            }

            // Try a few workgroup sizes for this exact shape, keep the fastest, and cache it so later
            // runs skip the measurement. Only the group==1 conv shader is tunable. Timing dispatches must
            // run on dedicated scratch buffers, never the real activation buffers, or they race and
            // corrupt the data path.
            //
            // Every race in this op times the epilogue-hosting kernel variant and keys its cache
            // signature by epi.suffix(). A fused pointwise chain raises the kernel's register demand,
            // and that is part of what decides which tile is fastest: measured on the efficientnet
            // stem, the register-blocked conv_reg tile beats the direct kernel by 7% raced without
            // the epilogue and loses to it by 87% raced with it (in-graph: +62%). Two nodes of the
            // same shape that differ in whether they host a chain therefore need separate
            // measurements, not a shared cache entry.
            uint32_t pickLocalSize(VkOpEnv &env) {
                char buf[96];
                snprintf(buf, sizeof(buf), "convls%s_%d_%d_%d_%d_%d_%d_%d_%d", epi.suffix(), pc.Cin, pc.H, pc.W, pc.Cout, pc.OH, pc.OW, pc.KH, pc.SH);
                std::string sig      = env.gpuTag + "/" + buf;
                const int   reqLevel = (int) env.tuning;
                // Consult the cache first. A cached pick is reused under --tuning none (none runs no new
                // sweep but honors a stored measurement) and whenever its measured level is at least the
                // requested one; a lower-level entry (fast cached, heavy requested) is re-swept so the
                // heavier candidate set is explored.
                if (env.weights)
                {
                    int cachedLevel = -1;
                    int cached      = env.weights->tuned(sig, 0, &cachedLevel);
                    if (cached > 0 && (reqLevel == (int) Tuning::None || cachedLevel >= reqLevel))
                    {
                        return (uint32_t) cached;
                    }
                }
                if (env.tuning == Tuning::None)
                {
                    return env.convLocalSize; // no cached pick and no new sweep -> the device's one-subgroup default
                }
                uint32_t best = env.convLocalSize;
                if (env.runner)
                {
                    int    es       = env.useFp16 ? 2 : 4;
                    size_t srcBytes = (size_t) pc.N * cBlocks(pc.Cin) * pc.H * pc.W * 4 * es;
                    size_t dstBytes = (size_t) pc.N * cBlocks(pc.Cout) * pc.OH * pc.OW * 4 * es;
                    auto   sSrc     = std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>(srcBytes, 16), vk::MemPref::kDeviceOnly);
                    auto   sDst     = std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>(dstBytes, 16), vk::MemPref::kDeviceOnly);
                    // The device default (one whole subgroup, env.convLocalSize) leads the list, so
                    // it is the incumbent; the challengers are its 2x and 4x subgroup multiples and
                    // (under Heavy) the half-subgroup narrower packing - the same ladder the old
                    // hardcoded {64, 128, 256} + Heavy 32 spelled on a 64-wide device, but spanning
                    // whatever width this device actually runs. Everything clamps to the exact caps.
                    const auto &cap = env.ctx->caps();
                    const uint32_t maxInv = std::min(cap.maxWorkGroupInvocations != 0u ? cap.maxWorkGroupInvocations : env.convLocalSize, cap.maxWorkGroupSize[0] != 0u ? cap.maxWorkGroupSize[0] : env.convLocalSize);
                    std::vector<uint32_t> cands {env.convLocalSize};
                    for (uint32_t mult: {2u, 4u})
                    {
                        if (env.convLocalSize * mult <= maxInv)
                        {
                            cands.push_back(env.convLocalSize * mult);
                        }
                    }
                    if (env.tuning == Tuning::Heavy && env.convLocalSize / 2u >= kConvRaceMinWidth)
                    {
                        cands.insert(cands.begin() + 1, env.convLocalSize / 2u);
                    }
                    // The workgroup size changes neither the thread count nor the traffic, only how
                    // the threads are packed, so every candidate carries the same cost inputs and
                    // the analytical prefilter leaves the list alone. It stays on the shared path so
                    // one selection rule covers every race.
                    const int64_t               Cinb = cBlocks(pc.Cin);
                    std::vector<vk::KernelCost> costs;
                    costs.reserve(cands.size());
                    for (uint32_t ls: cands)
                    {
                        vk::KernelCost cost;
                        cost.streamVec4            = (double) total * (double) Cinb * (double) pc.KH * (double) pc.KW + (double) total;
                        cost.residentVec4          = (double) total * 4.0 * (double) Cinb * (double) pc.KH * (double) pc.KW;
                        cost.streamFootprintVec4   = (double) (pc.N * Cinb * pc.H * pc.W) + (double) total;
                        cost.residentFootprintVec4 = (double) (pc.Cout * Cinb * pc.KH * pc.KW);
                        cost.waves                 = (double) groups(total, ls) * (double) ls / 64.0;
                        costs.push_back(cost);
                    }
                    std::vector<VkBuffer> bufs = {sSrc->handle(), wbuf->handle(), bbuf->handle(), sDst->handle()};
                    epi.appendForTiming(bufs, sDst->handle());
                    vk::TuneTimer       timer(env);
                    std::vector<double> ms     = vk::racePruned(costs, vk::deviceTuneModel(env), [&](int index) {
                        auto pipe = env.pipeline(shader((std::string("conv") + epi.suffix()).c_str(), env.useFp16), 4 + epi.extraBufs(), sizeof(ConvPC), {cands[(size_t) index]});
                        return timer.time([&](VkCommandBuffer cmd) {
                            pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(total, cands[(size_t) index]));
                        });
                    });
                    double              bestMs = ms[0];
                    for (size_t ci = 1; ci < cands.size(); ++ci)
                    {
                        if (ms[ci] < bestMs)
                        {
                            bestMs = ms[ci];
                            best   = cands[ci];
                        }
                    }
                    VKNN_DEBUG << "autotune " << sig << " -> local_size_x=" << best << vk::raceTimes(ms);
                }
                if (env.weights)
                {
                    env.weights->setTuned(sig, (int) best, reqLevel);
                }
                return best;
            }

            // The depthwise output tile is NOT raced: the 2x2 twin (dwconv_t2) is a measured
            // negative result. Racing it representatively — each candidate dispatched once from a
            // cold cache with the epilogue the graph actually runs — selects it on 1 of 32 depthwise
            // shapes across the suite, and an in-graph study measured it 195-265% SLOWER than the
            // 1-pixel kernel on the shapes the old warm-repeated race did pick it for: quartering
            // the thread count trades parallelism, which this device punishes. ShuffleNetV2 never
            // reaches this decision at all (its depthwise convs are consumed by FusedDwPw), so the
            // v1.4.1 ShuffleNet win came from the ChannelShuffle fold, not this tile. Keeping the
            // candidate also cost the most cold-tuning time of any entrant, since its losing
            // pipeline variants still compile. The kernel stays selectable through the tune table
            // for a deliberate experiment; nothing selects it automatically.
            int pickDwTile(VkOpEnv &env, NCHW x, NCHW y, int64_t Cb) {
                // Eligibility precedes the cache consult: a cached tile pick is only honored while
                // the shape still tiles (the sig carries OH/OW, so this re-gate is belt-and-braces).
                const int64_t tileThreads  = (int64_t) x.n * Cb * ((y.h + 1) / 2) * ((y.w + 1) / 2);
                bool          tileEligible = y.h >= 2 && y.w >= 2 && tileThreads >= kDwTileMinThreads;
                char          buf[112];
                snprintf(buf, sizeof(buf), "dwt%s_%d_%d_%d_%d_%d_%d", epi.suffix(), dpc.C, dpc.OH, dpc.OW, dpc.KH, dpc.KW, dpc.SH);
                std::string sig = env.gpuTag + "/" + buf;
                int         reuse;
                if (env.reuseTuned(sig, reuse) && (reuse == 0 || (reuse == 1 && tileEligible)))
                {
                    return reuse;
                }
                // The tile is never selected automatically (see above); only an explicit tune-table
                // entry, consulted before this point, can still reach it.
                return 0;
            }

            // Autotune the register tile of the 1x1 kernels: WTILE (output pixels per thread) and OCB
            // (output channel-blocks per thread), encoded as WTILE | OCB<<8. Per-output arithmetic is
            // identical for every tile — only the thread<->output mapping changes — so the choice never
            // affects output bits, unlike the direct-vs-Winograd choice. Measured on scratch buffers and
            // cached like the local-size tune. A plain cached value (2/4/8, from before OCB existed)
            // decodes as OCB=1.
            static bool valid1x1Tile(int v) {
                int wt = v & 0xff, ocb = v >> 8;
                return (wt == 2 || wt == 4 || wt == 8) && (ocb == 0 || ocb == 1 || ocb == 2);
            }
            uint32_t pickWTile(VkOpEnv &env, bool s2, NCHW x, NCHW y, int64_t Cout, int64_t Coutb) {
                char buf[96];
                snprintf(buf, sizeof(buf), "c1x1%s%s_%d_%d_%d_%d_%d", s2 ? "s2" : "", epi.suffix(), (int) x.c, (int) Cout, (int) y.h, (int) y.w, hasRes ? 1 : 0);
                std::string sig = env.gpuTag + "/" + buf;
                int         reuse;
                if (env.reuseTuned(sig, reuse) && reuse > 0 && valid1x1Tile(reuse))
                {
                    return (uint32_t) reuse;
                }
                if (env.tuning == Tuning::None || !env.runner)
                {
                    return kConv1x1DefaultWTile;
                }
                int    es       = env.useFp16 ? 2 : 4;
                size_t srcBytes = (size_t) x.n * cBlocks(x.c) * x.h * x.w * 4 * es;
                size_t dstBytes = (size_t) y.n * Coutb * y.h * y.w * 4 * es;
                auto   sSrc     = std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>(srcBytes, 16), vk::MemPref::kDeviceOnly);
                auto   sDst     = std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>(dstBytes, 16), vk::MemPref::kDeviceOnly);
                // The default tile leads the list, so it is the incumbent every other candidate has
                // to beat; a race that resolves nothing therefore keeps what Tuning::None dispatches.
                const std::vector<uint32_t> cands = {kConv1x1DefaultWTile, 2u, 8u, 4u | (2u << 8), 8u | (2u << 8)};
                // Cost inputs for the analytical prefilter (vk_tune_model.h), from geometry alone.
                // Per thread the 1x1 kernels issue WTILE input vec4 and OCB*4 weight vec4 per input
                // channel-block and store WTILE*OCB; a residual or an epilogue reads the output back.
                const int64_t               Cinb         = cBlocks(x.c);
                const int64_t               HW           = y.h * y.w;
                const double                storesPerOut = (hasRes || epi.active) ? 2.0 : 1.0;
                std::vector<int64_t>        totals;
                std::vector<vk::KernelCost> costs;
                totals.reserve(cands.size());
                costs.reserve(cands.size());
                for (uint32_t cand: cands)
                {
                    uint32_t      wt = cand & 0xffu, ocb = std::max(1u, cand >> 8);
                    const int64_t threads = x.n * ((Coutb + ocb - 1) / ocb) * ((HW + wt - 1) / wt);
                    totals.push_back(threads);
                    vk::KernelCost cost;
                    // The activation window and the output are the activation side; the weight
                    // block is the weight side. What each side's re-reads cost is set by how much
                    // that side spans, which the model derives from the footprints below.
                    cost.streamVec4            = (double) threads * ((double) wt * (double) Cinb + (double) wt * (double) ocb * storesPerOut);
                    cost.residentVec4          = (double) threads * 4.0 * (double) ocb * (double) Cinb;
                    cost.streamFootprintVec4   = (double) (x.n * Cinb * x.h * x.w) + (double) (x.n * Coutb * HW);
                    cost.residentFootprintVec4 = (double) (Cout * Cinb);
                    cost.waves                 = (double) groups(threads, 64);
                    costs.push_back(cost);
                }
                std::vector<VkBuffer> bufs = {sSrc->handle(), wbuf->handle(), bbuf->handle(), sDst->handle()};
                if (hasRes || epi.active)
                {
                    bufs.push_back(sDst->handle()); // timing only: any readable buffer serves as the residual
                }
                epi.appendForTiming(bufs, sDst->handle());
                vk::TuneTimer timer(env);
                // Pipelines are built inside the timed lambda so a pruned candidate never compiles
                // its variant - the compilation of losing variants, not the timing, is what the
                // prefilter is here to remove. env.pipeline() memoises, so the repeat rounds hit
                // the cache.
                std::vector<double> ms = vk::racePruned(costs, vk::deviceTuneModel(env), [&](int index) {
                    uint32_t wt = cands[(size_t) index] & 0xffu, ocb = std::max(1u, cands[(size_t) index] >> 8);
                    auto pipe = env.pipeline(shader((std::string(s2 ? "conv1x1_s2" : "conv1x1") + epi.suffix()).c_str(), env.useFp16), epi.active ? 5 + epi.extraBufs() : (hasRes ? 5u : 4u), sizeof(ConvPC), std::vector<uint32_t> {(uint32_t) (hasRes ? 1 : 0), wt, ocb, env.convLocalSize});
                    return timer.time([&](VkCommandBuffer cmd) {
                        pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(totals[(size_t) index], env.convLocalSize));
                    });
                });
                // cands[0] is the deterministic default and stays the incumbent: a challenger must
                // beat ITS time by the margin, and the fastest qualifier wins. Comparing against a
                // running best instead would let the margin compound and make the outcome depend on
                // the order the candidates happen to sit in the list. The margin applies to every
                // challenger, not just the OCB>1 class — race noise on these shapes is wider than
                // the differences being resolved, so an unmargined challenger displaces the proven
                // default on noise alone. It is waived only for a challenger the analytical model
                // also ranks cheaper (see kTuneRaceMargin), which is a second signal independent of
                // the measurement and not the noisy sample the margin exists to discard.
                const std::vector<double> model  = vk::modelEstimates(costs, vk::deviceTuneModel(env));
                uint32_t                  best   = cands[0];
                double                    bestMs = ms[0];
                for (size_t ci = 1; ci < cands.size(); ++ci)
                {
                    const double need = ms[0] * (model[ci] < model[0] ? 1.0 : vk::kTuneRaceMargin);
                    if (ms[ci] < need && ms[ci] < bestMs)
                    {
                        bestMs = ms[ci];
                        best   = cands[ci];
                    }
                }
                VKNN_DEBUG << "autotune " << sig << " -> wtile=" << (best & 0xffu) << " ocb=" << std::max(1u, best >> 8) << vk::raceTimes(ms);
                if (env.weights)
                {
                    env.weights->setTuned(sig, (int) best, (int) env.tuning);
                }
                return best;
            }

            // Autotune the general direct conv's bit-exact kernel set: race the 1-pixel/thread direct
            // kernel against conv_reg computing OCB channel-blocks per thread (each input vec4 reused
            // across 4*OCB output-channel dots), the conv_reg tiles replayed as 2/4 OC-split slice
            // dispatches (kChoiceOcSplit2/4), and — when `lds3x3` marks the shape eligible (3x3, s1,
            // p1, d1, >=14x14 output, fp16) — the LDS input-halo kernel. Returns 0 = direct, the OCB
            // block factor (optionally with a split flag), or kChoiceLds3x3. Per-output arithmetic is
            // identical for every candidate (only the thread<->output mapping and the input staging
            // change), so the choice never affects output bits (no anti-noise margin needed, unlike
            // Winograd) and a cache-off re-tune stays deterministic. Measured on scratch buffers +
            // cached like pickWTile.
            static constexpr int kChoiceLds3x3   = 9;       // pickOcb result: the LDS input-halo 3x3 kernel (8x8 tile) won
            static constexpr int kChoiceLds16    = 10;      // pickOcb result: the LDS input-halo 3x3 kernel (16x16 tile) won
            static constexpr int kChoice1D       = 1 << 16; // pickOcb result flag: the sliding-window 1-D kernel won
            static constexpr int kChoiceOcSplit2 = 1 << 17; // pickOcb result flag: conv_reg dispatched as 2 flat-gid slices
            static constexpr int kChoiceOcSplit4 = 1 << 18; // pickOcb result flag: conv_reg dispatched as 4 flat-gid slices
            // pickOcb results encode conv_reg's tile as OCB | WTILE<<8 (WTILE 0 = the classic 4),
            // with kChoice1D marking the conv_1d kernel at the same OCB/WTILE encoding and
            // kChoiceOcSplit2/4 marking a conv_reg tile whose flat gid range record() splits into
            // 2/4 back-to-back dispatches (MNN-style output-channel chunking; slices are 64-aligned
            // and disjoint, so no barrier separates them). The split flags never combine with each
            // other, the 1-D kernel, or the LDS choices. A plain cached value from before the
            // WTILE axis existed decodes unchanged.
            static bool validOcbChoice(int v) {
                if (v == kChoiceLds3x3 || v == kChoiceLds16)
                {
                    return true;
                }
                int split = v & (kChoiceOcSplit2 | kChoiceOcSplit4);
                if (split == (kChoiceOcSplit2 | kChoiceOcSplit4))
                {
                    return false;
                }
                if (split != 0 && ((v & kChoice1D) != 0 || (v & 0xff) == 0))
                {
                    return false; // a split flag rides only on a conv_reg OCB|WTILE encoding
                }
                int ocb = v & 0xff, wt = (v >> 8) & 0xff;
                return ocb >= 0 && ocb <= 3 && (wt == 0 || wt == 4 || wt == 8);
            }
            // Threads per OC-split slice: the flat gid range divided by the slice count, rounded up
            // to whole 64-thread workgroups so every non-final slice dispatches exactly its range
            // (64-aligned slices are disjoint - no padding thread of one slice reaches the next
            // one's outputs; the final slice's padding is cut by the kernel's total bound).
            static int64_t ocSplitSliceThreads(int64_t total, int parts) {
                return (((total + parts - 1) / parts + 63) / 64) * 64;
            }
            int pickOcb(VkOpEnv &env, NCHW x, NCHW y, int64_t Cout, int64_t Coutb, bool lds3x3) {
                // The LDS-halo kernel decodes a flat gl_WorkGroupID.x with no split spill, so its
                // group count must fit the device X limit outright. (Computed before the cache consult
                // because the reuse gate below needs it.)
                int64_t ldsG    = x.n * Coutb * ((y.h + 7) / 8) * ((y.w + 7) / 8);
                int64_t ldsG16  = x.n * Coutb * ((y.h + 15) / 16) * ((y.w + 15) / 16);
                lds3x3          = lds3x3 && ldsG <= (int64_t) env.ctx->caps().maxWorkGroupCount[0];
                bool    lds16Ok = lds3x3 && ldsG16 <= (int64_t) env.ctx->caps().maxWorkGroupCount[0];
                int64_t HW      = y.h * y.w;
                char    buf[128];
                // Stem convocb2_: the encoding gained the kChoiceOcSplit2/4 flags, so the stem is
                // renamed and a pre-split convocb_ entry can never decode against the wider field.
                snprintf(buf, sizeof(buf), "convocb2%s_%d_%d_%d_%d_%d_%d_%d_%d_%d", epi.suffix(), pc.Cin, pc.H, pc.W, pc.Cout, pc.OH, pc.OW, pc.KH, pc.KW, pc.SH);
                std::string sig = env.gpuTag + "/" + buf;
                int         reuse;
                // A 1-D kernel (1xK / Kx1, stride 1, dilation 1) is eligible for the sliding-window
                // candidates; computed before the cache consult because the reuse gate needs it.
                bool asym1dOk = env.useFp16 && ((pc.KH == 1) != (pc.KW == 1)) && pc.SH == 1 && pc.SW == 1 && pc.DH == 1 && pc.DW == 1;
                // An OC-split choice is eligible only while its conv_reg tile still yields enough
                // output-channel-block groups (>= 8 for 2 slices, >= 16 for 4) and every slice's
                // group count fits the device X limit on its own: the flat-gid slices rely on the
                // dispatch NOT being 2-D-spilled, or a padded slice would run into its neighbor's
                // range. Computed before the cache consult because the reuse gate needs it.
                auto ocSplitEligible = [&](int choice) {
                    int splitBits = choice & (kChoiceOcSplit2 | kChoiceOcSplit4);
                    if (splitBits == 0)
                    {
                        return true;
                    }
                    int     parts     = (splitBits == kChoiceOcSplit2) ? 2 : 4;
                    int     ocbBlk    = std::max(1, choice & 0xff);
                    int     wtile     = std::max(4, (choice >> 8) & 0xff);
                    int64_t ocbGroups = (Coutb + ocbBlk - 1) / ocbBlk;
                    if (ocbGroups < (parts == 2 ? 8 : 16))
                    {
                        return false;
                    }
                    int64_t tot = x.n * ocbGroups * ((HW + wtile - 1) / wtile);
                    return ocSplitSliceThreads(tot, parts) / 64 <= (int64_t) env.ctx->caps().maxWorkGroupCount[0];
                };
                // The sig omits pads/dilations (the direct and OCB kernels are pad/dilation-agnostic),
                // but choice kChoiceLds3x3 is only valid for the gated 3x3/s1/p1/d1 shape, a
                // kChoice1D value only for the 1-D-eligible shape, and an OC-split value only while
                // its slice geometry stays eligible — an ineligible node with the same sig fields
                // must re-race without it.
                if (env.reuseTuned(sig, reuse) && validOcbChoice(reuse) && (reuse != kChoiceLds3x3 || lds3x3) && (reuse != kChoiceLds16 || lds16Ok) && (!(reuse & kChoice1D) || asym1dOk) && ocSplitEligible(reuse))
                {
                    return reuse;
                }
                if (env.tuning == Tuning::None || !env.runner)
                {
                    return 0;
                }
                int                   es       = env.useFp16 ? 2 : 4;
                size_t                srcBytes = (size_t) pc.N * cBlocks(pc.Cin) * pc.H * pc.W * 4 * es;
                size_t                dstBytes = (size_t) pc.N * Coutb * pc.OH * pc.OW * 4 * es;
                auto                  sSrc     = std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>(srcBytes, 16), vk::MemPref::kDeviceOnly);
                auto                  sDst     = std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>(dstBytes, 16), vk::MemPref::kDeviceOnly);
                std::vector<VkBuffer> bufs     = {sSrc->handle(), wbuf->handle(), bbuf->handle(), sDst->handle()};
                epi.appendForTiming(bufs, sDst->handle());
                vk::TuneTimer timer(env);
                auto          timeIt = [&](std::shared_ptr<vk::ComputePipeline> p, int64_t tot, uint32_t ls) {
                    return timer.time([&](VkCommandBuffer cmd) {
                        p->dispatch(cmd, bufs, &pc, sizeof(pc), groups(tot, ls));
                    });
                };
                // OC-split timing twin of timeIt: the candidate's slice dispatches go back-to-back
                // with no barrier between them, exactly as record() replays the winner. The slices
                // are one op, so they are all inside the one measurement.
                auto timeSplit = [&](std::shared_ptr<vk::ComputePipeline> p, int64_t tot, int parts) {
                    int64_t sliceThreads = ocSplitSliceThreads(tot, parts);
                    return timer.time([&](VkCommandBuffer cmd) {
                        for (int64_t sliceBase = 0; sliceBase < tot; sliceBase += sliceThreads)
                        {
                            ConvPC slicePc  = pc;
                            slicePc.gidBase = (int) sliceBase;
                            p->dispatch(cmd, bufs, &slicePc, sizeof(slicePc), groups(std::min<int64_t>(sliceThreads, tot - sliceBase), env.convLocalSize));
                        }
                    });
                };
                std::vector<uint32_t> cands = (env.tuning == Tuning::Heavy) ? std::vector<uint32_t> {2, 3, 2 | (8u << 8), 3 | (8u << 8), 1 | (8u << 8)} : std::vector<uint32_t> {2, 2 | (8u << 8)};
                // A 1-D kernel (1xK / Kx1, stride 1, dilation 1) adds the sliding-window candidates:
                // same bit-exact accumulation order, ~K/((WTILE+K-1)/WTILE)x less input traffic.
                if (asym1dOk)
                {
                    cands.push_back(1u | (4u << 8) | kChoice1D);
                    cands.push_back(2u | (4u << 8) | kChoice1D);
                    if (env.tuning == Tuning::Heavy)
                    {
                        cands.push_back(1u | (8u << 8) | kChoice1D);
                        cands.push_back(2u | (8u << 8) | kChoice1D);
                    }
                }
                // OC-split slice variants of each conv_reg tile (MNN-style output-channel
                // chunking): the same kernel replayed as 2 or 4 disjoint flat-gid slice dispatches.
                // Bit-neutral - every output is still computed by exactly one thread with unchanged
                // arithmetic - so they join the same race, gated by ocSplitEligible. The LDS-halo
                // kernel decodes gl_WorkGroupID directly (no flat gid to offset), so the split
                // stays scoped to conv_reg.
                {
                    std::vector<uint32_t> sliceCands;
                    for (uint32_t cand: cands)
                    {
                        if (cand & kChoice1D)
                        {
                            continue;
                        }
                        for (int splitFlag: {kChoiceOcSplit2, kChoiceOcSplit4})
                        {
                            if (ocSplitEligible((int) cand | splitFlag))
                            {
                                sliceCands.push_back(cand | (uint32_t) splitFlag);
                            }
                        }
                    }
                    cands.insert(cands.end(), sliceCands.begin(), sliceCands.end());
                }
                uint32_t kaxis = (pc.KW > 1) ? 0u : 1u;
                uint32_t klen  = (uint32_t) std::max(pc.KH, pc.KW);
                // One entrant per raced choice, the plain direct kernel first: it is what
                // Tuning::None dispatches, so seeding the race with it keeps a race that resolves
                // nothing on the level's own default. `margin` is the factor a challenger's time must
                // clear (1.0 = strictly faster); the whole list is timed interleaved below, so no
                // entrant sits on a systematically warmer GPU than another.
                struct OcbEntrant {
                    int                     choice;
                    double                  margin;
                    vk::KernelCost          cost;
                    std::function<double()> time;
                };
                // Compulsory traffic per side: the activation side is the input map plus the
                // output map, the weight side is the weight set. Identical for every entrant - only
                // the ISSUED traffic and the wave count differ - so both are computed once.
                const int64_t Cinb              = cBlocks(pc.Cin);
                const double  streamFootprint   = (double) (x.n * Cinb * x.h * x.w) + (double) (x.n * Coutb * HW);
                const double  residentFootprint = (double) (pc.Cout * Cinb * pc.KH * pc.KW);
                // Per thread the direct and conv_reg kernels issue WTILE input vec4 and OCB*4
                // weight vec4 per (input channel-block, tap) and store WTILE*OCB.
                auto tileCost = [&](int64_t threads, int64_t taps, double wt, double ocb, double loadsPerTap, int64_t wgroups, int dispatchCount) {
                    vk::KernelCost cost;
                    cost.streamVec4            = (double) threads * ((double) Cinb * (double) taps * loadsPerTap + wt * ocb);
                    cost.residentVec4          = (double) threads * (double) Cinb * (double) taps * 4.0 * ocb;
                    cost.streamFootprintVec4   = streamFootprint;
                    cost.residentFootprintVec4 = residentFootprint;
                    cost.waves                 = (double) wgroups;
                    cost.dispatches            = dispatchCount;
                    return cost;
                };
                std::vector<OcbEntrant> entrants;
                entrants.push_back({0, 1.0, tileCost(x.n * Coutb * HW, pc.KH * pc.KW, 1.0, 1.0, 1.0, groups(x.n * Coutb * HW, 64), 1), [&] {
                                        return timeIt(env.pipeline(shader((std::string("conv") + epi.suffix()).c_str(), env.useFp16), 4 + epi.extraBufs(), sizeof(ConvPC), {64u}), x.n * Coutb * HW, 64);
                                    }});
                for (uint32_t cand: cands)
                {
                    uint32_t ocb = cand & 0xffu, wt = std::max(4u, (cand >> 8) & 0xffu);
                    int64_t  ocbGroups = (Coutb + ocb - 1) / ocb;
                    int      parts     = (cand & kChoiceOcSplit2) ? 2 : ((cand & kChoiceOcSplit4) ? 4 : 1);
                    // Every challenger carries the anti-noise margin, not just the tiled/split
                    // classes: the race's noise on these shapes is wider than the differences it
                    // resolves, so an unmargined challenger displaces the proven default on noise.
                    double margin = vk::kTuneRaceMargin;
                    if (cand & kChoice1D)
                    {
                        int64_t alen = (kaxis == 0) ? y.w : y.h;
                        int64_t clen = (kaxis == 0) ? y.h : y.w;
                        int64_t tot  = x.n * ocbGroups * clen * ((alen + wt - 1) / wt);
                        // The sliding window loads WTILE+K-1 input vec4 for the whole K-tap run
                        // instead of WTILE per tap, so its input term is per RUN, not per tap.
                        vk::KernelCost cost = tileCost(tot, klen, (double) wt, (double) ocb, (double) (wt + klen - 1) / (double) klen, groups(tot, 64), 1);
                        entrants.push_back({(int) cand, margin, cost, [&, tot, ocb, wt] {
                                                return timeIt(env.pipeline((std::string("conv_1d") + epi.suffix() + "_fp16").c_str(), 4 + epi.extraBufs(), sizeof(ConvPC), {ocb, wt, kaxis, klen, env.convLocalSize}), tot, env.convLocalSize);
                                            }});
                    } else
                    {
                        int64_t        tot  = x.n * ocbGroups * ((HW + wt - 1) / wt);
                        vk::KernelCost cost = tileCost(tot, pc.KH * pc.KW, (double) wt, (double) ocb, (double) wt, groups(tot, 64), parts);
                        entrants.push_back({(int) cand, margin, cost, [&, tot, parts, ocb, wt] {
                                                auto pipe = env.pipeline(shader((std::string("conv_reg") + epi.suffix()).c_str(), env.useFp16), 4 + epi.extraBufs(), sizeof(ConvPC), {ocb, wt, env.convLocalSize});
                                                return parts > 1 ? timeSplit(pipe, tot, parts) : timeIt(pipe, tot, env.convLocalSize);
                                            }});
                    }
                }
                // The LDS input-halo 3x3 joins the same bit-exact race when eligible: its
                // accumulation is value-identical to the direct kernel (fp32 accumulate, icb/ky/kx
                // tap order, pad taps add +0.0), so the swap needs no anti-noise margin either.
                // Its input term is the staged halo per workgroup, not a per-thread re-read.
                auto ldsCost = [&](int64_t wgroups, int64_t ts) {
                    vk::KernelCost cost;
                    // The halo is staged once per workgroup, so the LDS kernel's activation term
                    // is the halo plus the stores rather than a per-thread re-read.
                    cost.streamVec4            = (double) wgroups * ((double) Cinb * (double) (ts + 2) * (double) (ts + 2) + (double) (ts * ts));
                    cost.residentVec4          = (double) wgroups * (double) (ts * ts) * (double) Cinb * 9.0 * 4.0;
                    cost.streamFootprintVec4   = streamFootprint;
                    cost.residentFootprintVec4 = residentFootprint;
                    cost.waves                 = (double) wgroups * (double) (ts * ts) / 64.0;
                    return cost;
                };
                if (lds3x3)
                {
                    entrants.push_back({kChoiceLds3x3, 1.0, ldsCost(ldsG, 8), [&] {
                                            return timeIt(env.pipeline((std::string("conv3x3_lds") + epi.suffix() + "_fp16").c_str(), 4 + epi.extraBufs(), sizeof(ConvPC)), ldsG * 64, 64);
                                        }});
                }
                // The 16x16-tile halo variant joins the same bit-exact race: the tile size only
                // remaps threads to outputs (the halo overhead drops from 1.56x to 1.27x of the
                // tile's input reads); an anti-noise margin guards the new class.
                if (lds16Ok)
                {
                    entrants.push_back({kChoiceLds16, 0.97, ldsCost(ldsG16, 16), [&] {
                                            return timeIt(env.pipeline((std::string("conv3x3_lds") + epi.suffix() + "_fp16").c_str(), 4 + epi.extraBufs(), sizeof(ConvPC), {16u, 256u}), ldsG16 * 256, 256);
                                        }});
                }
                std::vector<vk::KernelCost> costs;
                costs.reserve(entrants.size());
                for (const OcbEntrant &entrant: entrants)
                {
                    costs.push_back(entrant.cost);
                }
                std::vector<double> ms = vk::racePruned(costs, vk::deviceTuneModel(env), [&](int index) {
                    return entrants[(size_t) index].time();
                });
                // entrants[0] is the deterministic default and stays the incumbent: each challenger
                // is measured against ITS time, not against a running best that would compound the
                // margin and make the outcome depend on list order.
                const std::vector<double> model  = vk::modelEstimates(costs, vk::deviceTuneModel(env));
                int                       best   = 0;
                double                    bestMs = ms[0];
                for (size_t ei = 1; ei < entrants.size(); ++ei)
                {
                    // The anti-noise margin is waived for a challenger the model also ranks cheaper
                    // than the incumbent (see kTuneRaceMargin).
                    const double margin = (model[ei] < model[0]) ? 1.0 : entrants[ei].margin;
                    if (ms[ei] < ms[0] * margin && ms[ei] < bestMs)
                    {
                        bestMs = ms[ei];
                        best   = entrants[ei].choice;
                    }
                }
                VKNN_DEBUG << "autotune " << sig << " -> ocb=" << best << vk::raceTimes(ms);
                if (env.weights)
                {
                    env.weights->setTuned(sig, best, (int) env.tuning);
                }
                return best;
            }

            // Decide the implicit-GEMM kernel (conv_gemm.comp, at its heuristic M tile) vs the direct
            // kernel for this shape: 0 = direct, else the M tile. This is how ConvGemm serves a shape
            // without the opt-in convert-time lowering. The two kernels reduce K in different orders
            // (fp16-floor equivalent, not byte-identical), so the choice is made by a deterministic
            // shape rule, NOT a timing race: a timing race lets thermal/DVFS noise flip the winner
            // between cold sweeps, which would change the output bits run-to-run and make --tuning
            // alter the result. The rule holds at every tuning level (None included), independent of
            // thermal and cache, so the output is byte-identical across runs and tuning levels.
            // `dchoice` is the direct race's winner (0 = plain direct, conv_reg's OCB, or kChoiceLds3x3),
            // still used to configure the direct kernel when this returns 0.
            int tuneConvGemm(const Node &node, VkOpEnv &env, NCHW x, NCHW y, int64_t Cout, int64_t Coutb, int64_t KH, int64_t KW, int dchoice) {
                (void) node;
                (void) env;
                (void) Coutb;
                (void) dchoice;
                // A residual add never takes the gemm path.
                if (hasRes)
                {
                    return 0;
                }
                int64_t M = y.h * y.w, K = x.c * KH * KW;
                int     tm = convGemmTileM(M);
                (void) K;
                // The gemm kernel tiles M on dispatch Y and batch on Z; neither survives the device
                // group-count limit (the runtime 1-D split only rescues X). An overflowing node keeps
                // the direct kernels.
                if ((M + tm - 1) / tm > 65535 || (Cout + kConvGemmTileN - 1) / kConvGemmTileN > 65535 || x.n > 65535)
                {
                    return 0;
                }
                // Implicit-GEMM wins only when its M x N tiling has enough parallelism to amortize the
                // tiled-GEMM setup: at least kGemmMinCoutTiles output-channel tiles (N axis, so
                // Cout >= 8*64 = 512) and kGemmMinM output rows (M axis). Below that the direct
                // kernel is faster. Measured winners on the conv suite: every CNN conv stays direct;
                // the wide patch-embed convs take implicit-GEMM.
                const bool useGemm = Cout >= (int64_t) kGemmMinCoutTiles * kConvGemmTileN && M >= kGemmMinM;
                return useGemm ? tm : 0;
            }

            // Autotune the 3x3 conv kernel for THIS shape. Returns 0 = direct, else a Winograd
            // bitfield: bits 0-1 = F-unit (1 = F(2,3), 2 = F(4,3), 3 = F(6,3)), bit 2 = wino_gemm
            // RM 8 (else 4), bit 3 = wino_gemm fp16 accumulation, bit 4 = the no-LDS register-tile
            // GEMM body (wino_gemm_reg). Winograd wins big on deep, square 3x3 but loses on
            // small-channel / spatially-large 3x3, so the choice is measured per-shape on scratch
            // buffers and cached like the local-size tune. F(4,3) (0.56x the V/M traffic, 4x FLOP
            // saving) is only considered when fp16-safe (allowF4).
            int tuneWino(VkOpEnv &env, NCHW x, NCHW y, int64_t Cin, int64_t Cout, int act) {
                if (env.winograd == Mode::Off)
                {
                    return 0;
                }
                // An explicit WinogradUnit hint (4 = F(4,3), 6 = F(6,3), the separable two-stage LDS
                // transforms of core/wino_f63.h) pins the F-unit and forces Winograd on every eligible
                // shape, but it still runs the shared bit-neutral GEMM-body race below — a forced unit
                // then executes in exactly the kernel configuration the automatic rule gives it, which
                // is what makes a forced-unit measurement comparable to the automatic one.
                const int  forcedUnit = cfgHint(env, Hint::WinogradUnit); // 0 = automatic, else the F-unit edge
                const bool forceUnit  = (forcedUnit == 2 || forcedUnit == 4 || forcedUnit == 6);
                bool       forceOn    = (env.winograd == Mode::On) || forceUnit;
                // Winograd-vs-direct is decided by a DETERMINISTIC shape rule, not a timing race: the
                // Winograd transform rounds fp16 differently from the direct kernel, so a timing race
                // lets thermal/DVFS noise flip the winner between cold sweeps, changing the output bits
                // run-to-run and making --tuning alter the result. Winograd's fp16 transform-domain
                // intermediates (V, M) grow with Cin*Cout and keep it memory-bound, so it loses to the
                // direct kernel once Cin*Cout exceeds kWinoMaxCinCout - UNLESS the output map supplies
                // enough tiles to keep the batched GEMM fed (kWinoLargeCMinPixels; probe data at the
                // constants). The rule holds at every tuning level, independent of thermal and cache.
                if (!(forceOn || Cin * Cout <= kWinoMaxCinCout || (int64_t) y.h * y.w >= kWinoLargeCMinPixels))
                {
                    return 0;
                }
                int64_t Cinb = cBlocks(Cin), Coutb = cBlocks(Cout);
                // Pick the Winograd output-tile edge (2=F(2,3), 4=F(4,3)) DETERMINISTICALLY from the shape,
                // NOT by timing: two candidate times within measurement noise flip the choice run-to-run, and
                // the F-units round fp16 differently, so a timing-raced tile breaks bit-exactness (identical
                // per run and across cache rebuilds). The cost model (winoCostPerOutput) and the rule live in
                // core/wino_f63.h so the host gating test pins the rule's choices over a shape sweep:
                // F(4,3)'s 4x FLOP / 0.56x V-M-traffic saving wins on deep channels, F(2,3)'s smaller
                // transform wins on shallow, and F(6,3) stays out of the automatic rule — device
                // measurement refuted it (accuracy gate + no shape-only win; evidence at winoAutoUnit).
                int U_ = forceUnit ? forcedUnit : winoAutoUnit(Cin, Cout);
                // RM (wino_gemm tiles/thread) and the GEMM body (LDS-staged vs the no-LDS register
                // twin wino_gemm_reg) are bit-neutral - they only remap threads to outputs / change
                // the operand staging while the per-output K order is unchanged, so any choice yields
                // identical output bits; both stay timing-raced for throughput under Fast/Heavy and
                // are cached, defaulting to LDS RM4 under None. ACC16 (fp16 accumulation) is
                // pinned OFF: the fp16-accumulate variant changes the output bits, so it must never be
                // selectable or it would reintroduce tuning/thermal-dependent output.
                // Cached value = RM (4|8) | 16 when the register body won. The stem is winorm3_ and
                // carries the F-unit: the body race runs against THAT unit's tile count, so a winner
                // raced for one unit must never be decoded for another (winorm_ entries encode RM alone
                // and winorm2_ entries omit the unit; neither may decode against this key).
                char buf[128];
                snprintf(buf, sizeof(buf), "winorm3_%d_%d_%d_%d_%d", (int) Cin, (int) Cout, (int) y.h, (int) y.w, U_);
                std::string sig         = env.gpuTag + "/" + buf;
                int         bestRm      = 4;
                bool        bestRegGemm = false;
                int         reuse;
                if (env.reuseTuned(sig, reuse) && ((reuse & ~16) == 4 || (reuse & ~16) == 8))
                {
                    bestRm      = reuse & ~16;
                    bestRegGemm = (reuse & 16) != 0;
                } else if (env.tuning != Tuning::None && env.runner)
                {
                    int     nPos = (U_ + 2) * (U_ + 2);
                    int64_t nTH = (y.h + U_ - 1) / U_, nTW = (y.w + U_ - 1) / U_, nT = x.n * nTH * nTW;
                    auto    mk = [&](size_t bytes) {
                        return std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>(bytes, 16), vk::MemPref::kDeviceOnly);
                    };
                    auto        sSrc   = mk((size_t) x.n * Cinb * x.h * x.w * 8);
                    auto        sU     = mk((size_t) nPos * Cout * Cinb * 8);
                    auto        sV     = mk((size_t) nPos * Cinb * nT * 8);
                    auto        sM     = mk((size_t) nPos * nT * Coutb * 8);
                    auto        sBias  = mk((size_t) Coutb * 8);
                    auto        sDst   = mk((size_t) x.n * Coutb * y.h * y.w * 8);
                    WinoInPC    ipc    = {(int) x.n, (int) Cin, (int) x.h, (int) x.w, (int) y.h, (int) y.w, (int) nTH, (int) nTW};
                    WinoGemmPC  gpc    = {(int) Cin, (int) Cout, (int) nT};
                    WinoFusedPC opc    = {(int) x.n, (int) Cin, (int) Cout, (int) y.h, (int) y.w, (int) nTH, (int) nTW, act, 0.f, 0.f};
                    auto        inPipe = env.pipeline(U_ == 2 ? "wino_input_fp16" : (U_ == 4 ? "wino_input4_fp16" : "wino_input6_fp16"), 2, sizeof(WinoInPC));
                    // The output pass hosts the fused epilogue, so the timed arm spells the same
                    // stem + suffix at the same binding count record() dispatches — the epilogue's
                    // register demand is part of what this race decides.
                    std::string           oName = (U_ == 2) ? std::string("wino_out") + epi.suffix() :
                                                  (U_ == 4) ? std::string("wino_out4") + epi.suffix() :
                                                              std::string("wino_out6") + epi.suffix();
                    auto                  oPipe = env.pipeline((oName + "_fp16").c_str(), 3 + epi.extraBufs(), sizeof(WinoFusedPC));
                    std::vector<VkBuffer> oBufs = {sM->handle(), sBias->handle(), sDst->handle()};
                    epi.appendForTiming(oBufs, sDst->handle());
                    uint32_t gy = groups(Coutb, kWinoGemmTileNB);
                    // F(6,3)'s separable transforms run kWinoF63TransformLanes cooperating threads per
                    // (channel-block, tile) unit, so the timed transform dispatches match record()'s.
                    int64_t       lanes = (U_ == 6) ? kWinoF63TransformLanes : 1;
                    vk::TuneTimer timer(env);
                    // Race only the bit-neutral choices — the RM tile and the GEMM body; ACC16 is
                    // fixed to 0 (fp32 accumulate). One full 3-pass per candidate so the GEMM's
                    // share of the pipeline is what is measured.
                    auto time3Pass = [&](std::shared_ptr<vk::ComputePipeline> gemmPipe, uint32_t gx) {
                        return timer.time([&](VkCommandBuffer cmd) {
                            inPipe->dispatch(cmd, {sSrc->handle(), sV->handle()}, &ipc, sizeof(ipc), groups(Cinb * nT * lanes, 64));
                            vk::computeBarrier(*env.ctx, cmd);
                            gemmPipe->dispatch(cmd, {sV->handle(), sU->handle(), sM->handle()}, &gpc, sizeof(gpc), gx, gy, (uint32_t) nPos);
                            vk::computeBarrier(*env.ctx, cmd);
                            oPipe->dispatch(cmd, oBufs, &opc, sizeof(opc), groups(Coutb * nT * lanes, 64));
                        });
                    };
                    // Entrants in one interleaved race: LDS RM4 (the incumbent Tuning::None
                    // dispatches), LDS RM8, then the register body at RM4/RM8.
                    constexpr int                        kWinoRmCands[] = {4, 8, 4, 8};
                    std::shared_ptr<vk::ComputePipeline> gemmPipes[]    = {
                        env.pipeline("wino_gemm_fp16", 3, sizeof(WinoGemmPC), {4u, 0u}),
                        env.pipeline("wino_gemm_fp16", 3, sizeof(WinoGemmPC), {8u, 0u}),
                        env.pipeline("wino_gemm_reg_fp16", 3, sizeof(WinoGemmPC), {4u}),
                        env.pipeline("wino_gemm_reg_fp16", 3, sizeof(WinoGemmPC), {8u}),
                    };
                    std::vector<double> ms    = vk::raceCandidates(4, [&](int index) {
                        return time3Pass(gemmPipes[index], groups(nT, winoGemmTileM(kWinoRmCands[index])));
                    });
                    double              ldsMs = ms[0], regMs = ms[2];
                    int                 ldsRm = 4, regRm = 4;
                    if (ms[1] < ldsMs)
                    {
                        ldsMs = ms[1];
                        ldsRm = 8;
                    }
                    // The no-LDS register-tile body joins the same bit-exact race (identical
                    // per-output accumulation; only the operand staging differs). A NEW candidate
                    // class carries the 3% anti-noise margin against the LDS incumbent.
                    if (ms[3] < regMs)
                    {
                        regMs = ms[3];
                        regRm = 8;
                    }
                    if (regMs < ldsMs * 0.97)
                    {
                        bestRm      = regRm;
                        bestRegGemm = true;
                    } else
                    {
                        bestRm = ldsRm;
                    }
                    if (env.weights)
                    {
                        env.weights->setTuned(sig, bestRm | (bestRegGemm ? 16 : 0), (int) env.tuning);
                    }
                }
                int winoChoice = ((U_ == 2) ? 1 : (U_ == 4) ? 2 : 3) | (bestRm == 8 ? 4 : 0) | (bestRegGemm ? 16 : 0); // ACC16 bit (8) never set
                VKNN_DEBUG << "tuneWino Cin=" << Cin << " Cout=" << Cout << " U=" << U_ << " rm=" << bestRm << " body=" << (bestRegGemm ? "reg" : "lds") << " -> " << winoChoice;
                return winoChoice;
            }

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g    = *env.graph;
                NCHW         x    = NCHW::from(g.desc(node.inputs[0]).shape);
                NCHW         y    = NCHW::from(g.desc(node.outputs[0]).shape);
                const Shape &ws   = g.desc(node.inputs[1]).shape; // [Cout, Cin/group, KH, KW]
                int64_t      Cout = ws[0], inCg = ws[1], KH = ws[2], KW = ws[3];
                auto         st  = attrInts(node, "strides", {1, 1});
                auto         dil = attrInts(node, "dilations", {1, 1});
                // Resolved through the shared forward geometry (core/conv_geom.h) so auto_pad convs
                // carry real begin/end pads here: the kernel-eligibility checks below (pointwise /
                // wino / lds3x3) and the push-constant begin pads all read the resolved values. The
                // output extent itself comes from the graph desc (inferShapes, same helper).
                auto    pad   = convGeom(x.h, x.w, KH, KW, node.attr).pads();
                int64_t group = node.attr.geti("group", 1);
                hasRes        = (node.fusedResidual != kNoTensor); // set by the residual-Add fusion pass (1x1 only)
                depthwise     = (group == x.c && group == Cout && inCg == 1);
                pointwise = (!depthwise && group == 1 && KH == 1 && KW == 1 && st[0] == 1 && st[1] == 1 && pad[0] == 0 && pad[1] == 0 && pad[2] == 0 && pad[3] == 0);
                pwS2 = (!depthwise && group == 1 && KH == 1 && KW == 1 && (st[0] > 1 || st[1] > 1) && pad[0] == 0 && pad[1] == 0 && pad[2] == 0 && pad[3] == 0);
                // Winograd F(2,3) via a tiled GEMM is eligible only for deep, square 3x3/s1/p1 group-1
                // convs in fp16; tuneWino picks it per shape (it loses on small-channel / spatially-large
                // 3x3) and caches the choice.
                // Cout >= 64: below two wino_gemm N-tiles (Coutb < 16) the transform-domain GEMM
                // starves its N axis and runs far under the direct kernels (the DenseNet 128->32
                // growth convs measured ~365 GF/s on it); those shapes take the direct/OCB race.
                bool winoShape = (env.useFp16 && !depthwise && group == 1 && KH == 3 && KW == 3 && st[0] == 1 && st[1] == 1 && pad[0] == 1 && pad[1] == 1 && pad[2] == 1 && pad[3] == 1 && x.c >= 32 && Cout >= 64);
                // The epilogue is resolved before any race runs: every race builds the kernel the
                // graph will actually dispatch (stem + epi.suffix()), and the epilogue's register
                // demand is part of what the race is deciding. Preparing it after tuneWino left the
                // Winograd race timing plain kernels while its siblings timed fused ones.
                epi.prepare(node, env, /*flat=*/false, g.desc(node.outputs[0]).shape);
                int wchoice = winoShape ? tuneWino(env, x, y, x.c, Cout, (int) node.fusedAct) : 0;
                winograd    = (wchoice > 0);
                winoUnit    = ((wchoice & 3) == 2) ? 4 : ((wchoice & 3) == 3) ? 6 : 2;
                winoRm      = (wchoice & 4) != 0 ? 8 : 4;
                winoAcc16   = (wchoice & 8) != 0 ? 1 : 0;
                winoRegGemm = (wchoice & 16) != 0;

                std::vector<float> wsrcv = initFloats(g, node.inputs[1]);
                const float       *wsrc  = wsrcv.data();
                int64_t            Coutb = cBlocks(Cout);

                // bias, padded out to a multiple of 4 so the kernel can read whole vec4s
                bbuf = uploadCached(env, node.name + "#b", [&] {
                    std::vector<float> bias(Coutb * 4, 0.f);
                    // inputs[2] is bias unless it's the appended residual (no-bias + fused-residual case)
                    if (pwCoreInputs(node) > 2 && node.inputs[2] != kNoTensor && node.inputs[2] != node.fusedResidual)
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
                    // Bit-exact tile race: 1 pixel/thread vs the 2x2 output tile (dwconv_t2).
                    bool dwTile2 = pickDwTile(env, x, y, Cb) == 1;
                    total        = dwTile2 ? x.n * Cb * ((y.h + 1) / 2) * ((y.w + 1) / 2) : x.n * Cb * y.h * y.w;
                    pipe = env.pipeline(shader((std::string(dwTile2 ? "dwconv_t2" : "dwconv") + epi.suffix()).c_str(), env.useFp16), 4 + epi.extraBufs(), sizeof(DwPC), std::vector<uint32_t> {env.convLocalSize});
                } else
                {
                    int64_t Cinb = cBlocks(x.c);
                    pc           = {(int) x.n,   (int) x.c,   (int) x.h,    (int) x.w,    (int) Cout,   (int) y.h,    (int) y.w,           (int) KH,   (int) KW,
                                    (int) st[0], (int) st[1], (int) pad[0], (int) pad[1], (int) dil[0], (int) dil[1], (int) node.fusedAct, node.actLo, node.actHi};
                    total        = x.n * Coutb * y.h * y.w;
                    wbuf         = uploadCached(env, node.name + "#w", [&] {
                        // [Cout,Cin,KH,KW] -> [Cout][Cinb][KH][KW][4], with the output-channel rows
                        // padded out to whole blocks (Coutb*4): the kernels read all 4 rows of a block,
                        // so a partial last block must resolve to zero rows, not out-of-bounds loads.
                        std::vector<float> wp((size_t) Coutb * 4 * Cinb * KH * KW * 4, 0.f);
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
                            // General split-K shape rule (non-pointwise): deep reduction + starved standard
                    // dispatch. Skipped when a DirectConv3x3 hint forces a specific kernel.
                    int64_t skOHW        = y.h * y.w;
                    int64_t skTaps       = Cinb * KH * KW;
                    int64_t skTapsFloor  = (KH * KW >= 3) ? kSplitKGenMinTapsMulti : kSplitKGenMinTaps;
                    int64_t skStdThreads = Coutb * ((skOHW + kTile - 1) / kTile);
                    // Hint::SplitKConv: Auto = the calibrated rule below; On = every structurally
                    // eligible shape; Off = never. An explicit DirectConv3x3 kernel force wins in
                    // every mode.
                    int  skHint       = cfgHint(env, Hint::SplitKConv);
                    bool skStructural = env.useFp16 && x.n == 1 && group == 1 && !pointwise && cfgHint(env, Hint::DirectConv3x3) == 0;
                    bool starvedDeep  = skHint == (int) Mode::Off ? false :
                                        skHint == (int) Mode::On  ? skStructural :
                                                                   skStructural && skTaps >= skTapsFloor && skOHW <= kSplitKGenMaxOHW && skStdThreads < kSplitKGenMaxThreads;
                    if (pointwise)
                    {
                        // Deep, small-spatial 1x1 convs have too few threads for the register-tiled kernel; use
                        // split-K there (parallelize the channel reduction). The rule lives in
                        // pw_splitk_rule.h so the fused depthwise+project op reproduces the same
                        // summation order for the pairs it swallows.
                        int64_t HW = y.h * y.w;
                        splitk     = pwSplitKActive(env.useFp16, x.n, x.c, Coutb, HW, skHint);
                        if (splitk)
                        {
                            prepareSplitK(node, env, x, y, Cout, Coutb);
                        } else
                        {
                            uint32_t pick     = pickWTile(env, false, x, y, Cout, Coutb);
                            wTile             = pick & 0xffu;
                            ocbTile           = std::max(1u, pick >> 8);
                            int64_t nTiles    = (HW + wTile - 1) / wTile;
                            int64_t ocbGroups = (Coutb + ocbTile - 1) / ocbTile;
                            total             = x.n * ocbGroups * nTiles;
                            pipe = env.pipeline(shader((std::string("conv1x1") + epi.suffix()).c_str(), env.useFp16), epi.active ? 5 + epi.extraBufs() : (hasRes ? 5u : 4u), sizeof(ConvPC), std::vector<uint32_t> {(uint32_t) (hasRes ? 1 : 0), wTile, ocbTile, env.convLocalSize});
                        }
                    } else if (starvedDeep)
                    {
                        // split-K partial + reduce for the starved deep shapes (strided 1x1 downsamples,
                        // small-map KxK); the reduce pass carries bias/residual/act/epilogue.
                        splitk    = true;
                        splitkGen = true;
                        prepareSplitKGeneral(node, env, x, y, Cout, Coutb, KH, KW, st, pad, dil);
                    } else if (pwS2)
                    {
                        // strided 1x1 (downsample): register-tiled kernel that gathers the input at the
                        // stride; reuses weights across WTILE output pixels (the general direct kernel does 1).
                        int64_t  HW       = y.h * y.w;
                        uint32_t pick     = pickWTile(env, true, x, y, Cout, Coutb);
                        wTile             = pick & 0xffu;
                        ocbTile           = std::max(1u, pick >> 8);
                        int64_t ocbGroups = (Coutb + ocbTile - 1) / ocbTile;
                        total             = x.n * ocbGroups * ((HW + wTile - 1) / wTile);
                        pipe = env.pipeline(shader((std::string("conv1x1_s2") + epi.suffix()).c_str(), env.useFp16), epi.active ? 5 + epi.extraBufs() : (hasRes ? 5u : 4u), sizeof(ConvPC), std::vector<uint32_t> {(uint32_t) (hasRes ? 1 : 0), wTile, ocbTile, env.convLocalSize});
                    } else if (cfgHint(env, Hint::DirectConv3x3) == 2 && env.useFp16 && KH == 3 && KW == 3 && st[0] == 1 && st[1] == 1 && pad[0] == 1 && pad[1] == 1 && pad[2] == 1 && pad[3] == 1 && dil[0] == 1 && dil[1] == 1 && y.h >= 14 && y.w >= 14)
                    {
                        // LDS input-halo 3x3 for the larger-spatial layers (input reuse via shared memory). 7x7
                        // layer4 stays on the direct kernel (tile barely fills, halo overhead dominates).
                        lds         = true;
                        int64_t nTX = (y.w + 7) / 8, nTY = (y.h + 7) / 8;
                        ldsGroups = x.n * Coutb * nTY * nTX;
                        pipe = env.pipeline((std::string("conv3x3_lds") + epi.suffix() + "_fp16").c_str(), 4 + epi.extraBufs(), sizeof(ConvPC), std::vector<uint32_t> {});
                    } else if (cfgHint(env, Hint::DirectConv3x3) == 1)
                    {
                        // register-tiled implicit-im2col (opt-in). Regresses 3x3 on this GPU: small weight
                        // tensors already cache well, so WTILE overhead + extra input loads dominate.
                        reg        = true;
                        int64_t HW = y.h * y.w;
                        total      = x.n * Coutb * ((HW + kTile - 1) / kTile);
                        pipe = env.pipeline(shader((std::string("conv_reg") + epi.suffix()).c_str(), env.useFp16), 4 + epi.extraBufs(), sizeof(ConvPC), std::vector<uint32_t> {});
                    } else
                    {
                        // DirectAuto: the bit-exact direct race picks the baseline (1-pixel direct,
                        // conv_reg OCB tiling, or the LDS-halo 3x3 when eligible), then the
                        // implicit-GEMM shape rule (fp16-floor, deterministic at every tuning
                        // level) may take the shape from that baseline.
                        bool lds3x3 = env.useFp16 && KH == 3 && KW == 3 && st[0] == 1 && st[1] == 1 && pad[0] == 1 && pad[1] == 1 && pad[2] == 1 && pad[3] == 1 && dil[0] == 1 && dil[1] == 1 && y.h >= 14 && y.w >= 14;
                        int ocb = env.useFp16 ? pickOcb(env, x, y, Cout, Coutb, lds3x3) : 0;
                        int gtm = (env.useFp16 && group == 1 && KH * KW > 1) ? tuneConvGemm(node, env, x, y, Cout, Coutb, KH, KW, ocb) : 0;
                        if (gtm > 0)
                        {
                            // implicit-GEMM kernel at the raced M tile; binds Src/Wt/Bs/Dst exactly like
                            // the ConvGemm op, with the same epilogue plumbing as the direct path. The
                            // zero-padded bbuf keeps the unconditional bias add the direct kernel does.
                            gemm = true;
                            // The weight panel is built here, so it can always present the
                            // 4-aligned physical row stride the vec4-weight twin needs: the route
                            // (convGemmWVec4Route, core/conv_gemm_route.h) is a pure alignment
                            // decision and the twin is byte-identical to conv_gemm.
                            const ConvGemmWVec4Route wRoute = convGemmWVec4Route(Cout, /*weightRepackable=*/true);
                            const int64_t            coutP  = wRoute.eligible ? wRoute.coutP : Cout;
                            gpc                             = {pc.Cin, pc.H,  pc.W,  pc.Cout, pc.OH,  pc.OW, pc.KH,    pc.KW,    pc.SH,      pc.SW,
                                                               pc.PT,  pc.PL, pc.DH, pc.DW,   pc.act, 1,     pc.actLo, pc.actHi, (int) coutP};
                            // A padded panel gets its own cache key: same node, different bytes at
                            // the same length-per-row, so it must never alias the packed entry.
                            gwbuf = uploadCached(env, node.name + (wRoute.padW ? "#gemmwp" : "#gemmw"), [&] {
                                // [Cout,Cin,KH,KW] -> [K][coutP], k = (ky*KW+kx)*Cin+ic (the kernel's
                                // channel-fastest k order; matches lowerConv's convert-time repack).
                                // coutP > Cout leaves each row's tail channels at the zero fill —
                                // the value the kernel's own column guard yields, so the pad is
                                // output-byte-neutral.
                                std::vector<float> wp((size_t) x.c * KH * KW * coutP, 0.f);
                                for (int64_t oc = 0; oc < Cout; ++oc)
                                {
                                    for (int64_t ic = 0; ic < x.c; ++ic)
                                    {
                                        for (int64_t t = 0; t < KH * KW; ++t)
                                        {
                                            wp[(size_t) ((t * x.c + ic) * coutP + oc)] = wsrc[(oc * x.c + ic) * KH * KW + t];
                                        }
                                    }
                                }
                                return wp;
                            });
                            wbuf.reset(); // the NC4HW4 pack only served the race timing
                            int64_t M = y.h * y.w;
                            ggx       = (uint32_t) ((Cout + kConvGemmTileN - 1) / kConvGemmTileN);
                            ggy       = (uint32_t) ((M + gtm - 1) / gtm);
                            ggz       = (uint32_t) x.n;
                            pipe = env.pipeline(shader((std::string(wRoute.eligible ? "conv_gemm_wv4" : "conv_gemm") + epi.suffix()).c_str(), env.useFp16), 4 + epi.extraBufs(), sizeof(ConvGemmPC), std::vector<uint32_t> {(uint32_t) gtm});
                        } else if (ocb == kChoiceLds3x3 || ocb == kChoiceLds16)
                        {
                            // LDS input-halo 3x3 at the autotuned tile edge (8x8 or 16x16; won the
                            // bit-exact direct race for this shape).
                            lds          = true;
                            uint32_t ts  = (ocb == kChoiceLds16) ? 16u : 8u;
                            int64_t  nTX = (y.w + ts - 1) / ts, nTY = (y.h + ts - 1) / ts;
                            ldsGroups = x.n * Coutb * nTY * nTX;
                            pipe = env.pipeline((std::string("conv3x3_lds") + epi.suffix() + "_fp16").c_str(), 4 + epi.extraBufs(), sizeof(ConvPC), std::vector<uint32_t> {ts, ts * ts});
                        } else if ((ocb & kChoice1D) != 0)
                        {
                            // sliding-window 1-D kernel (1xK / Kx1; autotuned, won the bit-exact race).
                            reg                = true;
                            uint32_t regOcb    = (uint32_t) (ocb & 0xff);
                            uint32_t regWt     = std::max(4u, (uint32_t) (ocb >> 8) & 0xffu);
                            uint32_t kaxis     = (KW > 1) ? 0u : 1u;
                            uint32_t klen      = (uint32_t) std::max(KH, KW);
                            int64_t  alen      = (kaxis == 0) ? y.w : y.h;
                            int64_t  clen      = (kaxis == 0) ? y.h : y.w;
                            int64_t  ocbGroups = (Coutb + regOcb - 1) / regOcb;
                            total              = x.n * ocbGroups * clen * ((alen + regWt - 1) / regWt);
                            pipe = env.pipeline((std::string("conv_1d") + epi.suffix() + "_fp16").c_str(), 4 + epi.extraBufs(), sizeof(ConvPC), std::vector<uint32_t> {regOcb, regWt, kaxis, klen, env.convLocalSize});
                        } else if (ocb > 0)
                        {
                            // register-tiled conv computing OCB output channel-blocks per thread for WTILE
                            // pixels (autotuned; won over the direct kernel for this shape), optionally
                            // replayed as 2/4 OC-split slice dispatches (kChoiceOcSplit2/4). Byte-identical
                            // to the direct kernel.
                            reg                = true;
                            int64_t  HW        = y.h * y.w;
                            uint32_t regOcb    = (uint32_t) (ocb & 0xff);
                            uint32_t regWt     = std::max(4u, (uint32_t) (ocb >> 8) & 0xffu);
                            int64_t  ocbGroups = (Coutb + regOcb - 1) / regOcb;
                            total              = x.n * ocbGroups * ((HW + regWt - 1) / regWt);
                            ocSplitParts       = (ocb & kChoiceOcSplit2) ? 2 : ((ocb & kChoiceOcSplit4) ? 4 : 1);
                            ocSliceThreads     = ocSplitSliceThreads(total, ocSplitParts);
                            pipe = env.pipeline(shader((std::string("conv_reg") + epi.suffix()).c_str(), env.useFp16), 4 + epi.extraBufs(), sizeof(ConvPC), std::vector<uint32_t> {regOcb, regWt, env.convLocalSize});
                        } else
                        {
                            // autotuned 1-pixel-per-thread direct kernel (the fallback when OCB tiling didn't win)
                            total     = x.n * Coutb * y.h * y.w;
                            localSize = pickLocalSize(env);
                            pipe = env.pipeline(shader((std::string("conv") + epi.suffix()).c_str(), env.useFp16), 4 + epi.extraBufs(), sizeof(ConvPC), std::vector<uint32_t> {localSize});
                        }
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
                        std::vector<VkBuffer> wb = {ubuf->handle(), src->handle(), bbuf->handle(), dst->handle()};
                        epi.append(wb, node, env, dst->handle());
                        wFullPipe->dispatch(cmd, wb, &wFusedPC, sizeof(wFusedPC), (uint32_t) wFullGroups);
                        return;
                    }
                    if (winogemm)
                    {
                        // 3-pass: input transform -> V, tiled batched GEMM -> M, output transform -> dst.
                        wInPipe->dispatch(cmd, {src->handle(), vbuf->handle()}, &wInPC, sizeof(wInPC), (uint32_t) wInGroups);
                        vk::computeBarrier(*env.ctx, cmd);
                        wGemmPipe->dispatch(cmd, {vbuf->handle(), ubuf->handle(), mbuf->handle()}, &wGemmPC, sizeof(wGemmPC), (uint32_t) wGemmGX, (uint32_t) wGemmGY, (uint32_t) wGemmGZ);
                        vk::computeBarrier(*env.ctx, cmd);
                        std::vector<VkBuffer> ob = {mbuf->handle(), bbuf->handle(), dst->handle()};
                        epi.append(ob, node, env, dst->handle());
                        wOutPipe->dispatch(cmd, ob, &wFusedPC, sizeof(wFusedPC), (uint32_t) wOutGroups);
                        return;
                    }
                    // 2 stages: input transform -> V, then fused matmul + output transform -> dst.
                    wInPipe->dispatch(cmd, {src->handle(), vbuf->handle()}, &wInPC, sizeof(wInPC), (uint32_t) wInGroups);
                    vk::computeBarrier(*env.ctx, cmd);
                    std::vector<VkBuffer> fb = {ubuf->handle(), vbuf->handle(), bbuf->handle(), dst->handle()};
                    epi.append(fb, node, env, dst->handle());
                    wFusedPipe->dispatch(cmd, fb, &wFusedPC, sizeof(wFusedPC), (uint32_t) wFusedGroups);
                    return;
                }
                // fused residual (1x1 path only); bias is a harmless dummy when not fused (shader won't read
                // it)
                VkBuffer res = (hasRes ? env.devBuf(node.fusedResidual) : bbuf.get())->handle();
                if (splitk)
                {
                    // partial pass (K-parallel) -> reduce pass (+bias [+residual] +act).
                    if (splitkGen)
                    {
                        skPipe->dispatch(cmd, {src->handle(), wbuf->handle(), partBuf->handle()}, &skGenPC, sizeof(skGenPC), (uint32_t) skGroups);
                    } else
                    {
                        skPipe->dispatch(cmd, {src->handle(), wbuf->handle(), partBuf->handle()}, &skPC, sizeof(skPC), (uint32_t) skGroups);
                    }
                    vk::computeBarrier(*env.ctx, cmd);
                    std::vector<VkBuffer> rb = {partBuf->handle(), bbuf->handle(), dst->handle()};
                    if (hasRes)
                    {
                        rb.push_back(res);
                    } else if (epi.active)
                    {
                        rb.push_back(dst->handle()); // epilogue binds after the Res slot: fill it
                    }
                    epi.append(rb, node, env, dst->handle());
                    skRed->dispatch(cmd, rb, &skRedPC, sizeof(skRedPC), (uint32_t) skRedGroups);
                    return;
                }
                if (gemm)
                {
                    // implicit-GEMM winner: Src/Wt/Bs/Dst (the ConvGemm op's binding set) + epilogue.
                    std::vector<VkBuffer> gb = {src->handle(), gwbuf->handle(), bbuf->handle(), dst->handle()};
                    epi.append(gb, node, env, dst->handle());
                    pipe->dispatch(cmd, gb, &gpc, sizeof(gpc), ggx, ggy, ggz);
                    return;
                }
                std::vector<VkBuffer> bufs = {src->handle(), wbuf->handle(), bbuf->handle(), dst->handle()};
                if (depthwise)
                {
                    epi.append(bufs, node, env, dst->handle());
                    pipe->dispatch(cmd, bufs, &dpc, sizeof(dpc), groups(total, env.convLocalSize));
                } else if (pointwise || pwS2)
                {
                    if (hasRes)
                    {
                        bufs.push_back(res);
                    } else if (epi.active)
                    {
                        bufs.push_back(dst->handle()); // epilogue binds after the Res slot: fill it
                    }
                    epi.append(bufs, node, env, dst->handle());
                    pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(total, env.convLocalSize));
                } else if (lds)
                {
                    epi.append(bufs, node, env, dst->handle());
                    pipe->dispatch(cmd, bufs, &pc, sizeof(pc), (uint32_t) ldsGroups);
                } else if (ocSplitParts > 1)
                {
                    // OC-split winner: the flat gid range replayed as 2/4 contiguous, 64-aligned
                    // slice dispatches back-to-back with NO barrier between them - the slices cover
                    // disjoint outputs, so the run stays one hazard-tracking node. Each dispatch
                    // pushes its slice base; the kernel body is unchanged.
                    epi.append(bufs, node, env, dst->handle());
                    for (int64_t sliceBase = 0; sliceBase < total; sliceBase += ocSliceThreads)
                    {
                        ConvPC slicePc  = pc;
                        slicePc.gidBase = (int) sliceBase;
                        pipe->dispatch(cmd, bufs, &slicePc, sizeof(slicePc), groups(std::min<int64_t>(ocSliceThreads, total - sliceBase), env.convLocalSize));
                    }
                } else
                {
                    epi.append(bufs, node, env, dst->handle());
                    pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(total, reg ? env.convLocalSize : localSize));
                }
            }
        };

    } // namespace

    VKNN_REGISTER_VK_OP(OpType::Conv, ConvOp);

} // namespace vknn
