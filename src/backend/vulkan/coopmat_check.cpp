#include "coopmat_check.h"
#include "vk_buffer.h"
#include "vk_command.h"
#include "vk_context.h"
#include "vk_op_env.h"
#include "vknn/dtype.h"
#include <functional>
#include <map>
#include <mutex>
#include <vector>

namespace vknn {

    namespace {
        // GEMM probe: asymmetric dimensions M != N != K, so a kernel that swaps rows for columns,
        // reads A column-major, or walks K with the wrong stride produces different bytes than the
        // oracle. Each dimension honors the kernel contract (M, N multiples of 32, K of 16).
        constexpr int kCheckM = 32;
        constexpr int kCheckN = 64;
        constexpr int kCheckK = 48;

        // Conv probe: a 3x3/s1/p1 conv whose GEMM view exercises every masking path of the staged
        // tile kernel - M = 36 (a full 32-row tile plus a ragged 4-row tail), N = Cout = 22 (a
        // masked N tile with two pad lanes in the last NC4HW4 block, which must store zero), and
        // K = C*9 = 54 (three full fragment depths plus a ragged 6-deep tail). C = 6 is
        // deliberately NOT a multiple of 4: the staged A gather then runs BOTH of its bodies with
        // live data - the aligned channel-block fast path (ic = 0 quads) and the per-element edge
        // decode (quads straddling a channel block or the tap boundary), the branch every
        // C % 4 != 0 conv (an RGB stem is C = 3) depends on exclusively. All values are small
        // integers: |acc| stays far under 2048, so fp16 storage, fp32 accumulation and the fp16
        // store are exact and the comparison is byte equality.
        constexpr int kConvCheckC    = 6;
        constexpr int kConvCheckHW   = 6;
        constexpr int kConvCheckCout = 22;
        constexpr int kConvCheckKHW  = 3;
        constexpr int kConvCheckPad  = 1;

        // Deterministic small-integer operands: every value, every fp32 partial sum and the fp16
        // result are exactly representable, so the device must match the host to the bit.
        inline float checkA(int row, int k) {
            return (float) ((row * 7 + k * 3) % 5 - 2); // in [-2, 2]
        }
        inline float checkB(int k, int col) {
            return (float) ((k * 5 + col) % 7 - 3); // in [-3, 3]
        }

        // Per-(device, kernel) verdict memo: the first caller pays for the probe dispatch, every
        // later instance reads the cached verdict. A no-runner environment memoizes false - the
        // gate must never pass a kernel the device has not actually executed.
        bool memoizedVerdict(VkDevice device, const char *kernel, const std::function<bool()> &probe) {
            static std::mutex                                       verdictMutex;
            static std::map<std::pair<VkDevice, std::string>, bool> verdictByKernel;
            const std::pair<VkDevice, std::string>                  key {device, kernel};
            {
                std::lock_guard<std::mutex> lock(verdictMutex);
                auto                        it = verdictByKernel.find(key);
                if (it != verdictByKernel.end())
                {
                    return it->second;
                }
            }
            bool passed = false;
            try
            { passed = probe(); } catch (const std::exception &e)
            {
                VKNN_WARN << kernel << " self-check failed to run (" << e.what() << "); cooperative-matrix path disabled";
                passed = false;
            }
            std::lock_guard<std::mutex> lock(verdictMutex);
            verdictByKernel[key] = passed;
            return passed;
        }
    } // namespace

    CoopmatGemmCaps fillCoopmatGemmCaps(const vk::VulkanCaps &caps) {
        CoopmatGemmCaps cm;
        cm.coopmatFp16Fp32Row16 = caps.hasCoopmatShape(16, 16, 16, (uint32_t) VK_COMPONENT_TYPE_FLOAT16_KHR, (uint32_t) VK_COMPONENT_TYPE_FLOAT32_KHR);
        cm.coopmatFp8Fp32Row16 = caps.shaderFloat8CoopMat && caps.hasCoopmatShape(16, 16, 16, (uint32_t) VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT, (uint32_t) VK_COMPONENT_TYPE_FLOAT32_KHR);
        cm.coopmatI8I32Row16 = caps.hasCoopmatShape(16, 16, 16, (uint32_t) VK_COMPONENT_TYPE_SINT8_KHR, (uint32_t) VK_COMPONENT_TYPE_SINT32_KHR);
        cm.subgroupWidth     = caps.subgroupSize;
        cm.widthPinnable = caps.subgroupSizeControl && caps.requiredSubgroupSizeCompute && caps.minSubgroupSize <= caps.subgroupSize && caps.subgroupSize <= caps.maxSubgroupSize;
        cm.vulkanMemoryModel = caps.vulkanMemoryModel;
        cm.selfCheckPassed   = true; // provisionally; the caller runs the on-device check after its rule matches
        return cm;
    }

    bool coopmatGemmSelfCheckPassed(VkOpEnv &env) {
        vk::VulkanContext &ctx = *env.ctx;
        return memoizedVerdict(ctx.device(), "coopmat_gemm", [&] {
            const uint32_t width = coopmatSubgroupWidth(fillCoopmatGemmCaps(ctx.caps()));
            if (!env.runner || width == 0u)
            {
                return false;
            }
            std::vector<fp16_t> hostA((size_t) kCheckM * kCheckK), hostB((size_t) kCheckK * kCheckN);
            std::vector<fp16_t> expected((size_t) kCheckM * kCheckN);
            for (int row = 0; row < kCheckM; ++row)
            {
                for (int k = 0; k < kCheckK; ++k)
                {
                    hostA[(size_t) row * kCheckK + k] = floatToHalf(checkA(row, k));
                }
            }
            for (int k = 0; k < kCheckK; ++k)
            {
                for (int col = 0; col < kCheckN; ++col)
                {
                    hostB[(size_t) k * kCheckN + col] = floatToHalf(checkB(k, col));
                }
            }
            for (int row = 0; row < kCheckM; ++row)
            {
                for (int col = 0; col < kCheckN; ++col)
                {
                    float acc = 0.f;
                    for (int k = 0; k < kCheckK; ++k)
                    {
                        acc += checkA(row, k) * checkB(k, col);
                    }
                    expected[(size_t) row * kCheckN + col] = floatToHalf(acc);
                }
            }

            vk::Buffer bufA(ctx, hostA.size() * sizeof(fp16_t));
            vk::Buffer bufB(ctx, hostB.size() * sizeof(fp16_t));
            vk::Buffer bufD(ctx, expected.size() * sizeof(fp16_t), vk::MemPref::kReadback);
            bufA.upload(hostA.data(), hostA.size() * sizeof(fp16_t));
            bufB.upload(hostB.data(), hostB.size() * sizeof(fp16_t));

            struct {
                int M, N, K;
            } pc {kCheckM, kCheckN, kCheckK};
            auto pipe = env.pipeline("coopmat_gemm", 3, sizeof(pc), {width}, /*requiredSubgroupSize=*/width);
            env.runner->oneShot([&](VkCommandBuffer cmd) {
                // Grid matches the op dispatch: one workgroup per 32x32 output tile.
                pipe->dispatch(cmd, {bufA.handle(), bufB.handle(), bufD.handle()}, &pc, sizeof(pc), (uint32_t) (kCheckN / 32), (uint32_t) (kCheckM / 32));
            });

            std::vector<fp16_t> got(expected.size());
            bufD.download(got.data(), got.size() * sizeof(fp16_t));
            const bool passed = std::memcmp(got.data(), expected.data(), expected.size() * sizeof(fp16_t)) == 0;
            if (!passed)
            {
                VKNN_WARN << "coopmat_gemm self-check mismatch: cooperative-matrix path disabled for this device";
            } else
            {
                VKNN_INFO << "coopmat_gemm self-check passed (asymmetric " << kCheckM << "x" << kCheckN << "x" << kCheckK << " exact match)";
            }
            return passed;
        });
    }

    bool coopmatConvGemmSelfCheckPassed(VkOpEnv &env) {
        vk::VulkanContext &ctx = *env.ctx;
        return memoizedVerdict(ctx.device(), "conv_gemm_cm", [&] {
            const uint32_t width = coopmatSubgroupWidth(fillCoopmatGemmCaps(ctx.caps()));
            if (!env.runner || width == 0u)
            {
                return false;
            }
            constexpr int C = kConvCheckC, H = kConvCheckHW, W = kConvCheckHW;
            constexpr int Cout = kConvCheckCout, KH = kConvCheckKHW, KW = kConvCheckKHW, P = kConvCheckPad;
            constexpr int OH = H, OW = W; // stride 1, pad 1
            constexpr int Cinb = (C + 3) / 4, Coutb = (Cout + 3) / 4;
            constexpr int K = C * KH * KW;

            // NC4HW4 input, [K][Cout] weight panel (conv_gemm's channel-fastest k order), bias.
            std::vector<fp16_t> src((size_t) Cinb * H * W * 4, floatToHalf(0.f));
            for (int ic = 0; ic < C; ++ic)
            {
                for (int iy = 0; iy < H; ++iy)
                {
                    for (int ix = 0; ix < W; ++ix)
                    {
                        const float v                                                        = checkA(iy * W + ix, ic);
                        src[(size_t) (((ic / 4) * H + iy) * W + ix) * 4 + (size_t) (ic % 4)] = floatToHalf(v);
                    }
                }
            }
            std::vector<fp16_t> wt((size_t) K * Cout);
            for (int k = 0; k < K; ++k)
            {
                for (int oc = 0; oc < Cout; ++oc)
                {
                    wt[(size_t) k * Cout + oc] = floatToHalf(checkB(k, oc));
                }
            }
            std::vector<fp16_t> bias((size_t) Coutb * 4, floatToHalf(0.f));
            for (int oc = 0; oc < Cout; ++oc)
            {
                bias[oc] = floatToHalf((float) (oc % 3 - 1));
            }

            // Host oracle: fp32-accumulated conv over the same operands, pad lanes zero.
            std::vector<fp16_t> expected((size_t) Coutb * OH * OW * 4, floatToHalf(0.f));
            for (int oc = 0; oc < Cout; ++oc)
            {
                for (int oy = 0; oy < OH; ++oy)
                {
                    for (int ox = 0; ox < OW; ++ox)
                    {
                        float acc = (float) (oc % 3 - 1);
                        for (int ic = 0; ic < C; ++ic)
                        {
                            for (int ky = 0; ky < KH; ++ky)
                            {
                                for (int kx = 0; kx < KW; ++kx)
                                {
                                    const int iy = oy - P + ky, ix = ox - P + kx;
                                    if (iy < 0 || iy >= H || ix < 0 || ix >= W)
                                    {
                                        continue;
                                    }
                                    acc += checkA(iy * W + ix, ic) * checkB((ky * KW + kx) * C + ic, oc);
                                }
                            }
                        }
                        expected[(size_t) (((oc / 4) * OH + oy) * OW + ox) * 4 + (size_t) (oc % 4)] = floatToHalf(acc);
                    }
                }
            }

            vk::Buffer bufS(ctx, src.size() * sizeof(fp16_t));
            vk::Buffer bufW(ctx, wt.size() * sizeof(fp16_t));
            vk::Buffer bufB(ctx, bias.size() * sizeof(fp16_t));
            vk::Buffer bufD(ctx, expected.size() * sizeof(fp16_t), vk::MemPref::kReadback);
            bufS.upload(src.data(), src.size() * sizeof(fp16_t));
            bufW.upload(wt.data(), wt.size() * sizeof(fp16_t));
            bufB.upload(bias.data(), bias.size() * sizeof(fp16_t));

            struct {
                int   C, H, W, Cout, OH, OW, KH, KW, SH, SW, PT, PL, DH, DW;
                int   act, hasBias;
                float actLo, actHi;
                int   coutP;
            } pc {C, H, W, Cout, OH, OW, KH, KW, 1, 1, P, P, 1, 1, 0, 1, 0.f, 0.f, Cout};
            auto pipe = env.pipeline("conv_gemm_cm", 4, sizeof(pc), {width}, /*requiredSubgroupSize=*/width);
            env.runner->oneShot([&](VkCommandBuffer cmd) {
                // Grid matches the op dispatch: (ceil(Cout/TN), ceil(OH*OW/TM), N).
                pipe->dispatch(cmd, {bufS.handle(), bufW.handle(), bufB.handle(), bufD.handle()}, &pc, sizeof(pc), (uint32_t) ((Cout + kCoopmatTileN - 1) / kCoopmatTileN), (uint32_t) ((OH * OW + kCoopmatTileM - 1) / kCoopmatTileM));
            });

            std::vector<fp16_t> got(expected.size());
            bufD.download(got.data(), got.size() * sizeof(fp16_t));
            const bool passed = std::memcmp(got.data(), expected.data(), expected.size() * sizeof(fp16_t)) == 0;
            if (!passed)
            {
                VKNN_WARN << "conv_gemm_cm self-check mismatch: cooperative-matrix conv path disabled for this device";
            } else
            {
                VKNN_INFO << "conv_gemm_cm self-check passed (ragged " << OH * OW << "x" << Cout << "x" << K << " exact match)";
            }
            return passed;
        });
    }

} // namespace vknn
