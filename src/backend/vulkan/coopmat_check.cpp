#include "coopmat_check.h"
#include "vk_op_env.h"
#include "vk_buffer.h"
#include "vk_command.h"
#include "vknn/dtype.h"
#include <map>
#include <mutex>
#include <vector>

namespace vknn {

    namespace {
        // Asymmetric probe dimensions: M != N != K, so a kernel that swaps rows for columns, reads
        // A column-major, or walks K with the wrong stride produces different bytes than the
        // oracle. Each dimension honors the kernel contract (M, N multiples of 32, K of 16).
        constexpr int kCheckM = 32;
        constexpr int kCheckN = 64;
        constexpr int kCheckK = 48;

        // Deterministic small-integer operands: every value, every fp32 partial sum and the fp16
        // result are exactly representable, so the device must match the host to the bit.
        inline float checkA(int row, int k) {
            return (float) ((row * 7 + k * 3) % 5 - 2); // in [-2, 2]
        }
        inline float checkB(int k, int col) {
            return (float) ((k * 5 + col) % 7 - 3); // in [-3, 3]
        }
    } // namespace

    bool coopmatGemmSelfCheckPassed(VkOpEnv &env) {
        static std::mutex                 verdictMutex;
        static std::map<VkDevice, bool>   verdictByDevice;

        vk::VulkanContext &ctx = *env.ctx;
        {
            std::lock_guard<std::mutex> lock(verdictMutex);
            auto                        it = verdictByDevice.find(ctx.device());
            if (it != verdictByDevice.end())
            {
                return it->second;
            }
        }
        bool passed = false;
        try
        {
            if (env.runner)
            {
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

                struct { int M, N, K; } pc {kCheckM, kCheckN, kCheckK};
                auto pipe = env.pipeline("coopmat_gemm", 3, sizeof(pc), {}, /*requiredSubgroupSize=*/32);
                env.runner->oneShot([&](VkCommandBuffer cmd) {
                    // Grid matches the op dispatch: one workgroup per 32x32 output tile.
                    pipe->dispatch(cmd, {bufA.handle(), bufB.handle(), bufD.handle()}, &pc, sizeof(pc), (uint32_t) (kCheckN / 32), (uint32_t) (kCheckM / 32));
                });

                std::vector<fp16_t> got(expected.size());
                bufD.download(got.data(), got.size() * sizeof(fp16_t));
                passed = std::memcmp(got.data(), expected.data(), expected.size() * sizeof(fp16_t)) == 0;
                if (!passed)
                {
                    VKNN_WARN << "coopmat_gemm self-check mismatch: cooperative-matrix path disabled for this device";
                }
                else
                {
                    VKNN_INFO << "coopmat_gemm self-check passed (asymmetric " << kCheckM << "x" << kCheckN << "x" << kCheckK << " exact match)";
                }
            }
        } catch (const std::exception &e)
        {
            VKNN_WARN << "coopmat_gemm self-check failed to run (" << e.what() << "); cooperative-matrix path disabled";
            passed = false;
        }
        std::lock_guard<std::mutex> lock(verdictMutex);
        verdictByDevice[ctx.device()] = passed;
        return passed;
    }

} // namespace vknn
