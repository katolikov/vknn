// One-time on-device validation of the cooperative-matrix kernels against host references, plus
// the shared device-capability mirror every coopmat routing site consumes.
#pragma once
#include "core/lowp_gemm.h"

namespace vknn {
    struct VkOpEnv;
    namespace vk {
        struct VulkanCaps;
    }

    /// Fills the plain-value CoopmatGemmCaps mirror from the device capability set — the one
    /// shared translation between VulkanCaps and the host-testable routing rules
    /// (coopmatGemmRoute / coopmatConvRoute). selfCheckPassed is left true provisionally; the
    /// caller runs the kernel's on-device check only after its shape rule matches, so the probe
    /// dispatch is paid exactly when a node would actually route.
    CoopmatGemmCaps fillCoopmatGemmCaps(const vk::VulkanCaps &caps);

    /// Runs an asymmetric exact-integer GEMM (M=32, N=64, K=48 - dimensions chosen so a transposed
    /// or mis-strided fragment mapping cannot produce the reference bytes) through the
    /// coopmat_gemm pipeline once per device and byte-compares the fp16 result against the host
    /// oracle. Operand values are small integers, so fp16 storage, fp32 accumulation and the fp16
    /// store are all exact and the comparison is equality, not a tolerance. The verdict is memoized
    /// per (VkDevice, kernel): every MatMul instance consults it, only the first pays for the
    /// dispatch. Returns false (path disabled, SSBO kernels serve the node) when the device lacks
    /// the capability set, the environment has no runner, or the device result mismatches - a
    /// mismatch also logs a warning naming the kernel so a misbehaving driver is visible in the
    /// session log.
    bool coopmatGemmSelfCheckPassed(VkOpEnv &env);

    /// Same gate for the cooperative-matrix implicit-GEMM conv (conv_gemm_cm.comp): a small
    /// asymmetric integer conv with bias, ragged M/N/K tile edges and pad-lane coverage,
    /// byte-compared against a host NC4HW4 oracle. Memoized per (VkDevice, kernel) alongside the
    /// GEMM verdict; every Conv instance consults it, only the first pays for the dispatch.
    bool coopmatConvGemmSelfCheckPassed(VkOpEnv &env);

} // namespace vknn
