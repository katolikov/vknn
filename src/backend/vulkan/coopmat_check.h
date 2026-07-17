// One-time on-device validation of the cooperative-matrix GEMM kernel against a host reference.
#pragma once

namespace vknn {
    struct VkOpEnv;

    /// Runs an asymmetric exact-integer GEMM (M=16, N=32, K=48 - dimensions chosen so a transposed
    /// or mis-strided fragment mapping cannot produce the reference bytes) through the
    /// coopmat_gemm pipeline once per device and byte-compares the fp16 result against the host
    /// oracle. Operand values are small integers, so fp16 storage, fp32 accumulation and the fp16
    /// store are all exact and the comparison is equality, not a tolerance. The verdict is memoized
    /// per VkDevice: every MatMul instance consults it, only the first pays for the dispatch.
    /// Returns false (path disabled, SSBO kernels serve the node) when the device lacks the
    /// capability set, the environment has no runner, or the device result mismatches - a mismatch
    /// also logs a warning naming the kernel so a misbehaving driver is visible in the session log.
    bool coopmatGemmSelfCheckPassed(VkOpEnv &env);

} // namespace vknn
