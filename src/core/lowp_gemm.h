// Low-precision GEMM helpers shared by the Vulkan cooperative-matrix path and the host tests:
// fp8 (e4m3) and int8 per-tensor symmetric codecs for host-side weight quantization, plus the
// deterministic shape rule that routes a MatMul onto the cooperative-matrix kernels. Everything
// here is pure host code (no Vulkan types), so the rules and codecs unit-test on the CPU-only
// host build; the device caps feed in through the plain CoopmatGemmCaps mirror.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace vknn {

    // e4m3 (1 sign, 4 exponent bits bias 7, 3 mantissa bits): finite range +-448, no infinities,
    // 0x7F/0xFF are NaN. Encoding rounds to nearest-even and saturates overflow to +-448, matching
    // the GPU-side float -> e4m3 conversion for in-range values.
    inline uint8_t encodeFp8E4M3(float value) {
        if (std::isnan(value))
        {
            return 0x7F;
        }
        const uint8_t sign = value < 0.f ? 0x80 : 0x00;
        float         mag  = std::fabs(value);
        if (mag > 448.f)
        {
            mag = 448.f; // saturate: e4m3 has no infinity
        }
        if (mag < std::ldexp(1.f, -9)) // below half the smallest subnormal step (2^-6 / 8 / 2)
        {
            return sign; // rounds to +-0
        }
        int exponent = 0;
        std::frexp(mag, &exponent); // mag = f * 2^exponent, f in [0.5, 1)
        exponent -= 1;              // normalized exponent so mag = 1.f * 2^exponent
        int biased = exponent + 7;
        if (biased < 1)
        {
            // Subnormal: value = m/8 * 2^-6 with m in [0,7]; round-to-nearest-even on m.
            const float stepUnits = mag * std::ldexp(1.f, 9); // mag / 2^-9 = mag in eighth-steps of 2^-6
            int         m         = (int) std::nearbyint(stepUnits);
            if (m > 7)
            {
                return (uint8_t) (sign | 0x08); // rounded up into the first normal encoding
            }
            return (uint8_t) (sign | m);
        }
        // Normal: mantissa in eighths, round-to-nearest-even, carry may bump the exponent.
        float frac = mag * std::ldexp(1.f, -exponent) - 1.f; // [0,1)
        int   m    = (int) std::nearbyint(frac * 8.f);
        if (m == 8)
        {
            m = 0;
            ++biased;
        }
        if (biased > 15 || (biased == 15 && m > 6))
        {
            return (uint8_t) (sign | 0x7E); // saturate to +-448 (0 1111 110)
        }
        return (uint8_t) (sign | (biased << 3) | m);
    }

    inline float decodeFp8E4M3(uint8_t bits) {
        const float sign     = (bits & 0x80) ? -1.f : 1.f;
        const int   exponent = (bits >> 3) & 0xF;
        const int   mantissa = bits & 0x7;
        if (exponent == 0xF && mantissa == 0x7)
        {
            return std::nanf("");
        }
        if (exponent == 0)
        {
            return sign * std::ldexp((float) mantissa / 8.f, -6);
        }
        return sign * std::ldexp(1.f + (float) mantissa / 8.f, exponent - 7);
    }

    // Per-tensor symmetric int8: scale = absmax / 127, x -> round(x / scale) clamped to [-127, 127].
    inline int8_t encodeInt8Symmetric(float value, float scale) {
        if (scale <= 0.f)
        {
            return 0;
        }
        const float q = std::nearbyint(value / scale);
        return (int8_t) std::max(-127.f, std::min(127.f, q));
    }

    // Device capabilities the coopmat routing rule consumes, mirrored as plain values so the rule
    // is testable on the host build (which compiles no Vulkan). The backend fills this from
    // VulkanCaps; the fields are AND-ed prerequisites for pinning a one-subgroup-per-workgroup
    // coopmat pipeline (see docs in include/vknn/hint.h under CoopmatGemm). RDNA-class devices run
    // compute at wave32 or wave64: the device's reported native width is the one the kernels pin
    // (the workgroup width rides a specialization constant), never a preference ladder over the
    // other width - see coopmatSubgroupWidth.
    struct CoopmatGemmCaps {
        bool     coopmatFp16Fp32Row16 = false; // subgroup-scope 16x16x16 row, A/B fp16, C/Result fp32
        bool     coopmatFp8Fp32Row16  = false; // same row with e4m3 operands
        bool     coopmatI8I32Row16    = false; // same row with int8 operands, int32 accumulator
        uint32_t subgroupWidth        = 0;     // the device's native compute subgroup width (VkPhysicalDeviceSubgroupProperties)
        bool     widthPinnable        = false; // subgroupSizeControl + COMPUTE stage + subgroupWidth in [min,max]
        bool     vulkanMemoryModel    = false; // coopmat SPIR-V declares OpMemoryModel Logical Vulkan
        bool     selfCheckPassed      = false; // one-time on-device asymmetric GEMM matched the host exactly
    };

    // The pinned subgroup (= workgroup) width the coopmat kernels run at: the device's native
    // compute width, pinned explicitly so the fragment shapes never shift under a driver's
    // per-pipeline width heuristic. Only the two RDNA compute widths are served; anything else
    // (or a width the driver refuses to pin) returns 0 and the routing rules keep the SSBO
    // kernels. Every coopmat pipeline and self-check must create with this width, both as
    // requiredSubgroupSize and as the workgroup-size spec constant.
    constexpr uint32_t kCoopmatWave32 = 32;
    constexpr uint32_t kCoopmatWave64 = 64;
    inline uint32_t    coopmatSubgroupWidth(const CoopmatGemmCaps &caps) {
        const bool served = caps.subgroupWidth == kCoopmatWave32 || caps.subgroupWidth == kCoopmatWave64;
        return served && caps.widthPinnable ? caps.subgroupWidth : 0u;
    }

    enum class CoopmatGemmKind {
        None = 0, // SSBO kernels (device lacks the path, the hint disables it, or the shape is ineligible)
        Fp16 = 1, // fp16 operands, fp32 accumulator (Auto/On)
        Fp8  = 2, // e4m3 operands, fp32 accumulator (opt-in Mode::Fp8)
        Int8 = 3, // int8 operands, int32 accumulator (opt-in Mode::Int8Coop)
    };

    // Deterministic routing rule (never timing-raced: the coopmat kernels reorder the K reduction
    // relative to the SSBO kernels, so the choice must be a pure function of shape, hint and
    // device capability). hintValue is the raw Hint::CoopmatGemm int (Mode::Auto/On/Off/Fp8/
    // Int8Coop); lowPrecisionWeightsAvailable reports whether the B operand is a host-quantizable
    // initializer (the opt-in paths quantize weights at prepare time).
    inline CoopmatGemmKind coopmatGemmRoute(const CoopmatGemmCaps &caps, int hintValue, bool fp16Storage, bool denseRank2Batch1, bool hasBiasOrEpilogue, int64_t M, int64_t N, int64_t K, bool lowPrecisionWeightsAvailable) {
        const bool base = coopmatSubgroupWidth(caps) != 0u && caps.vulkanMemoryModel && caps.selfCheckPassed && fp16Storage && denseRank2Batch1 && !hasBiasOrEpilogue && M >= 32 && N >= 32 && K >= 32 && M % 32 == 0 && N % 32 == 0 && K % 16 == 0;
        if (!base || hintValue == 2) // Mode::Off
        {
            return CoopmatGemmKind::None;
        }
        if (hintValue == 3 && caps.coopmatFp8Fp32Row16 && lowPrecisionWeightsAvailable) // Mode::Fp8
        {
            return CoopmatGemmKind::Fp8;
        }
        if (hintValue == 4 && caps.coopmatI8I32Row16 && lowPrecisionWeightsAvailable) // Mode::Int8Coop
        {
            return CoopmatGemmKind::Int8;
        }
        // Auto/On (and an opt-in request the device cannot honor) take the fp32-accumulator path.
        return caps.coopmatFp16Fp32Row16 ? CoopmatGemmKind::Fp16 : CoopmatGemmKind::None;
    }

    // Host mirror of the coopmat_tile.glsl workgroup tile (COOP_TM/COOP_TN/COOP_TK): the staged
    // consumers dispatch one pinned subgroup (32- or 64-wide) per kCoopmatTileM x kCoopmatTileN
    // output tile and advance K by kCoopmatTileK per fragment step.
    constexpr int64_t kCoopmatTileM = 32;
    constexpr int64_t kCoopmatTileN = 32;
    constexpr int64_t kCoopmatTileK = 16;

    // Profitability floors for the cooperative-matrix conv route: below one full output tile on
    // either GEMM axis (or under one fragment depth of reduction) the matrix units cannot fill and
    // the SSBO kernels keep the shape. The staged kernel masks ragged edges itself, so these are
    // minimums, not alignment requirements.
    constexpr int64_t kCoopmatConvMinM = kCoopmatTileM;
    constexpr int64_t kCoopmatConvMinN = kCoopmatTileN;
    constexpr int64_t kCoopmatConvMinK = kCoopmatTileK;

    // Deterministic routing rule for the cooperative-matrix implicit-GEMM conv (conv_gemm_cm.comp):
    // like coopmatGemmRoute, a pure function of device capability, the Hint::CoopmatGemm value and
    // the GEMM view of the conv shape (M = OH*OW, N = Cout, K = Cin*KH*KW), never a timing race —
    // the matrix-unit K order differs from every SSBO conv kernel. structuralOk carries the
    // conv-side prerequisites the shape numbers cannot express (fp16 storage, group == 1, not
    // depthwise, no fused residual, no pointwise-epilogue chain). The opt-in low-precision Modes
    // (Fp8/Int8Coop) are MatMul-weight modes; the conv route treats them as Auto and serves fp16.
    inline bool coopmatConvRoute(const CoopmatGemmCaps &caps, int hintValue, bool structuralOk, int64_t M, int64_t N, int64_t K) {
        const bool base = caps.coopmatFp16Fp32Row16 && coopmatSubgroupWidth(caps) != 0u && caps.vulkanMemoryModel && caps.selfCheckPassed && structuralOk;
        if (!base || hintValue == 2) // Mode::Off
        {
            return false;
        }
        return M >= kCoopmatConvMinM && N >= kCoopmatConvMinN && K >= kCoopmatConvMinK;
    }

} // namespace vknn
