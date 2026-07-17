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
    // VulkanCaps; the fields are AND-ed prerequisites for pinning a 32-wide subgroup coopmat
    // pipeline (see docs in include/vknn/hint.h under CoopmatGemm).
    struct CoopmatGemmCaps {
        bool coopmatFp16Fp32Row16 = false; // subgroup-scope 16x16x16 row, A/B fp16, C/Result fp32
        bool coopmatFp8Fp32Row16  = false; // same row with e4m3 operands
        bool coopmatI8I32Row16    = false; // same row with int8 operands, int32 accumulator
        bool wave32Pinnable       = false; // subgroupSizeControl + COMPUTE stage + 32 in [min,max]
        bool vulkanMemoryModel    = false; // coopmat SPIR-V declares OpMemoryModel Logical Vulkan
        bool selfCheckPassed      = false; // one-time on-device asymmetric GEMM matched the host exactly
    };

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
    inline CoopmatGemmKind coopmatGemmRoute(const CoopmatGemmCaps &caps, int hintValue, bool fp16Storage, bool denseRank2Batch1, bool hasBiasOrEpilogue, int64_t M, int64_t N, int64_t K,
                                            bool lowPrecisionWeightsAvailable) {
        const bool base = caps.wave32Pinnable && caps.vulkanMemoryModel && caps.selfCheckPassed && fp16Storage && denseRank2Batch1 && !hasBiasOrEpilogue && M >= 32 && N >= 32 && K >= 32 && M % 32 == 0 && N % 32 == 0 && K % 16 == 0;
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

} // namespace vknn
