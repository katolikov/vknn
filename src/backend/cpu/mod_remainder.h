// Scalar remainder rules of ONNX Mod, shared by the CPU Mod kernel and its host tests. The Vulkan
// kernel (shaders/mod.comp) computes the same values bit for bit: its exact fmod reproduces
// std::fmod, and every NaN result on both backends is the one canonical quiet NaN below.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace vknn { namespace cpu {

    /// ONNX Mod `fmod` attribute value (and its default) selecting the remainder whose sign follows
    /// the divisor: the integer semantics of the spec, Python's `%`.
    inline constexpr int64_t kModFloorRemainder = 0;
    /// ONNX Mod `fmod` attribute value selecting the C remainder whose sign follows the dividend
    /// (std::fmod, C++ integer `%`).
    inline constexpr int64_t kModTruncRemainder = 1;

    /// Bit pattern of the positive quiet NaN every NaN Mod result carries, on the CPU and in
    /// shaders/mod.comp (kQuietNaNBits), so the two backends agree byte for byte even on NaN.
    inline constexpr uint32_t kModQuietNaNBits = 0x7FC00000u;

    /// 2^63 in fp32 (exactly representable): the first fp32 magnitude outside the int64 range.
    inline constexpr float kModInt64RangeEndFp32 = 9223372036854775808.0f;

    /// The canonical quiet NaN (kModQuietNaNBits) as a float.
    inline float modQuietNaN() noexcept {
        const uint32_t bits = kModQuietNaNBits;
        float          value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    /// An fp32-carried operand of the int64 Mod path, truncated toward zero like the Binary int64
    /// path. Values outside the int64 range saturate and NaN reads as 0, so the conversion is never
    /// undefined; every in-range value converts exactly like `(int64_t) value`.
    inline int64_t modOperandToInt64(float value) noexcept {
        if (std::isnan(value))
        {
            return 0;
        }
        if (value >= kModInt64RangeEndFp32)
        {
            return std::numeric_limits<int64_t>::max();
        }
        if (value <= -kModInt64RangeEndFp32)
        {
            return std::numeric_limits<int64_t>::min(); // -2^63 is INT64_MIN exactly
        }
        return (int64_t) value;
    }

    /// Integer Mod remainder. A zero divisor yields 0, and INT64_MIN % -1 (undefined in C++) yields
    /// its exact value 0. With `floorRemainder` a nonzero remainder whose sign differs from the
    /// divisor's moves by one divisor, so its sign follows the divisor; |r| < |divisor| with opposite
    /// signs keeps that addition inside the int64 range.
    inline int64_t modRemainderInt(int64_t dividend, int64_t divisor, bool floorRemainder) noexcept {
        if (divisor == 0)
        {
            return 0;
        }
        if (dividend == std::numeric_limits<int64_t>::min() && divisor == -1)
        {
            return 0;
        }
        int64_t remainder = dividend % divisor;
        if (floorRemainder && remainder != 0 && ((remainder < 0) != (divisor < 0)))
        {
            remainder += divisor;
        }
        return remainder;
    }

    /// Floating-point Mod remainder.
    ///  - C remainder (`floorRemainder` false): std::fmod, exact in IEEE-754. NaN when the dividend
    ///    is infinite, the divisor is zero, or either operand is NaN; the dividend itself when the
    ///    divisor is infinite and the dividend finite; the sign of the dividend otherwise (-0 kept).
    ///  - Floor remainder (`floorRemainder` true): a zero divisor yields +0 (the integer semantics
    ///    of fmod 0, matching modRemainderInt); otherwise std::fmod, and a nonzero remainder whose
    ///    sign differs from the divisor's gains one divisor (an infinite divisor then yields it).
    /// Every NaN result is the canonical modQuietNaN().
    inline float modRemainderFloat(float dividend, float divisor, bool floorRemainder) noexcept {
        if (floorRemainder && divisor == 0.0f)
        {
            return 0.0f;
        }
        float remainder = std::fmod(dividend, divisor);
        if (std::isnan(remainder))
        {
            return modQuietNaN();
        }
        if (floorRemainder && remainder != 0.0f && ((remainder < 0.0f) != (divisor < 0.0f)))
        {
            remainder += divisor;
        }
        return remainder;
    }

}} // namespace vknn::cpu
