// Exact int64 arithmetic of the CPU Binary and Add int64 paths. Every function is defined for every
// operand value: sums, differences and products wrap modulo 2^64 (two's complement), division guards
// its two trapping cases, the integer power wraps like repeated multiplication, and an fp32-carried
// operand converts to int64 without an out-of-range float-to-integer conversion.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace vknn { namespace cpu {

    /// 2^63 in fp32 (exactly representable): the first fp32 magnitude outside the int64 range.
    inline constexpr float kInt64RangeEndFp32 = 9223372036854775808.0f;

    /// An fp32-carried operand of an int64 arithmetic path, truncated toward zero. NaN reads as 0 and a
    /// value outside the int64 range saturates to INT64_MIN / INT64_MAX, so the conversion is never
    /// undefined; every in-range value converts exactly like `(int64_t) value`.
    inline int64_t int64FromFp32Operand(float value) noexcept {
        if (std::isnan(value))
        {
            return 0;
        }
        if (value >= kInt64RangeEndFp32)
        {
            return std::numeric_limits<int64_t>::max();
        }
        if (value <= -kInt64RangeEndFp32)
        {
            return std::numeric_limits<int64_t>::min(); // -2^63 is INT64_MIN exactly
        }
        return (int64_t) value;
    }

    /// The int64 whose two's-complement bit pattern is `bits` (a byte copy, defined for every pattern).
    inline int64_t int64FromWrappedBits(uint64_t bits) noexcept {
        int64_t value;
        std::memcpy(&value, &bits, sizeof value);
        return value;
    }

    /// a + b modulo 2^64. Unsigned arithmetic is modular, so an overflowing sum wraps instead of invoking
    /// signed-overflow undefined behavior.
    inline int64_t wrappingAddInt64(int64_t a, int64_t b) noexcept {
        return int64FromWrappedBits((uint64_t) a + (uint64_t) b);
    }

    /// a - b modulo 2^64.
    inline int64_t wrappingSubInt64(int64_t a, int64_t b) noexcept {
        return int64FromWrappedBits((uint64_t) a - (uint64_t) b);
    }

    /// a * b modulo 2^64. The low 64 bits of a product do not depend on signedness, so the unsigned
    /// product carries the two's-complement result.
    inline int64_t wrappingMulInt64(int64_t a, int64_t b) noexcept {
        return int64FromWrappedBits((uint64_t) a * (uint64_t) b);
    }

    /// Integer division truncating toward zero (C++ `/`, ONNX integer Div). A zero divisor yields 0
    /// instead of a hardware trap. INT64_MIN / -1, whose true quotient 2^63 does not fit in int64, wraps
    /// to INT64_MIN: a divisor of -1 is a wrapping negation and never reaches the hardware division.
    inline int64_t divideInt64(int64_t dividend, int64_t divisor) noexcept {
        if (divisor == 0)
        {
            return 0;
        }
        if (divisor == -1)
        {
            return wrappingSubInt64(0, dividend);
        }
        return dividend / divisor;
    }

    /// Integer power. A non-negative exponent computes base^exponent by binary exponentiation with every
    /// product wrapping modulo 2^64, so the result equals repeated wrapping multiplication (0^0 == 1,
    /// 2^63 == INT64_MIN, 2^64 == 0). A negative exponent yields 1 / base^|exponent| truncated toward
    /// zero: 1 for base 1, +1 or -1 for base -1 by the exponent's parity, and 0 for every other base,
    /// base 0 included (a defined 0 in place of a division by zero).
    inline int64_t powInt64(int64_t base, int64_t exponent) noexcept {
        if (exponent < 0)
        {
            if (base == 1)
            {
                return 1;
            }
            if (base == -1)
            {
                return exponent % 2 != 0 ? -1 : 1;
            }
            return 0;
        }
        uint64_t result         = 1;
        uint64_t squaredBase    = (uint64_t) base;
        uint64_t remainingPower = (uint64_t) exponent; // one pass per significant exponent bit
        while (remainingPower != 0)
        {
            if ((remainingPower & 1u) != 0)
            {
                result *= squaredBase;
            }
            remainingPower >>= 1;
            squaredBase *= squaredBase;
        }
        return int64FromWrappedBits(result);
    }

}} // namespace vknn::cpu
