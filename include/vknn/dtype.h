// Tensor element types.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace vknn {

    /// Element type of a tensor. The underlying uint8_t values are stable and serialized into the
    /// compiled-model cache, so existing enumerators keep their numeric value; new types append.
    enum class DType : uint8_t {
        Float32 = 0, ///< 32-bit IEEE-754 single precision.
        Float16 = 1, ///< 16-bit IEEE-754 half precision (see halfToFloat()/floatToHalf()).
        Int32   = 2, ///< 32-bit signed integer.
        Int8    = 3, ///< 8-bit signed integer.
        UInt8   = 4, ///< 8-bit unsigned integer (e.g. image graph-inputs).
        Int64   = 5, ///< 64-bit signed integer (shape/index tensors).
    };

    /// Size in bytes of one element of type `d`; 0 for an unrecognized value.
    inline size_t dtypeSize(DType d) noexcept {
        switch (d)
        {
            case DType::Float32:
                return 4;
            case DType::Float16:
                return 2;
            case DType::Int32:
                return 4;
            case DType::Int8:
                return 1;
            case DType::UInt8:
                return 1;
            case DType::Int64:
                return 8;
        }
        return 0;
    }

    /// Short lowercase mnemonic for `d` (e.g. "f32", "i8"); "?" for an unrecognized value.
    /// The returned pointer is a string literal and remains valid for the program lifetime.
    inline const char *dtypeStr(DType d) noexcept {
        switch (d)
        {
            case DType::Float32:
                return "f32";
            case DType::Float16:
                return "f16";
            case DType::Int32:
                return "i32";
            case DType::Int8:
                return "i8";
            case DType::UInt8:
                return "u8";
            case DType::Int64:
                return "i64";
        }
        return "?";
    }

    // ---- Minimal IEEE-754 half <-> float conversion (host-side; no hardware dep) ----
    // Used for packing weights / comparing fp16 outputs on the host.

    /// Storage type of a half-precision value: the raw 16-bit IEEE-754 bit pattern, not an arithmetic
    /// type. Convert with halfToFloat()/floatToHalf() before doing math.
    using fp16_t = uint16_t;

    /// Widen a half-precision bit pattern to float. Handles subnormals, and preserves inf/NaN. Exact:
    /// every fp16 value is representable in fp32.
    inline float halfToFloat(fp16_t h) noexcept {
        uint32_t sign = (uint32_t) (h & 0x8000) << 16;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t f;
        if (exp == 0)
        {
            if (mant == 0)
            {
                f = sign;
            } else
            {
                // subnormal
                exp = 127 - 15 + 1;
                while ((mant & 0x400) == 0)
                {
                    mant <<= 1;
                    --exp;
                }
                mant &= 0x3FF;
                f = sign | (exp << 23) | (mant << 13);
            }
        } else if (exp == 0x1F)
        {
            f = sign | 0x7F800000 | (mant << 13); // inf/nan
        } else
        {
            f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        float out;
        std::memcpy(&out, &f, 4);
        return out;
    }

// Bulk fp16 -> fp32 for contiguous buffers (the flat-output download path, e.g. YOLO's 705K-element
// detection tensor). AArch64 NEON has a hardware half->single convert (vcvt_f32_f16, baseline
// ARMv8 - no fp16-arithmetic feature needed), 4 lanes/instr, ~6x the scalar bit-twiddle. Falls back
// to the scalar path on other targets / the tail.
#if defined(__aarch64__)
#include <arm_neon.h>
    /// Widen `n` contiguous half-precision values from `src` into `dst`. Equivalent to calling
    /// halfToFloat() per element; `src` and `dst` must not overlap.
    inline void halfToFloatBulk(const fp16_t *src, float *dst, int64_t n) noexcept {
        int64_t i = 0;
        for (; i + 4 <= n; i += 4)
        {
            uint16x4_t u = vld1_u16(src + i);
            vst1q_f32(dst + i, vcvt_f32_f16(vreinterpret_f16_u16(u)));
        }
        for (; i < n; ++i)
        {
            dst[i] = halfToFloat(src[i]);
        }
    }
#else
    inline void halfToFloatBulk(const fp16_t *src, float *dst, int64_t n) noexcept {
        for (int64_t i = 0; i < n; ++i)
        {
            dst[i] = halfToFloat(src[i]);
        }
    }
#endif

    /// Narrow a float to a half-precision bit pattern. The top bit of the dropped mantissa alone decides
    /// the rounding, so an exact halfway value rounds away from zero rather than to even. Values at or
    /// beyond 2^16 saturate to infinity; values below 2^-25 flush to a signed zero and the range between
    /// yields an fp16 subnormal; infinities pass through; every NaN collapses to the quiet pattern
    /// `sign | 0x7E00`, dropping its payload.
    ///
    /// floatToHalfBulk() reproduces this rounding bit-for-bit over a contiguous range.
    inline fp16_t floatToHalf(float v) noexcept {
        uint32_t f;
        std::memcpy(&f, &v, 4);
        uint32_t sign = (f >> 16) & 0x8000;
        int32_t  exp  = (int32_t) ((f >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = f & 0x7FFFFF;
        if (((f >> 23) & 0xFF) == 0xFF)
        { // inf/nan
            return (fp16_t) (sign | 0x7C00 | (mant ? 0x200 : 0));
        }
        if (exp >= 0x1F)
        {
            return (fp16_t) (sign | 0x7C00); // overflow -> inf
        }
        if (exp <= 0)
        {
            if (exp < -10)
            {
                return (fp16_t) sign; // underflow -> 0
            }
            mant |= 0x800000;
            uint32_t shift = (uint32_t) (14 - exp);
            uint32_t half  = (mant >> shift);
            // round up on the dropped top bit alone (a halfway value rounds away from zero)
            if ((mant >> (shift - 1)) & 1)
            {
                half += 1;
            }
            return (fp16_t) (sign | half);
        }
        fp16_t out = (fp16_t) (sign | (exp << 10) | (mant >> 13));
        if (mant & 0x1000)
        {
            out += 1; // round up on the dropped top bit; a carry rolls into the exponent
        }
        return out;
    }

// Bulk fp32 -> fp16 for contiguous buffers (the graph-input upload of a flat device tensor, an
// fp16-declared graph output, weight staging). AArch64 evaluates floatToHalf's branchy bit-twiddle four
// lanes at a time with integer NEON: each branch becomes a lane mask and the results merge with vbsl.
// The hardware convert vcvt_f16_f32 is NOT used here - it rounds a halfway value to even, where
// floatToHalf rounds it away from zero, so it would not be bit-identical. Falls back to the scalar path
// on other targets and on the tail.
#if defined(__aarch64__)
    /// Narrow `n` contiguous floats at `src` into half-precision bit patterns at `dst`. Bit-identical to
    /// floatToHalf() per element, tie rule, NaN collapse and saturation included; `src` and `dst` must not
    /// overlap.
    inline void floatToHalfBulk(const float *src, fp16_t *dst, int64_t n) noexcept {
        const uint32x4_t kOne = vdupq_n_u32(1);
        const uint32x4_t kInf = vdupq_n_u32(0x7C00);
        int64_t          i    = 0;
        for (; i + 4 <= n; i += 4)
        {
            uint32x4_t f    = vld1q_u32(reinterpret_cast<const uint32_t *>(src + i));
            uint32x4_t sign = vandq_u32(vshrq_n_u32(f, 16), vdupq_n_u32(0x8000));
            uint32x4_t e    = vandq_u32(vshrq_n_u32(f, 23), vdupq_n_u32(0xFF)); // biased fp32 exponent
            uint32x4_t mant = vandq_u32(f, vdupq_n_u32(0x7FFFFF));

            // Normal half: rebias the exponent (127 -> 15), drop 13 mantissa bits, round up on bit 12.
            // A lane outside the normal exponent range wraps here and is replaced by a select below.
            uint32x4_t nrm = vorrq_u32(sign, vorrq_u32(vshlq_n_u32(vsubq_u32(e, vdupq_n_u32(112)), 10), vshrq_n_u32(mant, 13)));
            nrm            = vaddq_u32(nrm, vandq_u32(vshrq_n_u32(mant, 12), kOne));

            // Subnormal half: restore the hidden bit and shift the significand down by 126 - e (14..24
            // across the subnormal exponent range), rounding up on the last dropped bit. vshlq_u32 takes a
            // signed per-lane count, so negating it turns the left shift into the right shift, and a count
            // of 32 or more -- every lane outside the range -- yields zero rather than undefined behavior.
            int32x4_t  sh    = vsubq_s32(vdupq_n_s32(126), vreinterpretq_s32_u32(e));
            uint32x4_t m     = vorrq_u32(mant, vdupq_n_u32(0x800000));
            uint32x4_t half  = vshlq_u32(m, vnegq_s32(sh));
            uint32x4_t guard = vandq_u32(vshlq_u32(m, vnegq_s32(vsubq_s32(sh, vdupq_n_s32(1)))), kOne);
            uint32x4_t sub   = vorrq_u32(sign, vaddq_u32(half, guard));

            // The exponent ranges, in the priority floatToHalf applies them: NaN/inf (e == 255) wins over
            // overflow (e >= 143, i.e. |v| >= 2^16), which is disjoint from the subnormal-or-zero range
            // (e <= 112); the flush-to-zero range (e < 102, i.e. |v| < 2^-25) sits inside the latter.
            uint32x4_t nanv = vorrq_u32(vorrq_u32(sign, kInf), vandq_u32(vdupq_n_u32(0x200), vtstq_u32(mant, mant)));
            uint32x4_t r    = nrm;
            r               = vbslq_u32(vcleq_u32(e, vdupq_n_u32(112)), sub, r);
            r               = vbslq_u32(vcltq_u32(e, vdupq_n_u32(102)), sign, r);
            r               = vbslq_u32(vcgeq_u32(e, vdupq_n_u32(143)), vorrq_u32(sign, kInf), r);
            r               = vbslq_u32(vceqq_u32(e, vdupq_n_u32(0xFF)), nanv, r);
            vst1_u16(dst + i, vmovn_u32(r));
        }
        for (; i < n; ++i)
        {
            dst[i] = floatToHalf(src[i]);
        }
    }
#else
    inline void floatToHalfBulk(const float *src, fp16_t *dst, int64_t n) noexcept {
        for (int64_t i = 0; i < n; ++i)
        {
            dst[i] = floatToHalf(src[i]);
        }
    }
#endif

} // namespace vknn
