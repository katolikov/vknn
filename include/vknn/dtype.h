// Tensor element types.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace vknn {

    enum class DType : uint8_t {
        Float32 = 0,
        Float16 = 1,
        Int32   = 2,
        Int8    = 3,
        UInt8   = 4,
        Int64   = 5,
    };

    inline size_t dtypeSize(DType d) {
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

    inline const char *dtypeStr(DType d) {
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
    using fp16_t = uint16_t;

    inline float halfToFloat(fp16_t h) {
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
    inline void halfToFloatBulk(const fp16_t *src, float *dst, int64_t n) {
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
    inline void halfToFloatBulk(const fp16_t *src, float *dst, int64_t n) {
        for (int64_t i = 0; i < n; ++i)
        {
            dst[i] = halfToFloat(src[i]);
        }
    }
#endif

    inline fp16_t floatToHalf(float v) {
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
            // round to nearest even
            if ((mant >> (shift - 1)) & 1)
            {
                half += 1;
            }
            return (fp16_t) (sign | half);
        }
        fp16_t out = (fp16_t) (sign | (exp << 10) | (mant >> 13));
        if (mant & 0x1000)
        {
            out += 1; // round
        }
        return out;
    }

} // namespace vknn
