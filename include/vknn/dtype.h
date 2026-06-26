// Tensor element types. Kept as an enum so int8 can be bolted on later without churn.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace vknn {

enum class DType : uint8_t {
  kFloat32 = 0,
  kFloat16 = 1,
  kInt32 = 2,
  kInt8 = 3,  // reserved for the int8 stretch goal
  kUInt8 = 4,
  kInt64 = 5,
};

inline size_t dtypeSize(DType d) {
  switch (d) {
    case DType::kFloat32:
      return 4;
    case DType::kFloat16:
      return 2;
    case DType::kInt32:
      return 4;
    case DType::kInt8:
      return 1;
    case DType::kUInt8:
      return 1;
    case DType::kInt64:
      return 8;
  }
  return 0;
}

inline const char* dtypeStr(DType d) {
  switch (d) {
    case DType::kFloat32:
      return "f32";
    case DType::kFloat16:
      return "f16";
    case DType::kInt32:
      return "i32";
    case DType::kInt8:
      return "i8";
    case DType::kUInt8:
      return "u8";
    case DType::kInt64:
      return "i64";
  }
  return "?";
}

// ---- Minimal IEEE-754 half <-> float conversion (host-side; no hardware dep) ----
// Used for packing weights / comparing fp16 outputs on the host.
using fp16_t = uint16_t;

inline float halfToFloat(fp16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      // subnormal
      exp = 127 - 15 + 1;
      while ((mant & 0x400) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FF;
      f = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    f = sign | 0x7F800000 | (mant << 13);  // inf/nan
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

inline fp16_t floatToHalf(float v) {
  uint32_t f;
  std::memcpy(&f, &v, 4);
  uint32_t sign = (f >> 16) & 0x8000;
  int32_t exp = (int32_t)((f >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = f & 0x7FFFFF;
  if (((f >> 23) & 0xFF) == 0xFF) {  // inf/nan
    return (fp16_t)(sign | 0x7C00 | (mant ? 0x200 : 0));
  }
  if (exp >= 0x1F)
    return (fp16_t)(sign | 0x7C00);  // overflow -> inf
  if (exp <= 0) {
    if (exp < -10)
      return (fp16_t)sign;  // underflow -> 0
    mant |= 0x800000;
    uint32_t shift = (uint32_t)(14 - exp);
    uint32_t half = (mant >> shift);
    // round to nearest even
    if ((mant >> (shift - 1)) & 1)
      half += 1;
    return (fp16_t)(sign | half);
  }
  fp16_t out = (fp16_t)(sign | (exp << 10) | (mant >> 13));
  if (mant & 0x1000)
    out += 1;  // round
  return out;
}

}  // namespace vknn
