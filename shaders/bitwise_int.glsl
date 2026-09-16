// Integer arithmetic on fp32 values shared by the bitwise kernels (bitwise.comp, bitwise_not.comp,
// bitshift.comp). Each function computes the integer the CPU oracle (src/backend/cpu/bitwise_int.h)
// computes in int64, using only operations that are exact on integer-valued fp32 values: truncation,
// floor, scaling by a power of two, and one subtraction per function whose exact result is rounded
// once. Results are exact while operands and results stay within +-2^24 (the fp32 integer range). No
// conversion is undefined: a float reaches an int conversion already clamped into range, and a uint
// shift count stays below 32. tests/test_bitwise_ops.cpp carries a statement-for-statement C++
// transcription of these functions, swept against the CPU oracle.
#ifndef VKNN_BITWISE_INT_GLSL
#define VKNN_BITWISE_INT_GLSL

const int   kInt64Bits             = 64;                     // widest integer width; its wrap is signed
const int   kShiftWordBits         = 32;                     // a uint shift count stays below this
const int   kFloatExactIntegerBits = 24;                     // every integer of this many bits is exact in fp32
const float kTwoPow32              = 4294967296.0;           // 2^32 (exact in fp32)
const float kTwoPow63              = 9223372036854775808.0;  // 2^63 (exact): the int64 magnitude bound
const float kTwoPow64              = 18446744073709551616.0; // 2^64 (exact): the int64 modulus
const float kInt32MinFloat         = -2147483648.0;          // INT32_MIN (exact)
const float kInt32MaxFloat         = 2147483520.0;           // largest fp32 value below 2^31

// Integer value of an operand lane: NaN reads 0, the value saturates to [-2^63, 2^63] and truncates
// toward zero. A zero result is +0.0 (trunc keeps the sign of -0.5), the oracle's integer zero, so no
// result downstream carries a negative zero.
float integerOperand(float value) {
  if (isnan(value)) return 0.0;
  precise float integer = trunc(clamp(value, -kTwoPow63, kTwoPow63));
  return integer == 0.0 ? 0.0 : integer;
}

// Integer value of an operand lane as int32: NaN reads 0, the value saturates to the int32 range and
// truncates toward zero, so the int conversion is always defined.
int int32Operand(float value) {
  if (isnan(value)) return 0;
  return int(trunc(clamp(value, kInt32MinFloat, kInt32MaxFloat)));
}

// 2^exponent for exponent in [0, 64], built from exact factors: whole 2^32 words, then a uint shift
// below the word size.
float powerOfTwo(int exponent) {
  precise float value = 1.0;
  int remaining = exponent;
  if (remaining >= kShiftWordBits) { value *= kTwoPow32; remaining -= kShiftWordBits; }
  if (remaining >= kShiftWordBits) { value *= kTwoPow32; remaining -= kShiftWordBits; }
  precise float result = value * float(1u << uint(remaining));
  return result;
}

// Residue of the integer x modulo 2^bits for bits in [1, 63], in [0, 2^bits): the unsigned value of x's
// low `bits` bits. floor(x / 2^bits) and its product with 2^bits are exact, so the one subtraction
// rounds the exact residue once (and returns it exactly whenever it is representable).
float residueModPowerOfTwo(float x, int bits) {
  precise float modulus = powerOfTwo(bits);
  precise float residue = x - floor(x / modulus) * modulus;
  return residue;
}

// x's two's-complement value at `bits` width in the form the CPU oracle's int64 result takes: the
// unsigned residue [0, 2^bits) for bits < 64, the signed range [-2^63, 2^63) for bits == 64.
float wrapToWidth(float x, int bits) {
  if (bits < kInt64Bits) return residueModPowerOfTwo(x, bits);
  if (x >= -kTwoPow63 && x < kTwoPow63) return x;
  precise float unsignedResidue = x - floor(x / kTwoPow64) * kTwoPow64;
  precise float signedResidue = unsignedResidue >= kTwoPow63 ? unsignedResidue - kTwoPow64 : unsignedResidue;
  return signedResidue;
}

#endif  // VKNN_BITWISE_INT_GLSL
