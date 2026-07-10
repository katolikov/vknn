// fp32 -> fp16 bit conversion in plain 32-bit integer math, shared by store16.glsl (the fp16
// activation-store rounding) and kernels that write fp16 half-words into a wider word buffer
// (chain_feedback.comp). Requires no 16-bit-type extensions, so a caller that never touches
// float16_t values can still produce host-identical fp16 bit patterns.
#ifndef VX_F16BITS_GLSL
#define VX_F16BITS_GLSL
// Saturate a finite fp32 activation result to the largest finite fp16 magnitude so the fp16-narrowing
// store never turns a finite value into +/-inf (an inf then feeds 0*inf = NaN and poisons the output).
// NaN passes through unchanged. Mirrors clampToFp16Range in src/import/convert_fp16.cpp, so an
// activation store saturates exactly like an imported constant. Identity for in-range values, so a
// store that never overflows fp16 is byte-unchanged.
float vxSatF16(float x) { return isnan(x) ? x : clamp(x, -65504.0, 65504.0); }
vec4  vxSatF16(vec4 v)  { return vec4(vxSatF16(v.x), vxSatF16(v.y), vxSatF16(v.z), vxSatF16(v.w)); }
uint vknnF32ToF16RteBits(float x) {  // same bits as a correct IEEE RTE FConvert
  uint f = floatBitsToUint(x);
  uint s = (f >> 16) & 0x8000u;
  uint e = (f >> 23) & 0xFFu;
  uint m = f & 0x7FFFFFu;
  if (e == 255u) return s | 0x7C00u | (m != 0u ? 0x200u : 0u); // inf / quieted nan
  int ne = int(e) - 112;                                       // rebias 127 -> 15
  if (ne >= 31) return s | 0x7C00u;                            // overflow -> inf
  if (ne <= 0) {                                               // subnormal half or zero
    if (ne < -10) return s;
    m |= 0x800000u;
    uint shift = uint(14 - ne);
    uint h = m >> shift;
    uint rem = m & ((1u << shift) - 1u);
    uint halfway = 1u << (shift - 1u);
    if (rem > halfway || (rem == halfway && (h & 1u) == 1u)) h += 1u;
    return s | h;
  }
  uint h = (uint(ne) << 10) | (m >> 13);
  uint rem = m & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) == 1u)) h += 1u; // carry rolls into the exponent correctly
  return s | h;
}
#endif  // VX_F16BITS_GLSL
