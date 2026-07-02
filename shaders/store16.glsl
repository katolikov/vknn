// Force round-to-nearest-even for fp32->fp16 stores. Include from every fp16 kernel (after the
// fp16 arithmetic extension is enabled).
//
// Some mobile drivers round an fp16-narrowing conversion (float16_t(x),
// f16vec4(x), STORE(x)) toward zero. That biases every activation store ~half a ULP toward
// zero; across a deep network the biased stores compound into a systematic magnitude shrink (SNR
// collapses while cosine stays ~ 1). The SPV_KHR_float_controls RoundingModeRTE execution mode makes
// every 16-bit-result conversion in this shader round to nearest even, at no runtime cost.
#ifndef VX_STORE16_GLSL
#define VX_STORE16_GLSL
#ifndef VKNN_NO_RTE
#extension GL_EXT_spirv_intrinsics : require
// RoundingModeRTE execution mode (4462) for 16-bit results; requires the RoundingModeRTE capability
// (4467) and the SPV_KHR_float_controls extension.
spirv_execution_mode(capabilities = [4467], extensions = ["SPV_KHR_float_controls"], 4462, 16);
#define TO_STORE(x) float16_t(x)
#else
// Explicit round-to-nearest-even conversion for kernels that must not carry the float-controls
// execution mode: some drivers miscompile kernels under RoundingModeRTE (nondeterministic
// output, faults under load), and both float16_t(x) and packHalf2x16 round toward zero there.
// The integer-math conversion below produces the same bits as a correct RTE FConvert.
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
uint vknnF32ToF16RteBits(float x) {
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
float16_t vknnRte16(float v) {
  return uint16BitsToFloat16(uint16_t(vknnF32ToF16RteBits(v)));
}
#define TO_STORE(x) vknnRte16(float(x))
#endif
#endif  // VX_STORE16_GLSL
