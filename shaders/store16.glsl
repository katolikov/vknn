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
// vxSatF16 + vknnF32ToF16RteBits live in f16bits.glsl (no 16-bit-type extensions), so a kernel that
// only writes fp16 BIT PATTERNS into a word buffer shares them without pulling in float16_t.
#include "f16bits.glsl"
// The integer-math vknnRte16 below is defined for EVERY fp16 kernel (not just VKNN_NO_RTE ones):
// the fused-pointwise VM rounds each step with it so a step's bytes never depend on which host
// kernel runs the unit — some drivers' RoundingModeRTE breaks round-to-nearest ties differently
// from IEEE ties-to-even, and a fused unit must round exactly like the standalone elementwise
// kernels it replaces (which also store through it).
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
float16_t vknnRte16(float v) {
  return uint16BitsToFloat16(uint16_t(vknnF32ToF16RteBits(vxSatF16(v))));
}
#ifndef VKNN_NO_RTE
#extension GL_EXT_spirv_intrinsics : require
// RoundingModeRTE execution mode (4462) for 16-bit results; requires the RoundingModeRTE capability
// (4467) and the SPV_KHR_float_controls extension.
spirv_execution_mode(capabilities = [4467], extensions = ["SPV_KHR_float_controls"], 4462, 16);
#define TO_STORE(x) float16_t(vxSatF16(x))
#else
// Kernels that must not carry the float-controls execution mode (some drivers miscompile them:
// nondeterministic output, faults under load; and both float16_t(x) and packHalf2x16 round toward
// zero there) convert explicitly instead.
#define TO_STORE(x) vknnRte16(float(x))
#endif
// Saturating fp16 store of a whole vec4 activation result. Narrows each lane through TO_STORE, so it
// saturates and rounds identically to the scalar store and stays byte-identical to f16vec4(v) for
// in-range v.
#define TO_STORE4(v) f16vec4(TO_STORE((v).x), TO_STORE((v).y), TO_STORE((v).z), TO_STORE((v).w))
#endif  // VX_STORE16_GLSL
