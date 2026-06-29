// Force round-to-nearest-even for fp32->fp16 stores. Include from every fp16 kernel (after the
// fp16 arithmetic extension is enabled).
//
// The Samsung Xclipse driver's default rounding for an fp16-narrowing conversion (float16_t(x),
// f16vec4(x), STORE(x)) is round-toward-zero. That biases every activation store ~half a ULP toward
// zero; across a deep network the biased stores compound into a systematic magnitude shrink (SNR
// collapses while cosine stays ~ 1). The SPV_KHR_float_controls RoundingModeRTE execution mode makes
// every 16-bit-result conversion in this shader round to nearest even, at no runtime cost.
#ifndef VX_STORE16_GLSL
#define VX_STORE16_GLSL
#extension GL_EXT_spirv_intrinsics : require
// RoundingModeRTE execution mode (4462) for 16-bit results; requires the RoundingModeRTE capability
// (4467) and the SPV_KHR_float_controls extension.
spirv_execution_mode(capabilities = [4467], extensions = ["SPV_KHR_float_controls"], 4462, 16);
#endif  // VX_STORE16_GLSL
