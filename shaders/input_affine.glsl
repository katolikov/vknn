// Consumer-side pointwise prologue ("input affine"). A pointwise unit whose producer cannot host it
// (a zero-copy Concat view, a graph input) is applied by the conv that consumes it instead: every
// input vec4 the conv loads passes through a per-channel scale, a per-channel shift and an
// activation, in that order, in place of the standalone FusedPointwise node that would write the
// transformed tensor for the conv to read back. Included by the conv kernels' _pro variants
// (compiled with -DPW_PRO=1); the host binds the scale and shift as NC4HW4 [N,C,1,1] tensors
// (one PRO_VEC per channel-block) at PRO_BASE and PRO_BASE + 1 and passes the unit's flags and
// activation in ConvPC's proFlags / proAct / proLo / proHi (src/backend/vulkan/ops/vk_op_common.h;
// the flag bits mirror kInputAffine* in src/core/input_affine.h).
//
// The unit computes in fp32 and hands the value to the conv's multiplies unrounded: the pass attaches
// a prologue only in its relaxed mode, where a fused unit already keeps fewer roundings than the
// unfused graph. The including kernel defines PRO_BASE and PRO_VEC (f16vec4 / vec4) before this
// file; vx_act comes from common.glsl.
#ifndef VKNN_INPUT_AFFINE_GLSL
#define VKNN_INPUT_AFFINE_GLSL
#define PRO_HAS_SCALE 1 // kInputAffineHasScale
#define PRO_HAS_SHIFT 2 // kInputAffineHasShift
#define PRO_BATCHED 8   // kInputAffineBatched: the scale/shift tensors carry one block row per batch
layout(std430, binding = PRO_BASE) readonly buffer ProScale { PRO_VEC proScale[]; };
layout(std430, binding = PRO_BASE + 1) readonly buffer ProShift { PRO_VEC proShift[]; };

vec4 proAct4(vec4 v, int act, float lo, float hi) {
  return vec4(vx_act(v.x, act, lo, hi), vx_act(v.y, act, lo, hi), vx_act(v.z, act, lo, hi), vx_act(v.w, act, lo, hi));
}
// The scale / shift vec4 of input channel-block `icb` of batch `n` (unity / zero when the unit has
// none, so a kernel can hoist one pair per block and apply it to every pixel of the block).
vec4 proScaleOf(int flags, int n, int icb, int Cinb) {
  return (flags & PRO_HAS_SCALE) != 0 ? vec4(proScale[((flags & PRO_BATCHED) != 0 ? n * Cinb : 0) + icb]) : vec4(1.0);
}
vec4 proShiftOf(int flags, int n, int icb, int Cinb) {
  return (flags & PRO_HAS_SHIFT) != 0 ? vec4(proShift[((flags & PRO_BATCHED) != 0 ? n * Cinb : 0) + icb]) : vec4(0.0);
}
// The unit applied to one loaded input vec4 of a block with scale s and shift b: one fma and the
// activation, so a kernel that reuses each load over many multiplies keeps its arithmetic rate.
vec4 proApply(vec4 x, vec4 s, vec4 b, int flags, int act, float lo, float hi) {
  x = fma(x, s, b);
  return proAct4(x, act, lo, hi);
}
#endif
