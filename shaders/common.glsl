// vxrt — shared GLSL definitions for compute shaders.
// Included by all *.comp. Compiled with: glslc --target-env=vulkan1.3
#ifndef VX_COMMON_GLSL
#define VX_COMMON_GLSL

#extension GL_GOOGLE_include_directive : require

// Activation fusion codes (kept in sync with vx::ActType in include/vx/op.h).
#define ACT_NONE  0
#define ACT_RELU  1
#define ACT_RELU6 2
#define ACT_CLIP  3

float vx_act(float x, int act, float lo, float hi) {
  if (act == ACT_RELU)  return max(x, 0.0);
  if (act == ACT_RELU6) return clamp(x, 0.0, 6.0);
  if (act == ACT_CLIP)  return clamp(x, lo, hi);
  return x;
}

#endif  // VX_COMMON_GLSL
