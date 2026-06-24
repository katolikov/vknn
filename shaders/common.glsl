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

// Unary family (codes must match vx::UnaryType in include/vx/op.h). a,b are op params.
float vx_unary(float x, int op, float a, float b) {
  if (op == 0)  return 1.0 / (1.0 + exp(-x));          // sigmoid
  if (op == 1)  return tanh(x);                        // tanh
  if (op == 2)  return x * clamp(x + 3.0, 0.0, 6.0) / 6.0;  // hardswish
  if (op == 3)  return clamp(a * x + b, 0.0, 1.0);     // hardsigmoid
  if (op == 4)  return x > 0.0 ? x : a * x;            // leakyrelu
  if (op == 5)  return x > 0.0 ? x : a * (exp(x) - 1.0);  // elu
  if (op == 6)  return abs(x);
  if (op == 7)  return -x;
  if (op == 8)  return exp(x);
  if (op == 9)  return log(x);
  if (op == 10) return sqrt(x);
  if (op == 11) return floor(x);
  if (op == 12) return ceil(x);
  if (op == 13) return max(x, 0.0);                    // relu
  return x;
}

// Binary family (codes must match vx::BinaryType).
float vx_binary(float a, float b, int op) {
  if (op == 0) return a * b;
  if (op == 1) return a - b;
  if (op == 2) return a / b;
  if (op == 3) return max(a, b);
  if (op == 4) return min(a, b);
  if (op == 5) return pow(a, b);
  return a + b;
}

#endif  // VX_COMMON_GLSL
