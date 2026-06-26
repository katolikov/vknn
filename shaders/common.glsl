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
  if (act == 4)         return x * clamp(x + 3.0, 0.0, 6.0) / 6.0;  // HardSwish
  if (act == 5)         return x / (1.0 + exp(-x));                 // SiLU / Swish
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
  if (op == 14) return x / (1.0 + exp(-x));            // silu
  if (op == 16) return cos(x);
  if (op == 17) return sin(x);
  if (op == 18) return 1.0 / x;                        // reciprocal
  if (op == 19) return max(x, 0.0) + log(1.0 + exp(-abs(x)));  // softplus
  if (op == 15) {                                      // erf (Abramowitz-Stegun 7.1.26, err<1.5e-7)
    float s = sign(x); float ax = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * ax);
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                      - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
    return s * y;
  }
  return x;
}

// Binary family (codes must match vx::BinaryType).
float vx_binary(float a, float b, int op) {
  if (op == 0) return a * b;
  if (op == 1) return a - b;
  if (op == 2) return a / b;
  if (op == 3) return max(a, b);
  if (op == 4) return min(a, b);
  if (op == 5) {
    // GLSL pow(x,y) is undefined (-> NaN) for x < 0, but ONNX/powf computes a negative base with an
    // INTEGER exponent (e.g. (-0.5)^2 = 0.25, used on the normalized ray directions of the camera
    // ray-map). Handle a negative base with an integer exponent by sign-correcting; a non-integer
    // exponent of a negative base is genuinely NaN (matches powf), so fall through to pow().
    if (a < 0.0 && b == round(b)) {
      float r = pow(-a, b);
      return (mod(round(b), 2.0) != 0.0) ? -r : r;
    }
    return pow(a, b);
  }
  return a + b;
}

#endif  // VX_COMMON_GLSL
