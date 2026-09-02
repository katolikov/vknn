// Shared head of the pointwise formulation probes (microbench-only; never built by CMake).
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#include "store16.glsl"
#include "common.glsl"
#define WTILE 4
#define OCB 2
layout(local_size_x = 64, local_size_x_id = 0) in;
layout(std430, binding = 0) readonly buffer Src { f16vec4 src[]; };
layout(std430, binding = 1) readonly buffer Wt { f16vec4 wt[]; };
layout(std430, binding = 2) readonly buffer Bs { f16vec4 bias[]; };
layout(std430, binding = 3) writeonly buffer Dst { f16vec4 dst[]; };
layout(push_constant) uniform PC {
  int N, Cin, H, W, Cout, OH, OW, KH, KW, SH, SW, PT, PL, DH, DW, act;
  float actLo, actHi;
} pc;
