// Shared head of the 3x3 formulation probes (microbench-only).
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#include "store16.glsl"
#include "common.glsl"
#ifndef OCB_BLK
#define OCB_BLK 2
#endif
#ifndef PTILE
#define PTILE 4
#endif
#ifndef STRIDE
#define STRIDE 1
#endif
#define KEXT 3
#define SEG ((PTILE - 1) * STRIDE + KEXT)
layout(local_size_x = 64, local_size_x_id = 0) in;
layout(std430, binding = 0) readonly buffer Src { f16vec4 src[]; };
layout(std430, binding = 1) readonly buffer Wt { f16vec4 wt[]; };
layout(std430, binding = 2) readonly buffer Bs { f16vec4 bias[]; };
layout(std430, binding = 3) writeonly buffer Dst { f16vec4 dst[]; };
layout(push_constant) uniform PC {
  int N, Cin, H, W, Cout, OH, OW, KH, KW, SH, SW, PT, PL, DH, DW, act;
  float actLo, actHi;
} pc;
