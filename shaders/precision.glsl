// Storage-precision selection for elementwise / data-movement shaders.
//
// One shader source compiles to BOTH the fp32 and the fp16 SPIR-V variant: the build emits
// <name>.spv normally and <name>_fp16.spv with -DUSE_FP16 (see CMakeLists). Use STORE as the buffer
// element type, read with float(buf[i]) and write buf[i] = STORE(value). Arithmetic is always fp32,
// so the casts are free no-ops in the fp32 variant. (Perf kernels that pack f16vec4 for bandwidth —
// conv/matmul/fc/… — keep a hand-written _fp16.comp instead; they're genuinely different, not just a
// type swap.)
#ifdef USE_FP16
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#define STORE float16_t
#include "store16.glsl"
#else
#define STORE float
#endif
