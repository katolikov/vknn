// Bindings, push constants, tile geometry, and LDS panels of the {128,128,16} register-blocked GEMM
// over a packed quantized weight (matmul_tiled_wq_core.glsl holds the body; matmul_tiled_i4*.comp
// are the format wrappers). The binding interface matches matmul_gemv_wq_head.glsl: activations at
// 0, packed uint word stream at 1, output at 2, fp16 group scales at 3, outlier columns at 4/5
// (core/quant_weights.h); a wrapper defines WQ_BIAS to append the fused rank-1 [N] bias at 6.
//
// B staging is block-granular and format-independent in thread layout: a tile k-row is TN/8 packed
// 8-column blocks, and the TK * (TN/8) blocks of a tile spread exactly one per thread. Only the
// word width of a block differs per format, which the wrapper encodes in WQ_WORD_T / WQ_LOAD_W /
// WQ_BLOCK_COL before including the core.
#define TM 128
#define TN 128
#define TK 16
#define TILE 16        // threads per dim (16x16 = 256)
#define RM (TM / TILE) // 8 output rows per thread
#define RN (TN / TILE) // 8 output cols per thread
#define BPR (TN / 8)   // packed 8-column blocks per B-tile k-row

#if TK * BPR != TILE * TILE
#error B-panel block staging assumes TK * (TN / 8) == TILE * TILE (one packed block per thread)
#endif

layout(local_size_x = TILE, local_size_y = TILE) in;
layout(std430, binding = 0) readonly  buffer A  { STORE a[]; };
layout(std430, binding = 1) readonly  buffer BP { uint bp[]; };
layout(std430, binding = 2) writeonly buffer D  { STORE d[]; };
layout(std430, binding = 3) readonly  buffer SC { STORE sc[]; };
layout(std430, binding = 4) readonly  buffer OI { int oi[]; };
layout(std430, binding = 5) readonly  buffer OV { STORE ov[]; };
#ifdef WQ_BIAS
layout(std430, binding = 6) readonly  buffer BS { STORE bias[]; };
#endif
layout(push_constant) uniform PC {
  int total, M, N, K, group, nOut, rowWords, nGroups;
} pc;

// LDS panels in storage precision, like matmul_tiled_fast. Bs holds DEQUANTIZED weight values
// (q * scale as fp32, narrowed to STORE), so the micro-kernel is format-blind; under fp16 storage
// that narrowing rounds once per staged element — the same single-rounding the fp16 kernel's B
// panel carries.
shared STORE As[TK][TM];
shared STORE Bs[TK][TN];
