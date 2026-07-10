// Bindings, push constants, and reduction scratch of the split-K mat-vec over a packed quantized
// weight (matmul_gemv_wq_core.glsl holds the body; matmul_gemv_i4*.comp are the format wrappers).
// Every format shares this interface: activations at 0, the packed payload as a uint word stream at
// 1, output at 2, fp16 group scales at 3, outlier columns at 4/5 (core/quant_weights.h). A wrapper
// defines WQ_BIAS to append the fused rank-1 [N] bias at binding 6.
//
// The thread layout is format-independent: WQ_NX lanes each own 8 ADJACENT outputs n and WQ_KS
// lanes split the k reduction, partials reduced through shared memory in ascending lane order
// (deterministic). A workgroup covers 64 outputs of one output row; the dispatch is
// (ceil(N/64), rows). Only the packed-word width of a lane's 8-output block differs per format,
// which the wrapper encodes in WQ_WORD_T / WQ_LOAD_W / WQ_BLOCK_COL before including the core.
#define WQ_NX 8  // lanes along n, 8 outputs each: 64 outputs per workgroup
#define WQ_KS 16 // lanes reducing over k; 8 x 16 = 128 == the Vulkan-guaranteed workgroup size
#define WQ_U 8   // k steps a lane loads before it consumes any of them (memory-level parallelism)
layout(local_size_x = WQ_NX, local_size_y = WQ_KS) in;
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

shared vec4 part0[WQ_KS][WQ_NX];
shared vec4 part1[WQ_KS][WQ_NX];
