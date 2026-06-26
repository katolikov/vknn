# ADR-0004: Internal Vulkan compute layout = NC4HW4; conv strategy without cooperative matrix

## Status
Accepted (2026-06-24)

## Context
The canonical IR layout is NCHW (ONNX convention). The GPU backend requires a layout that maps well
to the target GPU (an AMD RDNA-class mobile GPU, wavefront 64, fast vec4 ALU, fp16). `vknn_probe` reports
`VK_KHR_cooperative_matrix` as **absent**, so matrix-core tiles for GEMM/conv are unavailable.

## Decision
- **Internal Vulkan tensor layout is NC4HW4**: channels are grouped in blocks of 4 and stored as
  `vec4` (a `[N,C,H,W]` tensor becomes `[N, ceil(C/4), H, W, 4]`). This is the mobile-GPU layout
  MNN and ncnn use: every load/store is a 16-byte `vec4`, matching the GPU's vector path and
  keeping fp16 packing natural (a `vec4` of fp16 is 8 bytes).
- **Conv strategies** (no coop-matrix):
  - 1×1 pointwise is treated as GEMM over the channel blocks, vec4 accumulation, with an
    autotuned output-tile / workgroup size (M8).
  - depthwise k×k uses a specialized per-channel-block kernel (no im2col).
  - general k×k uses implicit-GEMM (gather on the fly) with vec4 accumulation; Winograd 3×3 is a
    further optimization (not required for the MobileNet slice).
- Layout conversions (NCHW↔NC4HW4, NHWC I/O) are explicit kernels, inserted automatically by the
  layout pass at backend boundaries.

## Consequences
- All Vulkan conv/pointwise kernels assume NC4HW4 inputs/outputs and `vec4` access.
- The packing kernels and the pre-packed-weight cache (M8) target NC4HW4.
- Winograd / subgroup-matmul tiling remains available for further speedups.
