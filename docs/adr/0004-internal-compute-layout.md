# ADR-0004: Internal Vulkan compute layout = NC4HW4; conv strategy without cooperative matrix

## Status
Accepted (2026-06-24)

## Context
Canonical IR layout is NCHW (ONNX convention). The GPU backend needs a layout that maps well
to the Xclipse 960 (RDNA, wavefront 64, strong vec4 ALU, fp16). `vx_probe` confirmed
`VK_KHR_cooperative_matrix` is **absent**, so we cannot use matrix-core tiles for GEMM/conv.

## Decision
- **Internal Vulkan tensor layout = NC4HW4**: channels are grouped in blocks of 4 and stored as
  `vec4` (so a `[N,C,H,W]` tensor becomes `[N, ceil(C/4), H, W, 4]`). This is the MNN/ncnn-proven
  mobile-GPU layout: every load/store is a 16-byte `vec4`, matching RDNA's vector path and
  keeping fp16 packing natural (a `vec4` of fp16 = 8 bytes).
- **Conv strategies** (no coop-matrix):
  - 1×1 pointwise → treated as GEMM over the channel blocks, vec4 accumulation, with an
    autotuned output-tile / workgroup size (M8).
  - depthwise k×k → specialized per-channel-block kernel (no im2col).
  - general k×k → implicit-GEMM (gather on the fly) with vec4 accumulation; Winograd 3×3 is a
    future optimization (logged, not required for the MobileNet slice).
- Layout conversions (NCHW↔NC4HW4, NHWC I/O) are explicit kernels inserted automatically at
  backend boundaries by the layout pass.

## Consequences
- All Vulkan conv/pointwise kernels assume NC4HW4 inputs/outputs and `vec4` access.
- The packing kernels and the pre-packed-weight cache (M8) target NC4HW4.
- Revisit Winograd / subgroup-matmul tiling for further speedups after the slice is correct.
