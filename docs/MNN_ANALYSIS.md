# MNN Vulkan backend — analysis, and what it means for VKNN

This document analyzes how MNN's Vulkan path performs relative to VKNN on the same device.
It is based on MNN's source (`source/backend/vulkan/`) and on-device measurements.

## What MNN does

1. **Two Vulkan backends, with an image-based default.**
   `source/backend/vulkan/CMakeLists.txt` line 1: `option(MNN_VULKAN_IMAGE "Use Image as basic type" ON)`.
   A stock `MNN_VULKAN=ON` build uses the **image** backend, storing
   activations as **`VkImage` (2D textures)** rather than SSBOs.
   - Conv reads input/weights via `sampler2D` + `texelFetch`, writes via `image2D` + `imageStore`.
   - NC4HW4 maps to a 2D image: texel = 4 channels (RGBA16F), x = `w + cBlock*W`, y = `h + n*H`.
   - A `buffer` (SSBO) backend also exists; it is structurally closest to VKNN.

2. **`c8w4` register tiling for 1×1 conv** (`convolution1x1_c8w4.comp`): each thread computes
   **8 output channels (2 blocks) × 4 width** as an *outer-product* microkernel
   (`acc += in.x*w0 + in.y*w1 + in.z*w2 + in.w*w3`), so each input texel feeds 2 output blocks and
   each weight texel feeds 4 width positions. This uses 8 `vec4` accumulators per thread.

3. **Winograd via image intermediates** (`winogradTransformSource2_3_1.comp` +
   `winogradTransformDest2_3_1.comp` + a matmul): F(2,3), with the V/M
   intermediates stored as **images**, so the multi-pass traffic hits the texture cache instead of plain
   global memory.

4. fp16 throughout (`GL_AMD_gpu_shader_half_float`), per-shape kernel selection, a `gemm16x16`
   tiled GEMM, and depthwise width-tiles (`convolutionDepthwise_s1d1_w2`).

## The key difference vs VKNN
VKNN is SSBO-only (like MNN's *buffer* backend); MNN defaults to the *image* backend. On mobile
GPUs (including this AMD RDNA-class mobile GPU) **textures have a dedicated cache with 2D spatial locality**,
so heavy register tiling (c8w4) pays off — the many input/weight reads hit cache cheaply. In plain
SSBO those same reads pressure the L1/L2, and the heavy tiling needs more registers, which
drops occupancy.

## On-device experiments
| Change ported from MNN | Result in VKNN (SSBO) |
|---|---|
| `c8w4` 1×1 tiling (8 acc) | **regressed** (MobileNetV2 14.7→22.5 ms): 8 `vec4` acc ≈ 80 VGPR → occupancy collapse |
| Depthwise width-tile (W4) | **neutral/regressed** (depthwise is bandwidth-light; weights already cached) |
| Winograd F(2,3), 3-pass | correct but memory-bound (V/M in global): 71 ms vs direct 61 ms |
| Winograd F(2,3), fused | correct but register-bound (16 acc): 142 ms |
| register-tiled 1×1 (4 acc) | **kept** — the SSBO sweet spot |

The heavy tiling that makes MNN fast pays off only with texture-cached reads. In SSBO the
register/occupancy wall is hit first.

## Image-backend evaluation
`vknn_image_bench` runs the same 1×1 conv three ways: SSBO, storage-image (`imageLoad`), and
sampler2D (`texelFetch`, the texture-cache path MNN uses). On the target GPU:

| shape (Cin→Cout @HW) | SSBO | storage-image | sampler2D (texture cache) |
|---|---|---|---|
| 256→256 @14×14 | **0.21 ms** | 0.42 ms (0.51×) | 0.42 ms (0.51×) |
| 960→160 @7×7 | **0.59 ms** | 1.44 ms (0.41×) | 1.46 ms (0.40×) |
| 64→256 @56×56 | **0.42 ms** | 0.44 ms (0.96×) | 0.45 ms (0.95×) |

Both image paths are ~2× slower than SSBO here (image outputs verified cosine=1.0; the SSBO
correctness print has a harness de-pack bug, but its timing is valid — same kernel that
gives cosine 0.999965 in MobileNetV2). On this RDNA-class mobile driver, textures do **not**
beat SSBO, and converting VKNN to images is slower.

## Conclusion
MNN's advantage is not the storage type — it is kernel quality plus per-shape tuning. MNN wins
even while paying the image penalty on this GPU, so its conv microkernels (c8w4, gemm16x16, tuned
Winograd) and op selection are more optimized. The direction for VKNN is better **SSBO** kernels
and **less inter-op traffic (operator/block fusion)**, not an image backend.
