# MNN Vulkan backend — analysis, and what it means for VKNN

Goal: understand *why* MNN's Vulkan path is faster than VKNN on the same device, and decide what to
change. This is from reading MNN's source (`source/backend/vulkan/`) + experiments on-device.

## What MNN actually does

1. **Two Vulkan backends, and the default is image-based.**
   `source/backend/vulkan/CMakeLists.txt` line 1: `option(MNN_VULKAN_IMAGE "Use Image as basic type" ON)`.
   So a stock `MNN_VULKAN=ON` build (what we benchmarked) uses the **image** backend, which stores
   activations as **`VkImage` (2D textures)**, not SSBOs.
   - Conv reads input/weights via `sampler2D` + `texelFetch`, writes via `image2D` + `imageStore`.
   - NC4HW4 maps to a 2D image: texel = 4 channels (RGBA16F), x = `w + cBlock*W`, y = `h + n*H`.
   - There is also a `buffer` (SSBO) backend — that's the one structurally closest to VKNN.

2. **`c8w4` register tiling for 1×1 conv** (`convolution1x1_c8w4.comp`): each thread computes
   **8 output channels (2 blocks) × 4 width** as an *outer-product* microkernel
   (`acc += in.x*w0 + in.y*w1 + in.z*w2 + in.w*w3`), so each input texel feeds 2 output blocks and
   each weight texel feeds 4 width positions. 8 `vec4` accumulators per thread.

3. **Winograd via image intermediates** (`winogradTransformSource2_3_1.comp` +
   `winogradTransformDest2_3_1.comp` + a matmul): same F(2,3) we implemented, but the V/M
   intermediates are **images**, so the multi-pass traffic hits the texture cache instead of plain
   global memory.

4. fp16 throughout (`GL_AMD_gpu_shader_half_float`), per-shape kernel selection, a `gemm16x16`
   tiled GEMM, depthwise width-tiles (`convolutionDepthwise_s1d1_w2`).

## The key difference vs VKNN
VKNN is SSBO-only (like MNN's *buffer* backend). MNN's default is the *image* backend. On mobile
GPUs (incl. this AMD RDNA-class mobile GPU) **textures have a dedicated cache with 2D spatial locality**,
so heavy register tiling (c8w4) pays off — the many input/weight reads hit cache cheaply. In plain
SSBO, those same reads pressure the L1/L2 and, more importantly, the heavy tiling needs more
registers, dropping occupancy.

## Experiments run on-device (to test the hypothesis, not guess)
| Change ported from MNN | Result in VKNN (SSBO) |
|---|---|
| `c8w4` 1×1 tiling (8 acc) | **regressed** (MobileNetV2 14.7→22.5 ms): 8 `vec4` acc ≈ 80 VGPR → occupancy collapse |
| Depthwise width-tile (W4) | **neutral/regressed** (depthwise is bandwidth-light; weights already cached) |
| Winograd F(2,3), 3-pass | correct but memory-bound (V/M in global): 71 ms vs direct 61 ms |
| Winograd F(2,3), fused | correct but register-bound (16 acc): 142 ms |
| register-tiled 1×1 (4 acc) | **kept** — the SSBO sweet spot |

**Conclusion (evidence-backed): the heavy tiling that makes MNN fast only pays off with
texture-cached reads. In SSBO we hit the register/occupancy wall first.** So the single highest-
impact change is to add an **image (VkImage) backend** — exactly MNN's default.

## Plan: add an image backend to VKNN (two backends, image default)
- `vk::Image` (RGBA16F 2D, `VK_IMAGE_LAYOUT_GENERAL`, storage usage) holding NC4HW4 as above.
- Ops read/write with `imageLoad`/`imageStore` on storage images (no samplers, no layout
  transitions — GENERAL throughout; simplest correct design). Barriers stay compute→compute.
- Pack/unpack at boundaries via a staging buffer + `vkCmdCopyBufferToImage`/`CopyImageToBuffer`.
- Port conv (c8w4 1×1, depthwise w4, general), add, pool, fc to image kernels.
- Selectable via config (`imageBackend`, default on when the device supports the format); the SSBO
  path stays as the portable fallback (and for devices without the image features).

Before doing the full port, `vknn_image_bench` measures image-backed c8w4 1×1 conv against the SSBO
version on representative shapes — so the rewrite is justified by a number, not a hunch.

## VERDICT (measured on-device) — do NOT build an image backend
`vknn_image_bench` runs the same 1×1 conv three ways: SSBO, storage-image (`imageLoad`), and
sampler2D (`texelFetch`, the real texture-cache path MNN uses). On the target GPU:

| shape (Cin→Cout @HW) | SSBO | storage-image | sampler2D (texture cache) |
|---|---|---|---|
| 256→256 @14×14 | **0.21 ms** | 0.42 ms (0.51×) | 0.42 ms (0.51×) |
| 960→160 @7×7 | **0.59 ms** | 1.44 ms (0.41×) | 1.46 ms (0.40×) |
| 64→256 @56×56 | **0.42 ms** | 0.44 ms (0.96×) | 0.45 ms (0.95×) |

**Both image paths are ~2× SLOWER than SSBO here** (image outputs verified cosine=1.0; the SSBO
correctness print has a harness de-pack bug, but its timing is valid — it's the same kernel that
gives cosine 0.999965 in MobileNetV2). So on this RDNA-class mobile driver, textures do **not**
beat SSBO — converting VKNN to images would make it *slower*.

**Revised conclusion: MNN's advantage is not the storage type — it's kernel quality + per-shape
tuning.** MNN is faster *even while paying the image penalty on this GPU*, which means its conv
microkernels (c8w4, gemm16x16, tuned Winograd) and op selection are simply better optimized than
ours. The right direction for VKNN is therefore better **SSBO** kernels and **less inter-op
traffic (operator/block fusion)** — not an image backend.
