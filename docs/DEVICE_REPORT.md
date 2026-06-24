# Device & Environment Report

> Source of truth: probed directly from the connected device on 2026-06-24 via `adb`,
> `cmd gpu vkjson`, and the `vx_probe` binary (see `examples/probe`). Re-runnable with
> `scripts/device_report.sh`.

## Connected device

| Property | Value |
|---|---|
| `ro.product.model` | **SM-S942B** (Galaxy S26) |
| `ro.product.device` | `m1s` |
| `ro.soc.manufacturer` | Samsung |
| `ro.soc.model` | **s5e9965** (Exynos 2600) |
| `ro.board.platform` | `erd9965` |
| `ro.hardware` | `s5e9965` |
| `ro.build.version.release` | **16** (Android 16) |
| `ro.build.version.sdk` | **36** |
| `ro.product.cpu.abi` | `arm64-v8a` |
| adb serial | `R3CY905E04M` |

✅ This is the intended target: **Exynos 2600 / Xclipse 960 / Android 16 / API 36**. (Not a
Snapdragon unit — the Exynos-specific ION/ENN paths apply.)

## CPU features (`/proc/cpuinfo`)

```
fp asimd aes pmull sha1 sha2 crc32 atomics fphp asimdhp asimdrdm jscvt fcma lrcpc dcpop
sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm ssbs sb sve2 svesha3
svesm4 flagm2 frint svei8mm svebf16 i8mm bf16 dgh bti ecv afp sme sme2 ... lrcpc3
```

Relevant for the NEON CPU backend: **`asimd` (NEON), `fphp`/`asimdhp` (fp16 arithmetic),
`asimddp` (SDOT/UDOT int8 dot product), `i8mm`, `bf16`, `sve2`, `sme2`.** We target NEON
(asimd) + fp16 (asimdhp) for hot kernels; SVE/SME are available but not used (NEON is
portable and sufficient for MobileNet).

## GPU / Vulkan (`cmd gpu vkjson` + `vx_probe`)

| Property | Value |
|---|---|
| Device name | **Samsung Xclipse 960** |
| Driver | Samsung Proprietary (SPAL) **25.2.39**, git `ee686460f4` |
| `driverID` | 21 = `VK_DRIVER_ID_SAMSUNG_PROPRIETARY` |
| API version | **1.4.0** (`0x00404000`) |
| Architecture | AMD **RDNA**-derived (`VK_AMD_shader_core_properties` present) |
| Wavefront / subgroupSize | **64** |
| Compute units (active) | 16 (2 shader engines × 2 arrays × 4 CU) |
| `maxComputeSharedMemorySize` | **65536** (64 KiB) |
| `maxComputeWorkGroupInvocations` | **1024** |
| `maxComputeWorkGroupSize` | `[1024, 1024, 1024]` |
| `timestampComputeAndGraphics` | true |
| `timestampPeriod` | **39.0625 ns** (used by the GPU profiler) |

### Queue families

| Family | Flags | Count | timestampValidBits |
|---|---|---|---|
| 0 | `0x1f` = GRAPHICS\|COMPUTE\|TRANSFER\|SPARSE\|PROTECTED | 4 | 64 |
| 1 | `0x1e` = COMPUTE\|TRANSFER\|SPARSE\|PROTECTED (**no graphics**) | 2 | 64 |

→ Family 1 is a **dedicated compute queue** — we select it preferentially for the compute backend.

### Performance-relevant features & extensions (all probed present unless noted)

| Capability | Status | Use in vxrt |
|---|---|---|
| `shaderFloat16` (`VK_KHR_shader_float16_int8`) | ✅ | **fp16 fast path** (default precision) |
| `shaderInt16` | ✅ | fp16 packing helpers |
| `VK_KHR_16bit_storage` / `storageBuffer16BitAccess` | ✅ | fp16 SSBO storage |
| `VK_KHR_8bit_storage` | ✅ | int8 stretch goal |
| `VK_KHR_shader_integer_dot_product` (8-bit signed accelerated) | ✅ | reserved for int8 stretch |
| **`VK_KHR_cooperative_matrix`** | ❌ **absent** | → use subgroup-tiled GEMM instead (ADR-0004) |
| `VK_EXT_external_memory_dma_buf` | ✅ | **ION/DMA-BUF zero-copy import** |
| `VK_KHR_external_memory_fd` (`vkGetMemoryFdKHR`/`VkImportMemoryFdInfoKHR`) | ✅ | zero-copy import path (preferred) |
| `VK_ANDROID_external_memory_android_hardware_buffer` | ✅ | zero-copy fallback path |
| `VK_KHR_timeline_semaphore` | ✅ | async submit / overlap |
| `VK_KHR_push_descriptor` | ✅ | fast per-dispatch descriptor binding |
| `VK_KHR_dedicated_allocation` | ✅ | dedicated allocs for large tensors / imports |
| `VK_EXT_memory_budget` | ✅ | memory stats in profiler |
| subgroup ballot/vote/extended-types/rotate | ✅ | reductions (softmax, pooling, GEMM) |
| `VK_KHR_shader_float_controls2` | ✅ | fp16 denorm/rounding control |

## ION / DMA-BUF allocation mechanism

- `/dev/ion` — **absent** (classic ION removed; this is Android 12+).
- `/dev/dma_heap/` — **present**, with heaps including:
  `system`, `system-uncached`, `system-direct`, `system-direct-uncached`, `crypto`,
  `dmabuf_container`, plus camera/secure heaps.
- Libraries present in `/vendor/lib64` **and** `/system/lib64`:
  `libdmabufheap.so`, `libion.so`, `libion_exynos.so`.

**Decision (ADR-0005):** the "exynos_ion" mechanism on this build is **DMA-BUF heaps**.
vxrt allocates from `/dev/dma_heap/system` via the kernel `DMA_HEAP_IOCTL_ALLOC` ioctl
(no external lib dependency — header `linux/dma-heap.h`), obtains a dma-buf **fd**, `mmap`s
it for CPU access, and imports the **same fd** into Vulkan via
`VkImportMemoryFdInfoKHR` (handle type `DMA_BUF_BIT_EXT`). This is true zero-copy on the
unified-memory SoC.

## ENN / NPU

- ENN runtime libraries **present** in `/vendor/lib64`:
  `libenn_public_api_cpp.so`, `libenn_engine.so`, `libenn_model.so`,
  `libenn_user_driver_gpu.so`, `libenn_user_driver_cpu.so`, `libenn_user_driver_unified.so`,
  `vendor.samsung_slsi.hardware.enn_aidl-V1-ndk.so`.
- `.nnc` model samples present (e.g. `aiAutoFocus_v1_0_O2_MultiCore.nnc`) → confirms NNC is
  the on-device model format consumed by ENN.
- `/vendor/etc/enn` present (config).
- **No public ENN headers and no on-device NNC compiler** are available (the NNC compiler is
  an offline Samsung SDK tool we do not have). Therefore the ENN backend is shipped as a
  **documented, runtime-probing stub** (M7): it `dlopen`s the ENN libs to prove presence,
  registers + is config-selectable, and returns a clear "unavailable: no NNC model / no public
  headers" at execute. See `LIMITATIONS.md` and ADR-0007.

## Host toolchain

| Tool | Version / path |
|---|---|
| Android NDK | **r27** (`27.0.12077973`) — Vulkan headers v1.3.275 |
| CMake | 4.1.2 |
| Ninja | installed (brew) |
| glslc (shaderc) | installed (brew) — SPIR-V from GLSL |
| Python | 3.14.0 |
| onnxruntime | 1.25.1 (golden reference, CPU) |
| onnx | 1.21.0 |
| numpy | 2.4.3 |
| adb | 1.0.41 (36.0.0) |

Scratch dir on device: `/data/local/tmp/vxrt/` (writable, confirmed).

## Architecture decisions driven by these facts

1. **Default precision = fp16** (shaderFloat16 + 16-bit storage), fp32 kept for reference/correctness.
2. **No cooperative matrix** → pointwise/Gemm uses a subgroup-tiled vec4 GEMM; 3×3 depthwise gets a specialized kernel.
3. **Internal Vulkan layout = NC4HW4** (channel-packed ×4, vec4 loads) — proven on mobile GPUs, fits RDNA vec4 ALUs. (ADR-0004)
4. **ION = DMA-BUF heap** + `VkImportMemoryFdInfoKHR` zero-copy. (ADR-0005)
5. **Dedicated compute queue** (family 1) for the Vulkan backend.
6. **ENN = runtime-probing stub** (no headers / no NNC compiler on device). (ADR-0007)
