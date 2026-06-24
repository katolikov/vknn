# vxrt Worklog

Timestamped running log of work, device findings, decisions, blockers, workarounds.

## 2026-06-24

### Phase 0 — discovery (DONE)
- Probed host toolchain: NDK r27 (`27.0.12077973`, Vulkan hdrs 1.3.275), cmake 4.1.2, python3.14 +
  onnxruntime 1.25.1 / onnx 1.21.0 / numpy 2.4.3, adb 1.0.41. Installed missing `ninja` + `shaderc`
  (glslc) via brew.
- Device connected (`adb serial R3CY905E04M`): **Galaxy S26 SM-S942B, Exynos 2600 (s5e9965),
  Xclipse 960, Vulkan 1.4, Android 16 / API 36, arm64-v8a.** Exactly the intended target.
- Vulkan caps (vkjson): fp16 ✅, subgroupSize 64, 64KB shared, dma_buf import ✅, int dot product ✅,
  timeline semaphore / push descriptor ✅, **cooperative matrix ABSENT**, dedicated compute queue
  (family 1). timestampPeriod 39.0625ns.
- ION: `/dev/ion` gone → use DMA-BUF heaps (`/dev/dma_heap/system`), import fd into Vulkan.
- ENN: runtime libs present, `.nnc` samples present, but no headers / no on-device NNC compiler →
  documented stub.
- Wrote `docs/DEVICE_REPORT.md`. git initialised. Skeleton dirs created.

### M0 — skeleton & probe (in progress)
- Writing CMake + NDK build, core headers, Vulkan device init + `vx_probe`.

### M0 — DONE (verified on device)
- `vx_probe` built (arm64-v8a) and run on device. Reads identical caps to vkjson.
- Confirmed: Xclipse 960, Vulkan **1.4.304**, fp16/int8/storage16/storage8 all ON, int8dot ON,
  **coopmat OFF**, dma_buf/fd/ahb import ON, push-descriptor/timeline/dedicated ON.
- **UMA finding:** memory types 0 & 1 are DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT (type 1 also
  HOST_CACHED). → engine maps device-local memory directly, NO staging copies for upload. Big win.
- Compute queue family = 1 (dedicated compute), selected automatically.
