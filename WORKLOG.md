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

### M1 — DONE (verified on device)
- Vulkan compute foundation: `vk::Buffer` (UMA direct-mapped, dma-buf import path),
  `vk::ComputePipeline` (push descriptors, spec constants, push constants), `vk::PipelineCache`
  (disk-serialized), `vk::CommandRunner` (one-shot + recordable, barriers).
- `add.comp` (elementwise add) compiled by glslc, embedded, run on device:
  **GPU add == CPU add, maxAbsErr = 0.0**, pipeline cache written (1012 B) and reused on run 2.
- ADRs 0001-0005,0007 recorded.

### M2 — DONE (verified on device)
- Core IR (Tensor/TensorDesc/RtTensor, Graph, Node, Attributes), Config (+JSON parser),
  Profiler, Backend abstraction + registry, CPU op registry. Segment-based execution model
  (maximal same-backend runs; boundary residency reconciliation) designed for Vulkan + fallback.
- **Dependency-free ONNX importer** (hand-rolled protobuf wire parser) — no protobuf lib.
- CPU reference ops: Conv (general/depthwise/pointwise), Gemm, Clip, Relu, Add(broadcast),
  GlobalAveragePool, Softmax, BatchNorm, Identity, Reshape, Flatten, Shape, Constant, Gather,
  Unsqueeze, Concat. MobileNetV2 classifier preamble runs directly (no const-fold needed on CPU).
- Golden: onnxruntime on cat/dog image -> top-1 class 258 (Samoyed), 105 per-layer dumps.
- **MobileNetV2 CPU on device: cosine=1.000000, maxAbsErr=1.62e-05, top-1 258==258 PASS.**
  Host build identical. Session create ~12 ms on device.

### M3 — Vulkan MobileNetV2 fp32 DONE (verified on device, GPU)
- Graph passes wired into Session: inferShapes (incl. Conv/Gemm/Reshape), foldBatchNorm,
  fuseActivations (35 Clip/Relu -> Conv/Gemm), constFold (5 shape nodes), DCE. 105 -> 65 nodes.
- NC4HW4 packed layout. Shaders: pack/unpack, conv (general group=1), dwconv (depthwise),
  avgpool (global), fc (gemm), add (reused). Weight prepacking on host.
- VulkanBackend + VulkanSegment: per-tensor device buffers, ops prepare()+record(), ONE
  pre-recorded command buffer for the whole static graph, timestamp query pool, CPU<->NC4HW4
  pack/unpack at boundaries.
- Bugs found & fixed on device: (1) Session member destruction order freed Vulkan buffers
  after the VkDevice (teardown segfault) -> reordered. (2) Reshape output shape not inferred ->
  undersized device buffer -> Gemm read OOB garbage -> added Reshape shape inference.
- **RESULT on Xclipse 960 GPU: top-5 exact match, cosine=1.000000, maxAbsErr=1.3e-05,
  top-1 258==258 PASS. Latency: median 24.35 ms = 41.1 fps (fp32). CPU ref: 672 ms = 1.5 fps.**
- Debug tooling: per-layer dump (--layer-dump) + tools/compare_layers.py used to localize bugs.

### M3 — fp16 fast path DONE (verified on device)
- fp16 shader variants (conv/dwconv/avgpool/fc/add _fp16): fp16 storage + fp32 accumulation,
  fp16 prepacked weights. Device buffers fp16 (2 bytes/elem). Selected via precision=fp16.
- **fp16 on Xclipse 960: cosine=0.999965, maxAbsErr=0.08, top-1 258==258 PASS,
  median 23.05 ms = 43.4 fps.** (fp32 was 24.35 ms; naive kernels are memory-bound so fp16's
  arithmetic win is modest — tiling/vectorized fp16 loads are future work, logged.)
