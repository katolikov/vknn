# vxrt — Summary

**vxrt** (Vulkan-eXynos RunTime, namespace `vx`) is a from-scratch, on-device CNN inference
engine for Android/Exynos with a Vulkan compute backend, ONNX import, a pluggable multi-backend
architecture, ION/DMA-BUF zero-copy, and a full profiling/debug toolkit. Everything below was
**verified on real hardware** over `adb`.

## Device it was verified on (probed, not assumed)
Samsung **Galaxy S26 (SM-S942B)** · Exynos **2600 (s5e9965)** · GPU **Xclipse 960** (AMD RDNA,
Samsung SPAL driver 25.2.39) · **Vulkan 1.4.304** · Android 16 / API 36 · arm64-v8a.
Full probe in [`docs/DEVICE_REPORT.md`](docs/DEVICE_REPORT.md).

## What works, verified on-device (with numbers)

| Capability | Result on Xclipse 960 |
|---|---|
| **MobileNetV2 on GPU (Vulkan, fp32)** | top-1 **258** (==onnxruntime), cosine **1.000000**, maxAbsErr 1.3e-5, **24.35 ms / 41 fps** |
| **MobileNetV2 on GPU (Vulkan, fp16)** | top-1 258, cosine **0.999965**, maxAbsErr 0.08, **22.0 ms / 45.4 fps** |
| MobileNetV2 on CPU reference | cosine **1.000000** vs onnxruntime, 672 ms / 1.5 fps |
| Per-op GPU profiler (timestamp queries) | GPU compute 12.1 ms (Conv 10.7, Gemm 0.9, Add 0.26, Pool 0.09, Reshape 0.11) |
| NEON CPU fallback + warning | forcing 2 ops off → 23 Vulkan/CPU segments, warnings logged, output still cosine **1.000000** |
| ION/DMA-BUF zero-copy | dma-heap alloc + dma-buf fd imported into Vulkan; GPU compute on the ION buffer == staged (maxAbsErr **0.0**), **both Mode A and Mode B** |
| Backend selectable via config | VULKAN / CPU / ENN all run; ENN consulted then falls back (stub); all top-1 258 |
| Caches & autotuning | cold session **445 ms** (first run + full autotune) → **warm 68 ms** (**6.5×**); 106 weights + 20 tuning entries cached |
| Host unit + integration tests | 7/7 pass (incl. full MobileNetV2 CPU vs golden, cosine ≥ 0.999) |

## Milestones (all green, committed)
- **M0** project skeleton + Vulkan probe — capabilities read on-device match `vkjson`.
- **M1** Vulkan compute foundation — GPU add == CPU, pipeline cache produced & reused.
- **M2** ONNX import (hand-rolled protobuf) + IR + CPU reference backend — MobileNetV2 matches ORT.
- **M3** Vulkan MobileNetV2 (NC4HW4) fp32 **and** fp16 — the vertical slice, on the GPU.
- **M4** per-op profiler (GPU timestamps + CPU wall), JSON + Chrome trace, per-layer dump.
- **M5** NEON CPU fallback with throttled warning + seamless cross-backend hand-off.
- **M6** ION zero-copy (DMA-BUF heap + `VkImportMemoryFdInfoKHR`), both modes.
- **M7** pluggable ENN/NPU backend (documented stub, config-selectable, probes libs).
- **M8** pipeline + prepacked-weights + autotune caches → fast warm sessions.
- **M9** docs (Doxygen + handwritten), tests, examples, build/run scripts.

## Architecture (one line)
ONNX → hand-rolled protobuf parser → backend-agnostic **NCHW IR** → passes (shape-infer,
BatchNorm-fold, activation-fusion [35], constant-fold [5], DCE; 105→65 nodes) → **Session**
partitions into same-backend **segments** → backends: **Vulkan** (NC4HW4, one pre-recorded command
buffer/segment, push descriptors, fp16) · **CPU** (scalar reference + NEON) · **ENN** (stub).
Adding an operator or a backend needs **no core edits** (self-registering, whole-archive linked).
Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## What is stubbed / not done (honest)
- **ENN/NPU backend is a documented stub.** The ENN runtime libs are present (4/5 probed) but
  there is **no on-device NNC compiler** and no public ENN headers, so a real ENN inference path
  cannot be built here. It registers, is selectable, probes the libs, and falls back. The offline
  flow (ONNX → Samsung NNC compiler → `.nnc` → ENN runtime) is documented. See
  [`docs/adr/0007`](docs/adr/0007-enn-backend-stub.md) and [`LIMITATIONS.md`](docs/LIMITATIONS.md).
- **INT8 quantization** — designed for (DType enum, int8 dot-product probed) but not implemented (stretch goal).
- **Static batch = 1** (resolved at plan time for static-graph pre-recording).
- **Kernels are correct but not yet state-of-the-art.** They are straightforward NC4HW4 kernels;
  no Winograd, no subgroup-tiled GEMM, and cooperative matrix is absent on this driver. ~11 ms of
  the 22 ms wall is CPU↔device pack/unpack at the I/O boundary.

## Concrete next steps (highest-leverage first)
1. **Cut the I/O boundary cost** — pack/unpack on the GPU (the `pack`/`unpack` shaders already
   exist) or feed NC4HW4 directly via ION zero-copy input, removing ~11 ms of CPU work.
2. **Faster conv** — subgroup-tiled / register-blocked pointwise GEMM, Winograd 3×3 depthwise,
   wider fp16 vec loads; expand autotuning to tile sizes.
3. **INT8** — quantized conv using `VK_KHR_shader_integer_dot_product` (probed-accelerated).
4. **Memory planning** — reuse activation buffers by lifetime to cut footprint/allocations.
5. **Real ENN path** once an offline NNC compiler is available.

## Reproduce
```bash
python3 scripts/get_golden.py --image assets/cat.jpg   # onnxruntime golden (host)
./scripts/build_android.sh                              # NDK arm64-v8a build
./scripts/run_on_device.sh vulkan fp16                  # push + run + verify on the phone
./scripts/bench.sh 50                                   # latency + cold/warm session numbers
cmake -S . -B build-host -DVXRT_ENABLE_VULKAN=OFF && cmake --build build-host && ./build-host/vx_tests
```
See [`README.md`](README.md) and [`WORKLOG.md`](WORKLOG.md) for the full story.
