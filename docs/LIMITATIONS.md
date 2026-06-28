# VKNN — Limitations & Known Gaps

This document describes what VKNN (`vknn::`) does **not** do, what is stubbed, and where
the verified numbers fall short of state of the art. Every number below is measured on
**one** device (see [Test coverage](#9-test-coverage-one-device)) unless stated otherwise.
All numbers come from on-device runs against onnxruntime goldens.

VKNN runs image CNNs (ResNet-50, MobileNetV2/V3, EfficientNet, Inception, DenseNet, ShuffleNet),
YOLOv8n detection, and the 965M-param YoNoSplat transformer encoder. Per-model latencies and the
VKNN-vs-MNN comparison are in [BENCHMARK.md](BENCHMARK.md); this document covers what the engine
does **not** do.

---

## 1. Static `batch = 1`, resolved at plan time

The engine plans a graph for a **fixed, fully-static shape**. `Session::plan()`
runs `inferShapes` once at construction; every segment, every Vulkan command buffer,
and every prepacked weight is specialized to those shapes. The Vulkan backend
**pre-records one `VkCommandBuffer` per segment** (`Segment` in
`include/vknn/backend.h`), so there is no mechanism to vary `N`, `H`, or `W` at run
time.

Concretely:

- `batch = 1` only. Dynamic batch, dynamic spatial dims, and symbolic dimensions
  are not supported. A model with an unresolved dim will not plan.
- Changing input size requires **building a new `Session`** (and paying cold-start cost,
  see below).
- This is a deliberate trade: static planning is what makes the pre-recorded
  command buffer + push-descriptor + prepacked-weight design possible. It is a hard
  constraint callers must design around.

---

## 2. No NPU / accelerator backend (Vulkan + CPU only)

The backends are **Vulkan** (the on-device compute path) and **CPU** (host oracle +
fallback). There is no NPU / vendor-accelerator backend. Such accelerators usually
consume a model artifact built by an **offline**, host-side toolchain rather than
JIT-compiling from ONNX on device, and that toolchain (plus matching public headers)
is not available for the target hardware.

The pluggable-backend architecture supports one: adding a backend is a new
`Backend` subclass + `VKNN_REGISTER_BACKEND`, with no edits to core dispatch — see
`docs/ADDING_A_BACKEND.md`, which documents the offline-compiled-accelerator pattern.

---

## 3. fp16 trades accuracy for speed

The default GPU path uses **fp16 storage with fp32 accumulation**. It is faster but not
bit-accurate against the fp32 / CPU reference — cosine vs onnxruntime lands in the
**0.9995–1.0** range across the benchmarked models (e.g. MobileNetV3 0.99954, ResNet-50 and
YOLOv8n 1.000000), with small absolute error on intermediate activations. The Vulkan fp32 path is
bit-close (cosine 1.0, maxAbsErr ~1e-5), and the CPU backend is the bit-exact reference.

For accuracy-sensitive callers, set `precision = Precision::Fp32` or fall back to CPU. fp16 is the
default in `vknn::Config` because the accuracy cost is small and the bandwidth saving is real.

---

## 4. Conv kernels trail a years-tuned engine on the 3×3-heavy nets

VKNN beats MNN's Vulkan backend on every benchmarked model (often ~4×). Against MNN's
**OpenCL HEAVY-tuned** best, VKNN is faster on 8 of 9 models and at **parity on the 9th, ResNet-50**.
The remaining ResNet edge appears only under a warm device.

- **Winograd F(2,3) via a tiled GEMM** is the default for deep/square 3×3 convs (`Config::winograd =
  Auto`, autotuned vs the direct kernel per shape).
- **No cooperative-matrix / matrix-core path.** `VK_KHR_cooperative_matrix` is **absent on the
  target driver**, so that avenue is closed.
- **F(4,3) Winograd** is implemented (numerically fine at fp16) but slower here — its 6×6 transforms
  are register-heavy; available via `setHint(Hint::WinogradUnit, 4)` for research.

The proven kernels are the tiled-GEMM Winograd 3×3, a direct 3×3, a register-tiled (WTILE=4) 1×1,
an untiled depthwise, and **split-K** for deep low-parallelism 1×1 convs. Other restructurings that add
register/LDS/occupancy pressure (register-tiled 3×3, LDS input-halo, naive-matmul Winograd, packed-math)
regress on this driver — it punishes occupancy pressure, and the 3×3 weights already L2-cache, so
cutting weight reads does not cut DRAM traffic. Matching MNN on ResNet/YOLO requires a production
fused-cooperative Winograd, a large kernel. See [BENCHMARK.md](BENCHMARK.md).

---

## 5. CPU↔device pack/unpack at the I/O boundary

Converting the caller's NCHW fp32 input into the internal `NC4HW4` packed layout (and the reverse on
output), plus the `toHost`/`toDevice` residency reconciliation at segment boundaries
(`Backend::toHost` / `Backend::toDevice` in `include/vknn/backend.h`), is host-side overhead. On
small CNNs where GPU compute is only a few milliseconds, this boundary work is a large fraction of
the wall time.

The device is UMA (memory types are `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`, so there are **no
staging copies**), but the pack/unpack itself is CPU work. Feeding NC4HW4 directly, or doing the
conversion on the GPU, would remove most of it. It is not optimized.

> Caller-owned DMA-BUF I/O (`Tensor::fromDmaBuf` / `Tensor::toDmaBuf`, binding model I/O to
> a caller-provided dma-buf fd that `vknn::IonBuffer::wrapFd` mmaps) removes the caller-side
> *I/O buffer / copy*, and is verified bit-identical to the staged path (maxAbsErr 0). It does
> **not** remove vknn's internal layout pack/unpack (NCHW fp32 ↔ device NC4HW4/fp16), which is
> the dominant host cost above.

---

## 6. int8 is a stretch goal — not implemented

The device advertises the capabilities for it (`shaderInt8 = 1`, 8-bit storage,
`VK_KHR_shader_integer_dot_product`), and `Config::precision` only exposes
`Fp32 | Fp16 | Auto` — there is **no int8 enum value**. There is no quantization
pass, no int8 kernel, and no calibration tooling. int8 inference is a documented
stretch goal that is **not** built.

---

## 7. Layer-dump names map to the golden *post-Clip* tensor

With `layerDump = true`, `Session::run` (`src/core/session.cpp`) writes one `.bin`
per live, non-initializer pool tensor, named after the **IR tensor name**
(`/` and `:` rewritten to `_`). Because `fuseActivations` folds **35 Clip/Relu nodes
into the preceding Conv/Gemm**, the activation's own output tensor is consumed and the
fused producer writes the post-activation result directly.

In practice: a dumped Conv-with-fused-Clip6 tensor corresponds to the
golden's **post-Clip** name, not a separate pre-activation Conv output. `tools/compare_layers.py`
matches dumps to goldens by name, so when hunting a first divergence, the
fused layers line up against the activation-output golden, not an (absent)
pre-activation one. This is correct behavior, but it can be confusing while debugging.

---

## 8. ONNX coverage is a fixed op table (broad, but not the whole opset)

The importer (hand-rolled, dependency-free protobuf parser in `src/import/onnx/onnx_parser.cpp`) maps
a **fixed** op set (`opTypeFromOnnx` in `src/core/op.cpp`). Anything not in that table imports as
`OpType::Unknown` and will not plan.

The supported set is broad — it covers CNNs, detection, **and** transformer/attention models:
convolution/pooling, the full elementwise unary/binary families, MatMul (batched N-D), Gemm,
LayerNorm, Softmax (channel + last-axis), Einsum, RoPE, Gather/Scatter, and the shape/data-movement
ops. The full table with per-op GPU/CPU coverage is in [OP_COVERAGE.md](OP_COVERAGE.md).

**Not** supported: RNN/LSTM/GRU, dynamic control flow (`Loop` / `If` / `Scan`), training ops, sparse
tensors, and the long tail of the ONNX opset. Adding an op is mechanical (see
[ADDING_AN_OPERATOR.md](ADDING_AN_OPERATOR.md)); until it is in the table the model will not
import.

---

## 9. Test coverage: one device

Every on-device number in this repo comes from a **single** unit:

- An **Android arm64-v8a** device with an **AMD RDNA-class mobile GPU**
- The GPU's **proprietary Vulkan driver**, **Vulkan 1.3+**

There is no cross-device, cross-driver, or cross-vendor validation. Key behaviors are
**driver-specific**: the absence of `VK_KHR_cooperative_matrix`, `subgroupSize = 64`,
the UMA `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` memory types (no staging), the
caller-owned dma-buf import path (`/dev/ion` is gone; fds come from
`/dev/dma_heap/system`), and the autotuned workgroup sizes are all tuned to **this
GPU and this driver**. On other hardware the correctness holds (the CPU
reference is the ground truth and is bit-exact), but the **performance numbers and
the zero-copy / capability assumptions do not transfer** and are not retested.

---

## Summary table

| Area | Status |
|------|--------|
| Batch / shapes | Static `batch = 1`, resolved at plan time; reshape ⇒ new Session |
| NPU / accelerator | None; Vulkan + CPU only (pluggable — see ADDING_A_BACKEND.md) |
| fp16 | cosine 0.9995–1.0 across models; fp16 storage + fp32 accum |
| Kernels | Beats MNN-Vulkan everywhere; trails MNN-OpenCL-tuned on ResNet/YOLO (3×3 convs); no Winograd/coopmat/tiled GEMM |
| Host overhead | NC4HW4 pack/unpack at the I/O boundary (a large fraction on small CNNs) |
| int8 | Not implemented (stretch goal) |
| Layer dump | Fused-activation tensors map to golden *post-Clip* name |
| ONNX ops | See OP_COVERAGE.md |
| Devices tested | One (Android arm64-v8a, AMD RDNA-class mobile GPU) |
