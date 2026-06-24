# vxrt — Limitations & Known Gaps

This document is an honest account of what vxrt (`vx::`) does **not** do, what is
stubbed, and where the verified numbers fall short of state of the art. Everything
below was measured on **one** device (see [Test coverage](#test-coverage-one-device))
unless stated otherwise. No numbers here are aspirational — they are the figures
produced by the on-device runs recorded in `docs/DEVICE_REPORT.md` and `WORKLOG.md`.

The reference workload throughout is **MobileNetV2**, fixed input image, top-1 class
258 (Samoyed), compared against an onnxruntime golden.

---

## 1. Static `batch = 1`, resolved at plan time

The engine plans a graph for a **fixed, fully-static shape**. `Session::plan()`
runs `inferShapes` once at construction; every segment, every Vulkan command buffer,
and every prepacked weight is specialized to those shapes. The Vulkan backend
literally **pre-records one `VkCommandBuffer` per segment** (`Segment` in
`include/vx/backend.h`), so there is no mechanism to vary `N`, `H`, or `W` at run
time.

Concretely:

- `batch = 1` only. Dynamic batch, dynamic spatial dims, and symbolic dimensions
  are not supported. A model with an unresolved dim will not plan.
- Changing input size means **building a new `Session`** (and paying cold-start cost,
  see below).
- This is a deliberate trade: static planning is what makes the pre-recorded
  command buffer + push-descriptor + prepacked-weight design possible. It is not a
  bug, but it is a hard constraint callers must design around.

---

## 2. ENN / NPU backend is a documented stub

`EnnBackend` (`src/backends/enn/enn_backend.cpp`, ADR-0007) is **not** a working NPU
path. It exists to prove the pluggable-backend architecture end-to-end, and to record
the gap honestly.

What it actually does:

- Subclasses `vx::Backend`, registers via `VX_REGISTER_BACKEND(BackendKind::kEnn, …)`,
  and is selectable through `config.backend = BackendKind::kEnn` with **no edits to
  core dispatch**.
- On construction it `dlopen`-probes the on-device ENN runtime libraries
  (`libenn_public_api_cpp.so`, `libenn_engine.so`, `libenn_model.so`,
  `libenn_user_driver_gpu.so`, `libenn_user_driver_unified.so`). On the target
  device **4 of the 5** resolve (`libenn_engine.so` is not reachable via `dlopen`
  from the app namespace; the vendor namespace hides it).
- `available()` returns `true`, but `supports(OpType, DType)` returns **`false` for
  every op**, so the configured fallback list (Vulkan → CPU) executes the entire
  graph.
- `compileSegment()` throws `Status::kUnsupported` with an explicit message pointing
  here.

Why it is not real, and what a real path needs:

1. **No public ENN C++ headers** are available to us.
2. **No on-device NNC compiler.** ENN consumes `.nnc` models, which are produced by
   an **offline** Samsung SDK tool we do not have.

A genuine ENN backend therefore requires the offline flow
**source ONNX → Samsung NNC compiler → `.nnc` → ENN runtime**, plus the headers.
If both become available, **only `EnnBackend`'s body changes** — registration,
selection, and fallback are already wired. See ADR-0007 and
`docs/ADDING_A_BACKEND.md`.

---

## 3. fp16 trades accuracy for ~10% speed

The fp16 path uses **fp16 storage with fp32 accumulation**. It is faster but not
bit-accurate against the fp32 / CPU reference:

| Path          | cosine vs onnxruntime | maxAbsErr | latency       |
|---------------|-----------------------|-----------|---------------|
| CPU reference | 1.000000              | —         | 672 ms / 1.5 fps |
| Vulkan fp32   | 1.000000              | 1.3e-5    | 24.35 ms / 41 fps |
| Vulkan fp16   | **0.999965**          | **0.08**  | 22.0 ms / 45.4 fps |

So fp16 (`precision = Precision::kFp16`, the default in `vx::Config`) costs roughly a
**0.000035 cosine drop and up to 0.08 absolute error** on intermediate activations,
in exchange for ~2.4 ms / ~4 fps over fp32. For accuracy-sensitive callers, set
`precision = Precision::kFp32` (still real-time at 41 fps) or fall back to CPU for a
bit-exact reference.

---

## 4. Kernels are naive and memory-bound — 22 ms is not SOTA

vxrt's compute kernels are straightforward and **deliberately unsophisticated**. The
22 ms fp16 wall time is a competent baseline, not a tuned ceiling.

Specifically, the Conv/GEMM kernels:

- Use **no Winograd** transform for 3×3 convolutions.
- Use **no cooperative-matrix / matrix-core path**. `VK_KHR_cooperative_matrix` is
  **absent on this driver** (Samsung SPAL 25.2.39), so that avenue is closed here
  regardless.
- Have **no subgroup-tiled GEMM yet** — the GEMM/FC and pointwise paths are plain
  subgroup/vec4 (`NC4HW4`) loops, not register-blocked tiled microkernels.

Conv strategies that *do* exist: a general `group = 1` kernel (handling 1×1 pointwise
and 3×3 strided) and a specialized depthwise kernel. They are correct and packed
(`NC4HW4`, channels in vec4 blocks, ADR-0004), but they are bandwidth-bound, not
compute-optimal. The conv kernel does autotune `local_size_x` via a spec constant
(20 cached entries), which helps occupancy but does not change the algorithmic class.

GPU compute time alone (from timestamp queries, `timestampPeriod` 39.0625 ns):

```
total 12.1 ms   Conv 10.7   Gemm 0.9   Add 0.26   Pool 0.09   Reshape 0.11
```

Conv dominates (10.7 of 12.1 ms), which is exactly where Winograd / a tiled GEMM
microkernel would pay off. Those are unimplemented.

---

## 5. CPU↔device pack/unpack is ~half the wall time

Of the 22 ms fp16 wall, only **12.1 ms is GPU compute**. The remaining **~11 ms** is
host-side: converting the caller's NCHW fp32 input into the internal `NC4HW4` packed
layout (and the reverse on output), plus the `toHost`/`toDevice` residency
reconciliation at segment boundaries (`Backend::toHost` / `Backend::toDevice` in
`include/vx/backend.h`).

The device is UMA (memory types are `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`, so
there are **no staging copies**), yet the pack/unpack itself is CPU work done with the
`pack` / `unpack` compute shaders and host glue. This is the single largest, most
addressable inefficiency in the pipeline today — it is larger than the entire Gemm +
Add + Pool + Reshape GPU cost combined. It has not been optimized.

> Zero-copy I/O (`enableZeroCopy`, `vx::IonBuffer` over DMA-BUF heaps) removes the
> *input/output buffer copy*, and was verified bit-identical to the staged path
> (maxAbsErr 0, both `Mode A` alloc and `Mode B` `wrapFd`). It does **not** remove the
> layout pack/unpack, which is the dominant host cost above.

---

## 6. int8 is a stretch goal — not implemented

The device advertises the capabilities for it (`shaderInt8 = 1`, 8-bit storage,
`VK_KHR_shader_integer_dot_product`), and `Config::precision` only exposes
`kFp32 | kFp16 | kAuto` — there is **no int8 enum value**. There is no quantization
pass, no int8 kernel, and no calibration tooling. int8 inference is a documented
stretch goal that was **not** built.

---

## 7. Layer-dump names map to the golden *post-Clip* tensor

With `layerDump = true`, `Session::run` (`src/core/session.cpp`) writes one `.bin`
per live, non-initializer pool tensor, named after the **IR tensor name**
(`/` and `:` rewritten to `_`). Because `fuseActivations` folds **35 Clip/Relu nodes
into the preceding Conv/Gemm**, the activation's own output tensor is consumed and the
fused producer writes the post-activation result directly.

The practical consequence: a dumped Conv-with-fused-Clip6 tensor corresponds to the
golden's **post-Clip** name, not a separate pre-activation Conv output. `tools/compare_layers.py`
matches dumps to goldens by name, so when locating a first divergence, expect the
fused layers to line up against the activation-output golden, not an (absent)
pre-activation one. This is correct behavior but can be confusing during debugging.

---

## 8. ONNX coverage is limited to the CNN ops MobileNetV2 needs

The importer (hand-rolled, dependency-free protobuf parser in
`src/import/onnx/onnx_parser.cpp`) maps a **fixed, small** op set
(`opTypeFromOnnx` in `src/core/op.cpp`). Anything not in that table imports as
`OpType::kUnknown`. There is **no** support for transformers/attention, RNNs,
dynamic control flow, training ops, or most of the broader ONNX opset.

Implemented compute ops: `Conv` (general / depthwise / pointwise), `Gemm`/FC,
`Clip`(relu6) / `Relu` (fused into the producer), `Add` (residual; broadcast handled
on CPU only), `GlobalAveragePool`, `Reshape`, `Flatten`, `Softmax`, `BatchNorm`
(CPU reference only), `Identity`.

Shape-path ops `Shape`, `Constant`, `Gather`, `Unsqueeze`, `Concat` exist but run on
**CPU**, and for the Vulkan path they are expected to be **const-folded away**
(`constFold` removed 5 shape-path nodes; the full pass pipeline reduces MobileNetV2
from 105 → 65 nodes). `AvgPool`, `MaxPool`, `MatMul`, and `Pad` have op-type entries
but are not part of the verified Vulkan path. Treat the supported set as "what
MobileNetV2-class CNNs need," not as general ONNX coverage.

---

## 9. Test coverage: one device

Every on-device number in this repo comes from a **single** unit:

- **Samsung Galaxy S26 (SM-S942B)**, Exynos 2600 (s5e9965)
- GPU **Samsung Xclipse 960** (AMD RDNA), Samsung Proprietary **SPAL driver 25.2.39**
- **Vulkan 1.4.304**, Android 16 / API 36, arm64-v8a
- adb serial `R3CY905E04M`

There is no cross-device, cross-driver, or cross-vendor validation. Key behaviors are
**driver-specific**: the absence of `VK_KHR_cooperative_matrix`, `subgroupSize = 64`,
the UMA `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` memory types (no staging), the
DMA-BUF-heap zero-copy import path (`/dev/ion` is gone; uses
`/dev/dma_heap/system`), and the autotuned workgroup sizes are all tuned to **this
GPU and this driver**. On other hardware the correctness should hold (the CPU
reference is the ground truth and is bit-exact), but the **performance numbers and
the zero-copy / capability assumptions do not transfer** and have not been retested.

---

## Summary table

| Area | Status |
|------|--------|
| Batch / shapes | Static `batch = 1`, resolved at plan time; reshape ⇒ new Session |
| ENN / NPU | Documented stub; 4/5 libs probed; no NNC compiler, no public headers; always falls back |
| fp16 | cosine 0.999965 (vs 1.0), maxAbsErr 0.08; fp16 storage + fp32 accum |
| Kernels | Memory-bound; no Winograd, no cooperative matrix (absent on driver), no tiled GEMM |
| Host overhead | ~11 ms of the 22 ms wall is CPU↔device pack/unpack |
| int8 | Not implemented (stretch goal) |
| Layer dump | Fused-activation tensors map to golden *post-Clip* name |
| ONNX ops | Only the CNN ops MobileNetV2 needs |
| Devices tested | One (Galaxy S26 / Xclipse 960 / SPAL 25.2.39) |
