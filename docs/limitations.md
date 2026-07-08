# VKNN — Limitations & Known Gaps

This document describes what VKNN (`vknn::`) does **not** do, what is stubbed, and where
the verified numbers fall short of state of the art. Every number below is measured on
**one** device (see [Test coverage](#9-test-coverage-one-device)) unless stated otherwise.
All numbers come from on-device runs against onnxruntime goldens.

VKNN runs image CNNs (ResNet-50, MobileNetV2/V3, EfficientNet, Inception, DenseNet, ShuffleNet),
YOLOv8n detection, and the 965M-param YoNoSplat transformer encoder. Per-model latencies and the
VKNN-vs-MNN comparison are in [benchmark.md](benchmark.md); this document covers what the engine
does **not** do.

---

## 1. Shapes are resolved at plan time; dynamic shapes need declared buckets

The engine plans a graph for a **fixed, fully-static shape**. `Session::plan()`
runs the import passes' shape inference to a fixed point at construction; every
segment, every Vulkan command buffer, and every prepacked weight is specialized to
those shapes. The Vulkan backend **pre-records the segment's `VkCommandBuffer`s**
(`Segment` in `include/vknn/segment.h`; a segment above `Config::maxSubmitNodes`
records as multiple chunked submits), so a single plan cannot vary `N`, `H`, or `W`
at run time. A shape that was not planned has no plan — the engine never re-plans
implicitly inside `run()`.

Dynamic shapes are supported by declaring the shapes ahead of time as **plan
buckets**: each bucket is a full pass-and-plan product for one input-shape set, and
`run()` selects the bucket by the bound input shapes (an exact-match map lookup; an
unmatched shape is `Status::InvalidArgument` listing the available buckets). Buckets
share the backends, caches, pipeline pool, and device weight pool, so a second bucket
reuses tuning and uploaded weights.

- **Compile-time buckets:** `vknn_compile --bucket "NAME=D0xD1x...;NAME2=..."`
  (repeatable) compiles the model once per bucket into one multi-bucket `.vxm` (see
  [config.md](config.md)). A `.vxm` session dispatches among its stored buckets and
  cannot add more at run time. The `--graph "FILE.onnx[;segments]"` form generalizes
  a bucket to its **own source graph**: several graphs (a vision tower + a decoder's
  prefill/decode plans) compile into one `.vxm` over a content-deduped weight pool,
  and `run()` dispatches to the bucket matching the bound input **names + shapes**
  ([running-a-vlm.md](running-a-vlm.md), [ADR-0014](adr/0014-multi-graph-vxm.md)).
- **Runtime buckets (ONNX-loaded sessions only):** `Session::prepareShapes(shapes)`
  re-runs the passes and plan from the retained pristine graph to add a bucket
  explicitly. It never happens implicitly on `run()`. A `.vxm` session returns
  `Status::Unsupported`.
- **Declaring a symbolic axis:** a dynamic **batch** axis — a leading axis that is
  unnamed or batch-named (`N`/`B`/`*batch*`, case-insensitive) — falls back to
  `batch = 1` (or `--batch N`). Any **other** dynamic axis — spatial / feature, or a
  leading axis with a non-batch `dim_param` name like `num_frames` — that stays
  unresolved is a **hard error** — one aggregated message listing the unbound
  `dim_param` symbol names — rather than a silent freeze to 1. Resolve it by
  **binding the symbol** with `vknn_compile --dim NAME=VALUE` / `Config::dimBindings`
  (the compiler evaluates each input axis's `dim_param`, including a compound like
  `past_sequence_length + sequence_length`, so a with-past decoder needs a couple of
  bindings), or declare a full per-tensor shape with `--shape NAME=D0xD1x...` /
  `--bucket` / `Config::inputShapes`. `--list-dims` prints the free symbols to bind.

The **fixed-shape path is unchanged and zero-cost**: a model with no dynamic axes (or
only a dynamic batch) plans exactly **one** bucket, its `.vxm` bytes are byte-identical
to before this feature, and `run()` takes a fast path that builds no shape key and adds
no allocation, pipeline creation, or re-record. Static planning is what makes the
pre-recorded command buffer + push-descriptor + prepacked-weight design possible; a
bucket is a second such plan, not a per-run reshape.

---

## 2. No NPU / accelerator backend (Vulkan + CPU only)

The backends are **Vulkan** (the on-device compute path) and **CPU** (host oracle +
fallback). There is no NPU / vendor-accelerator backend. Such accelerators usually
consume a model artifact built by an **offline**, host-side toolchain rather than
JIT-compiling from ONNX on device, and that toolchain (plus matching public headers)
is not available for the target hardware.

The pluggable-backend architecture supports one: adding a backend is a new
`Backend` subclass + `VKNN_REGISTER_BACKEND`, with no edits to core dispatch — see
`docs/adding-a-backend.md`, which documents the offline-compiled-accelerator pattern.

---

## 3. fp16 trades accuracy for speed

The default GPU path uses **fp16 storage with fp32 accumulation**. It is faster but not
bit-accurate against the fp32 / CPU reference — cosine vs onnxruntime lands in the
**0.9995–1.0** range across the benchmarked models (e.g. MobileNetV3 0.99954, ResNet-50 and
YOLOv8n 1.000000), with small absolute error on intermediate activations. The Vulkan fp32 path is
bit-close (cosine 1.0, maxAbsErr ~1e-5), and the CPU backend is the bit-exact reference.

For accuracy-sensitive callers, set `precision = Precision::High` or fall back to CPU. fp16 is the
default in `vknn::Config` because the accuracy cost is small and the bandwidth saving is real.

An fp16 activation store **saturates to the finite extreme** (`±65504`) rather than overflowing to
`±inf`, so a finite fp32 accumulation past the fp16 range never becomes a `NaN` that silently zeroes
the output — the same clamp already applied to out-of-fp16-range constants. Saturation is not free
accuracy, though: a model with **no normalization ops** whose activations genuinely exceed the fp16
range (a deep conv stack that relies on trained weights to bound its intermediates) loses magnitude
information at the clamp. Such a model wants fp32 storage — `Precision::High`, or `Precision::Normal`
with its tensors named in `Config::fp32Tensors` — for full accuracy; at `Low` it stays finite but
approximate on the overflowing tensors.

---

## 4. Conv kernels trail a years-tuned engine on the 3×3-heavy nets

VKNN beats MNN's Vulkan backend on every benchmarked model (often ~4×). Against MNN's
**OpenCL HEAVY-tuned** best, VKNN is faster on 8 of 9 models and trails on the 9th,
**ResNet-50 (~15%)** — MNN's CLBlast-autotuned batched GEMM (fp16 accumulation) wins
the 3×3-conv bulk there.

- **Winograd F(2,3) via a tiled GEMM** is the default for deep/square 3×3 convs
  (`setHint(Hint::Winograd, Mode::Auto)`, autotuned vs the direct kernel per shape).
- **No cooperative-matrix / matrix-core path.** `VK_KHR_cooperative_matrix` is **absent on the
  target driver**, so that avenue is closed.
- **F(4,3) Winograd** is implemented (numerically fine at fp16) but slower here — its 6×6 transforms
  are register-heavy; available via `setHint(Hint::WinogradUnit, 4)` for research.

The proven kernels are the tiled-GEMM Winograd 3×3, a direct 3×3, a register-tiled (WTILE=4) 1×1,
an untiled depthwise, and **split-K** for deep low-parallelism 1×1 convs. Other restructurings that add
register/LDS/occupancy pressure (register-tiled 3×3, LDS input-halo, naive-matmul Winograd, packed-math)
regress on this driver — it punishes occupancy pressure, and the 3×3 weights already L2-cache, so
cutting weight reads does not cut DRAM traffic. Matching MNN on ResNet/YOLO requires a production
fused-cooperative Winograd, a large kernel. See [benchmark.md](benchmark.md).

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

## 6. Quantized models run dequantized to float — not int-exact

There is no int8 compute tier: the device advertises the capabilities for it
(`shaderInt8 = 1`, 8-bit storage, `VK_KHR_shader_integer_dot_product`) and
`Config::precision` only exposes `Low | Normal | High` (fp16 / fp16 + selective
fp32 / fp32), but no kernel computes in int8. Instead, a quantized checkpoint runs
through the **import-time dequantize pass** (`src/import/dequantize_graph.cpp`,
default on; `PassOptions::dequantize` / `--no-dequantize` disables it):

- DequantizeLinear over an initializer folds the weight to fp32 (`(W_q − zp) · scale`,
  per-tensor and per-axis).
- The fused QLinear family (QLinearConv / QLinearMatMul / QGemm / QLinearAdd /
  QLinearGlobalAveragePool) lowers to its plain float op plus a `Clip` to the op's own
  output quant range.
- A QuantizeLinear→DequantizeLinear activation sandwich collapses to a `Clip` over the
  quant range, not to a raw passthrough.

**Dequantized execution drops the 8-bit rounding but preserves the saturation clamp.**
Collapsing a quantize round-trip is *not* an identity: `QuantizeLinear` composed with
`DequantizeLinear` (ignoring rounding) equals `clamp(x, (qmin − zp)·scale,
(qmax − zp)·scale)`, and ORT's QDQ quantizer folds a preceding ReLU into that range
(lower bound 0). Dropping the round-trip to raw float would silently delete that ReLU
and let activations explode, so the pass always keeps the clamp as a `Clip` with
constant bounds (pointwise-fusable, GPU-eligible) and only drops the grid snap. Results
are therefore **close to but not bit-identical** with an int-exact runtime — the residual
is the interior per-tensor requantization that a float chain does not re-apply
(mobilenetv2_int8 cosine ≈ 0.96, resnet50_int8 ≈ 0.998 vs the ORT int-exact goldens,
argmax matching).

Both static QDQ / QLinear **and** the common dynamic-quantization pattern are lowered.
A dynamic-quant export emits a `DynamicQuantizeLinear → MatMulInteger / ConvInteger →
Cast → Mul` cluster (the canonical shape ONNX Runtime's dynamic quantizer produces for
BERT / GPT / ViT); the pass matches that cluster and lowers it to a plain float `MatMul`
/ `Conv` with the weight folded to fp32, dropping the `DynamicQuantizeLinear`, both
`Mul`s, and the `Cast`. Unlike the static QLinear path, **no output clamp is inserted**:
the integer matmul has no output quant range, so the only difference from an int-exact
runtime is the activation rounding `DynamicQuantizeLinear` would have applied. A cluster
that does *not* match this shape (a `MatMulInteger` whose int32 output escapes to another
consumer, a non-initializer weight, a missing `Cast`/`Mul`) is left intact, and — having
no backend kernel — fails at planning. There is no calibration tooling.

---

## 7. Layer-dump names map to the golden *post-Clip* tensor

With `layerDump = true`, `Session::run` (`src/core/session.cpp`) writes one `.bin`
per live, non-initializer pool tensor, named after the **IR tensor name**
(`/` and `:` rewritten to `_`). Because `fuseActivations` folds Clip/Relu nodes
into the preceding Conv/Gemm/Add, and `fusePointwiseChains` (default on) folds whole
pointwise chains into the producing kernel's epilogue, a fused op's intermediate
tensors are consumed and the producer writes the post-chain result directly.

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
LayerNorm, Softmax (channel + last-axis), Einsum, RoPE, Gather/Scatter, generator ops
(Range / ConstantOfShape / EyeLike, const-folded), and the shape/data-movement
ops. The full table with per-op GPU/CPU coverage is in [op-coverage.md](op-coverage.md).

**Not** supported: RNN/LSTM/GRU, data-dependent control flow (`Loop` / `If` / `Scan` /
`NonMaxSuppression` — their output shapes are not known at plan time), training ops, sparse
tensors, and the long tail of the ONNX opset. Adding an op is mechanical (see
[adding-an-operator.md](adding-an-operator.md)); until it is in the table the model will not
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
| Batch / shapes | Resolved at plan time. Dynamic shapes supported via **declared plan buckets** (`--bucket` at compile, `Session::prepareShapes()` at run on ONNX sessions); fixed-shape path unchanged and zero-cost (one bucket). A dynamic non-batch axis with no declared shape is a hard error, not a silent `1x1` plan |
| NPU / accelerator | None; Vulkan + CPU only (pluggable — see adding-a-backend.md) |
| fp16 | cosine 0.9995–1.0 across models; fp16 storage + fp32 accum |
| Kernels | Beats MNN-Vulkan everywhere; trails MNN-OpenCL-tuned on ResNet-50 (~15%, CLBlast-autotuned GEMM); tiled-GEMM Winograd F(2,3) is the default; no coopmat path (extension absent on the target driver) |
| Host overhead | NC4HW4 pack/unpack at the I/O boundary (a large fraction on small CNNs) |
| Quantized models | Static QDQ / QLinear **and** the canonical dynamic-quant cluster run dequantized to float (static: clamps preserved, rounding dropped — not int-exact; dynamic: folded to float MatMul/Conv, no output clamp); a non-canonical dynamic-quant cluster fails at planning; no int8 compute tier |
| Layer dump | Fused-activation tensors map to golden *post-Clip* name |
| ONNX ops | See op-coverage.md |
| Devices tested | One (Android arm64-v8a, AMD RDNA-class mobile GPU) |
