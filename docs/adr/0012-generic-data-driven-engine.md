# ADR-0012: generic data-driven engine (dynamic shapes, quantized path, full GPU parity)

## Status
Accepted (2026-07-07). Supersedes ADR-0011 §5 (grouped Conv now runs on the GPU via import
lowering). Hardens ADR-0006 (segment execution) with a plan-lifetime invariant. Extends ADR-0009
(caches) with per-shape plan buckets.

## Context
The engine already satisfied most of the "generic, data-driven, extensible" goal: an ND dtyped
`TensorDesc` IR, `VkOpRegistry`/`CpuOpRegistry` factory registries with virtual `prepare`/`record`/
`run`, a `Graph` DAG built by the ONNX importer, geometry carried through push/specialization
constants, and record-once command buffers with a pipeline cache. An audit against the five
guidelines found the real gaps were narrower than a rewrite: shapes froze at compile time
(dynamic dims silently collapsed to 1), quantized checkpoints did not import at all, `matmul_tiled`
used a fixed tile, and a set of operators and operator configurations fell back to the CPU backend.
This ADR records the decisions that closed those gaps. The specialized NC4HW4 fp16 kernels are the
product; the work genericized the plumbing around them, never flattening them into ND kernels.

## Decisions

### 1. Runtime dynamic shapes are declared plan buckets, not inferred at run
`inferShapes` resolves each input's negative dims from a declared shape (`PassOptions::inputShapes`,
`vknn_compile --shape NAME=D0xD1x...`, or `Config::inputShapes`), falling back to the batch axis;
an undeclared dynamic non-batch axis is a hard compile error naming the axis, never a silent 1×1
plan. A model may declare several buckets (`--bucket`), each a full pass+plan product for one input
shape. The container (VXM4) stores per-bucket graphs over one content-deduped initializer pool;
a single-bucket file is byte-identical to the legacy VXM3. A session holds `PlanBucket{key, graph,
segments, pool}` over shared backends; `run()` selects the bucket by an exact input-shape match.
A fixed-shape model has exactly one bucket and takes a zero-key fast path — no allocation, no
lookup string, no re-record — so the hot path is unchanged. ONNX-loaded sessions retain the
pristine imported graph and add a bucket at runtime through `Session::prepareShapes()` (explicit,
never implicit on `run()`). Weights are shape-independent and shared across buckets through a
backend-level device weight pool keyed (initializer id, pack-kind, precision).

### 2. Quantized checkpoints run dequantized to float, with the saturation clamp preserved
`dequantizeGraph` (default on, `--no-dequantize` opts out) rewrites the quantized operator family
at import so QDQ, QLinear, and dynamic-quant checkpoints run on the existing fp16 engine:
`DequantizeLinear` over an initializer folds to an fp32 weight `(W_q − zp)·scale`; a matching
`QuantizeLinear→DequantizeLinear` activation sandwich collapses; `QLinearConv/MatMul/QGemm/Add/
GlobalAveragePool` decompose to their float op with folded weights; the `DynamicQuantizeLinear +
MatMulInteger/ConvInteger + Cast + Mul` cluster lowers to a float `MatMul`/`Conv`.

The load-bearing decision: **dropping a quantize round-trip is not an identity.** A `Q→DQ`
composition equals `clamp(x, (qmin−zp)·scale, (qmax−zp)·scale)` — dropping only the 8-bit rounding.
On checkpoints where the quantizer fused a ReLU into an activation's quant range, collapsing the
sandwich to raw float deletes the nonlinearity and the activations diverge (proven: a mobilenetv2
QDQ export went to cosine −0.14 / logits ±5e6 before this was fixed). So the collapse and the
QLinear decompositions insert a `Clip` to the dequantized `[qmin, qmax]` range: rounding dropped,
clamp preserved (`0 · −65504 = 0`, and the clamp is a no-op when the range does not saturate).
Dynamic-quant clusters get no clamp — `MatMulInteger`/`ConvInteger` emit a full-range int32
accumulation with no output quant range, so they are genuinely rounding-only.

This is dequantized execution, not int-exact quantized inference: the honest gate is CPU-vs-GPU
byte self-consistency plus SNR/cosine against onnxruntime, never a byte match to an int-exact
reference. Int8 storage and int8-dot compute kernels are a later project; the device
`shaderIntegerDotProduct` feature is enabled and probed but no int8 kernel ships yet.

### 3. `matmul_tiled` tiles are specialization constants raced per shape
`TM`/`TN`/`TK` become `constant_id`s; `MatMulOp::prepare` races a small candidate set per shape
under `--tuning fast/heavy` and caches the winning tile index by a shape-keyed signature, exactly
like the conv autotuners. Every candidate is bit-neutral — per-output accumulation is one
k-ascending fp32 chain regardless of tiling — so `--tuning none` keeps the default tile and today's
bytes, and the race needs no accuracy margin. Measured −11% GPU time on the Whisper encoder.

### 4. Every executable operator has a Vulkan kernel
The support surface, emitted by the engine itself (`vknn_compile --support-report`,
`Session::fallbackReasons`, the `vk_gates.cpp` gate that device `supports()` and the tool share so
they cannot drift), shows every compute node on Vulkan across the benchmark zoo. Closing the gaps:
GPU `QuantizeLinear`/`DequantizeLinear` (bit-identical to the CPU op), `Cast` from int64 (decoded
at the pack boundary; narrow-integer targets match the CPU op's modulo truncation), `TopK`
(per-slice sort, indices tie-broken on ascending source index), general grouped Conv (lowered to
per-group dense Convs — the Slice materializes a contiguous channel slice that repacks into clean
NC4HW4 blocks, dissolving the packing hazard that ADR-0011 §5 gated off; pure depthwise and
group==1 keep their native kernels), runtime-operand paths for Clip/BatchNorm/LayerNorm/
ConvTranspose/Pad (parameter read from an SSBO instead of a baked constant), and integer variants
of ConstantOfShape/Range/Cast.

Two classes stay off the GPU by design: **control-flow ops** (`Loop`/`If`/`NonMaxSuppression`) and
anything whose output shape is data-dependent (runtime pads-geometry, runtime `TopK` k, unresolved
shapes) — a static per-shape plan cannot size their buffers; and **const-folded import ops**
(`Shape`/`Constant`/`Identity`/`EyeLike`) — eliminated at import, they never execute on any
backend. Narrow-integer `Cast` targets (int16/uint16/bool) and rank>8 flat tensors remain on the
CPU op; no benchmark model exercises them, and both are documented in `docs/limitations.md`.

### 5. Correctness invariants restored during the work
Fixing the above surfaced latent bugs whose fixes are part of this ADR: the per-shape plan bucket
held its `Graph` by value, so moving it into the session vector left every segment's `Graph&`
dangling — a use-after-free that crashed Vulkan segments with boundary inputs (any GPU model);
`PlanBucket::graph` is now a `unique_ptr` with a `segmentGraphsLive()` invariant. An int64/int32
boundary tensor crossing into a Vulkan segment is decoded to fp32 at the pack boundary (raw int
bytes reinterpreted as fp32 read as ~0). An out-of-fp16-range constant is clamped to the fp16
finite extreme at conversion (`−3.4e38 → −65504`), so an attention mask's `(1−mask)·bias` computes
`0 · −65504 = 0` instead of `0 · −inf = NaN`. Forward Conv and the pools now resolve `auto_pad`
through shared `conv_geom.h` geometry; 1-D Conv is normalized to canonical 2-D at import; the
reduce/split/pad/resize CPU ops derive geometry from runtime shapes; rank-0 scalars count as one
element through `cpu::elemCount`.

## Gates (host, and both devices — Xclipse 960 + 940)
- Host: `./build.sh`, `./build-host/vknn_tests`, `./build.sh --android` green.
- Fixed-shape byte invariant: a fixed-shape `.vxm` and its output are byte-identical before/after
  the dynamic-shape work; the single-bucket path adds no allocation or re-record.
- Quantized clamp recovery: the mobilenetv2 QDQ export runs cosine 0.999 vs onnxruntime (from
  −0.14), argmax restored; all five quantized zoo models compile with zero unsupported nodes at
  cosine 0.997–0.999 vs their int-exact references.
- GPU parity: the `--support-report` oracle shows zero CPU fallbacks across the benchmark zoo;
  each closed gate is verified GPU-vs-CPU on device (Q/DQ bit-identical, grouped conv and the
  runtime-operand/int-variant paths cosine ~1.0).
- Device GPU restored: after the plan-lifetime fix, mnasnet runs GPU-vs-CPU cosine 0.999989.

## Consequences
- VXM4 is additive: single-bucket files are byte-identical VXM3 and load unchanged; a multi-bucket
  file adds a per-bucket section over a shared weight pool.
- A fixed-shape model pays nothing for the bucket machinery; dynamic shapes are opt-in via declared
  shapes or `prepareShapes`.
- The CPU backend is the byte/numeric oracle for every GPU kernel; it must stay a complete,
  correct fallback (ConvertLayout gained a CPU kernel for this reason).
- Quantized accuracy is dequantized-execution accuracy, stated as such in the docs and tools — it
  is not, and does not claim to be, int-exact.
- Adding an operator now also touches the shared `vk_gates.cpp` capability gate and `gpuFlatNode`;
  `docs/adding-an-operator.md` lists these.
