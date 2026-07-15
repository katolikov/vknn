# VKNN Architecture

`VKNN` (Vulkan Neural Network, namespace `vknn`) is a small C++17 inference runtime
that loads an ONNX model — image CNNs, detection nets, and transformer/attention
models, including statically-quantized checkpoints — lowers it to a backend-agnostic
NCHW IR, optimizes it with graph passes, partitions it into backend-specific
*segments*, and executes those segments — primarily on a Vulkan compute backend tuned
for a mobile integrated GPU, with a CPU backend that doubles as the reference path
and the fallback. Nearly every executable op has a Vulkan kernel; the const-fold / import-time ops
(`Identity` / `Constant` / `Shape` / `EyeLike`) fall to the CPU, and data-dependent control flow
(`Loop` / `If` / `NonMaxSuppression`) has no `OpType` at all — a model containing one fails at
planning rather than falling back.

This document covers the end-to-end pipeline, the core abstractions, the internal
`NC4HW4` tensor layout, the segment execution model (which provides both pre-recorded
Vulkan command buffers *and* transparent CPU fallback), and the cache subsystem. It
references source under `include/vknn/` and `src/`.

---

## 1. End-to-end pipeline

```
   model.onnx
       │
       ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ IMPORT                                                                  │
 │   src/import/onnx/onnx_parser.cpp  — hand-rolled, dependency-free       │
 │   protobuf parser                                                       │
 │   importOnnx(path)  →  Graph  (backend-agnostic NCHW IR)                │
 │   (include/vknn/graph.h)                                                │
 └───────────────────────────────────────────────────────────────────────┘
       │   Graph { tensors[], nodes[], inputs, outputs, initializers }
       ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ GRAPH PASSES   src/import/ (passes.h, one .cpp per pass)                │
 │   runStandardPasses(g, PassOptions)                                     │
 │   dequantizeGraph    (QDQ/QLinear → float + saturation Clip; default on) │
 │   inferShapes        (resolves dynamic dims from declared shapes;        │
 │                       batch defaults to 1)                               │
 │   foldBatchNorm      (BN → scale/bias folded into Conv)                 │
 │   lowerBatchNorm     (remaining BN → per-channel Mul+Add)               │
 │   constFold+inferShapes to convergence (shape-path subgraphs)           │
 │   subpixelConvTranspose, lowerConv (KxK Conv → ConvGemm)                │
 │   eliminateDeadNodes, fusePointwiseChains (the general region fusion:   │
 │   producer epilogues + standalone units), pruneDeadInitializers         │
 │                                                                         │
 │   MobileNetV2: 105 → 55 nodes at -O1                                    │
 └───────────────────────────────────────────────────────────────────────┘
       │   optimized Graph, then Graph::topoSort()
       ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ SESSION::ensureBackends() + buildBucket()   src/core/session.cpp        │
 │   1. instantiate backends in priority order:                           │
 │        cfg.backend, cfg.fallback..., (CPU if allowCpuFallback)         │
 │      (skip unregistered / !available())                                 │
 │   2. allocate RtTensor pool, load initializers (host-resident)         │
 │   3. per-node backend assignment: first backend that supports(op,dt)   │
 │   4. partition topo-ordered nodes into maximal same-backend SEGMENTS   │
 │   5. backend->compileSegment(nodeIdx, g, cfg) for each segment         │
 │   6. compute per-segment boundaryInputs / boundaryOutputs              │
 │   (cache flush deferred to teardown:                                   │
 │     ~Session → updateCache → backend->finalize())                      │
 └───────────────────────────────────────────────────────────────────────┘
       │   std::vector<std::unique_ptr<Segment>> segments_
       ▼
 ┌──────────────────────────┐      ┌──────────────────────────┐
 │ VulkanBackend (primary)  │      │ CpuBackend (ref+fallback) │
 │  NC4HW4 packed layout    │      │  scalar + NEON kernels     │
 │  1 pre-recorded VkCmdBuf  │ ...  │  (Add, Gemm)               │
 │  per static segment       │      │                            │
 │  push descriptors,        │      │                            │
 │  timestamp queries        │      │                            │
 └──────────────────────────┘      └──────────────────────────┘
       │
       ▼
 Session::run(inputs, outputs):
   bind inputs → run segments in order → residency reconciled at boundaries
   → collect graph outputs (host, NCHW, declared dtype)
```

The canonical IR layout is **NCHW** throughout (`graph.h` header comment, and
`TensorFormat::NCHW` is the default in `TensorDesc`). Only the Vulkan backend
re-packs into `NC4HW4` internally; the rest of the engine never sees that layout.

---

## 2. Core abstractions

### 2.1 Tensors: `TensorDesc`, `HostBuffer`, `RtTensor` (`include/vknn/tensor.h`)

There are two tensor representations, split by lifetime:

- **`TensorDesc`** — compile-time description living in `Graph::tensors`. Holds the
  logical NCHW `shape` (dynamic dims as `-1`), `dtype`, `format`
  (`TensorFormat::NCHW` by default), and the `isInput` / `isOutput` /
  `isInitializer` role flags.

- **`RtTensor`** — runtime storage in the `Session` pool, indexed by `TensorId`.
  It models **dual residency**:

  ```cpp
  struct RtTensor {
    TensorId id = kNoTensor;
    Shape shape;
    DType dtype = DType::Float32;

    // host residency (canonical NCHW, fp32 for compute/IO)
    HostBuffer host;
    bool hostValid = false;

    // device residency (managed by a backend)
    std::shared_ptr<DeviceStorage> device;     // null until a backend allocates
    TensorFormat deviceFormat = TensorFormat::Unknown;
    DType        deviceDtype  = DType::Float32;
    bool         deviceValid  = false;
  };
  ```

  The `host` side is always **canonical NCHW, fp32** (`HostBuffer` is a raw byte
  vector with `f32()` / `i64()` views). The `device` side is an opaque
  `DeviceStorage` (forward-declared in the core; the Vulkan backend defines it as a
  `std::shared_ptr<vk::Buffer>` in `vk_backend.h`), tagged with its own
  `deviceFormat` (e.g. `NC4HW4`) and `deviceDtype` (e.g. fp16). The two `*Valid`
  flags drive the residency-reconciliation logic (§4.3): a tensor is packed,
  unpacked, or copied only when needed.

### 2.2 Graph IR: `Graph`, `Node`, `Attributes` (`include/vknn/graph.h`, `include/vknn/op.h`)

`Graph` is the whole model:

```cpp
class Graph {
  std::vector<TensorDesc>          tensors;       // indexed by TensorId
  std::vector<Node>                nodes;          // topologically ordered after import
  std::map<std::string, TensorId>  tensorByName;
  std::vector<TensorId>            inputs, outputs;
  std::map<TensorId, HostBuffer>   initializers;   // weight host data, keyed by id
  ...
  void topoSort();   // stable, throws on cycle
};
```

A `Node` (`op.h`) references tensors by id and carries operator metadata plus the
fusion results from the passes:

```cpp
struct Node {
  OpType type = OpType::Unknown;
  std::string name;
  std::vector<TensorId> inputs, outputs;
  Attributes attr;
  ActType fusedAct = ActType::None;        // set by the pointwise fusion's inline-act path
  float   actLo = 0, actHi = 0;             // Clip bounds when fusedAct == Clip
  int32_t subOp = 0;                        // Unary/Binary/Reduce sub-code
  TensorId fusedResidual = kNoTensor;       // residual epilogue (runtime keeps support for older .vxm)
  TensorId fusedBias = kNoTensor;           // MatMul bias epilogue (runtime keeps support for older .vxm)
};
```

`OpType` (`include/vknn/op_type.h`, append-only — `.vxm` files store the raw
integer) enumerates the full supported set (~80 ops, per-op coverage in
[op-coverage.md](op-coverage.md)): the conv family (`Conv`, `ConvTranspose`),
`Gemm`/`MatMul`/`Einsum`, pooling, normalization (`BatchNorm`, `LayerNorm`,
`Softmax`), the elementwise `Unary`/`Binary` families, data movement
(`Reshape`, `Transpose`, `Slice`, `Concat`, `Gather`, `ScatterND`, ...), the
generator ops `ConstantOfShape` / `Range` (which run on the GPU once their output
size resolves), the quantized family the dequantize pass lowers (`QuantizeLinear`,
`DequantizeLinear`, `QLinearConv`, `MatMulInteger`, ...), and the fused ops the
passes synthesize (`FusedSE`, `FusedDwPw`, `FusedPointwise`), the ORT contrib family lowered at
import (`SkipLayerNorm`, `RotaryEmbedding`, `GroupQueryAttention`, `MatMulNBits`, ...), and the
load-time-only fused ops (`Rope`, `FusedAttention`) created at session build and never
serialized. The pure shape ops
`Shape` / `Constant` const-fold away before planning. `ActType`
(`None`/`Relu`/`Relu6`/`Clip`/`HardSwish`/`SiLU`) is kept in sync with
`shaders/common.glsl`; the last two are set by the swish self-gating fusion. `Attributes` is a typed `name → Attr` map with `geti` /
`getf` / `getints` / `gets` accessors used by kernels to read pads, strides, etc.

### 2.3 Backend + Segment execution model (`include/vknn/backend.h`)

A `Backend` is the per-device abstraction:

```cpp
class Backend {
  virtual BackendKind kind() const = 0;
  virtual const char* name() const = 0;
  virtual bool available() const = 0;                 // usable on this device?
  virtual bool supports(OpType t, DType dt) const = 0; // per-op capability query

  // residency reconciliation hooks (defaults are no-ops):
  virtual void toHost  (RtTensor& rt, ExecContext& ctx) {}   // ensure valid host NCHW
  virtual void toDevice(RtTensor& rt, ExecContext& ctx) {}   // ensure resident on device

  virtual std::unique_ptr<Segment> compileSegment(
      const std::vector<int>& nodeIdx, Graph& g, const Config& cfg) = 0;

  virtual void finalize() {}   // flush caches after all segments compiled
};
```

A `Segment` is an executable run of nodes that all belong to one backend:

```cpp
class Segment {
  virtual void run(ExecContext& ctx) = 0;
  Backend* backend = nullptr;
  bool isFallback = false;            // true when a CPU segment exists only because the
                                      // primary backend could not run these ops
  std::vector<int> nodeIdx;
  std::vector<TensorId> boundaryInputs;   // consumed from outside this segment
  std::vector<TensorId> boundaryOutputs;  // produced for outside this segment
};
```

`ExecContext` is the per-run bundle handed to every op: a pointer to the `RtTensor`
pool (with `RtTensor& t(TensorId)`), the `Graph`, the `Config`, and the `Profiler`.

Backends self-register via `VKNN_REGISTER_BACKEND(KIND, TYPE)` into the
`BackendRegistry` singleton; `Session::ensureBackends()` calls `BackendRegistry::create(kind, cfg)`.
This requires the static lib to be linked whole-archive
(`$<LINK_LIBRARY:WHOLE_ARCHIVE,vknn>`), which keeps the registrar globals from being stripped.

### 2.4 Config (`include/vknn/config.h`)

`vknn::Config` is an MNN-inspired struct, loadable from JSON
(`Config::fromJsonFile` / `fromJsonString` / `toJson`). Key fields:

- `backend` (`Vulkan`/`Cpu`), ordered `fallback` list, `allowCpuFallback`
  (CPU is the implicit final fallback).
- `precision` (`Low`/`Normal`/`High` quality tiers; default `Low` = fp16), `maxSubmitNodes`,
  `freeWeightsAfterUpload`.
- `cpuThreads` (CPU-backend worker threads, default 4), `maxSubmitBindings` (second
  command-buffer split trigger by recorded push-descriptor writes, default 1024),
  `decodeChainSteps` (decode iterations recorded as one command-buffer chain with on-device
  token feedback, default 1).
- Cache controls: `cacheFile` (the per-model cache, §7), `cacheDir` (the graph-only fallback location),
  `noCache` (skip caching), and `tuning` (`None`/`Fast`/`Heavy` autotune effort).
- Caller-owned dma-buf I/O via `Tensor::fromDmaBuf` / `Tensor::toDmaBuf` (§6).
- Diagnostics: `profile`, `verbosity`, `layerDump` / `layerDumpDir`.
- Conv kernel + GPU-pass knobs via `setHint(Hint, Mode)`: `Hint::Winograd` (`Auto`/`On`/`Off`),
  `Hint::FlatLayout` / `Hint::GpuIslandFold` / `Hint::MatMulViewFold` / `Hint::RopeFusion` /
  `Hint::FusedAttention` / `Hint::KvConcatFold` (`On`/`Off`, all default On), and the
  Winograd/direct-conv variant hints (`WinogradVariant`, `WinogradUnit`, `DirectConv3x3`).

### 2.5 Session / Runtime (`include/vknn/session.h`, `src/core/session.cpp`)

`Session` owns the active backends, the caches (via the backends), and one `PlanBucket` per
compiled input-shape set; each bucket holds its planned graph, segments, `RtTensor` pool, and
fallback records. `run()` dispatches to the bucket whose bound input shapes match, and
`prepareShapes()` compiles an additional bucket (ONNX-built sessions only — `.vxm` buckets are
fixed at compile time). `Runtime` is a thin façade:
`Runtime::load(path, cfg, cacheFile)` dispatches on extension — a `.vxm`
(pre-optimized offline by `vknn_compile`) loads via `Session::createFromVxm`
and skips the parser and passes; an `.onnx` goes through
`Session::createFromOnnx`. `cacheFile` defaults to `<model>.cache`, see §7.

The build flow is `createFromOnnx` → `importOnnx` → `create(Graph&&, cfg)` →
`ensureBackends()` + `buildBucket()`. `ensureBackends()` instantiates the backends once per
session; `buildBucket()` runs the passes (ONNX path — a `.vxm` arrives pre-optimized), applies
the load-time-only graph rewrites (`fuseRope` → `foldMatMulViews` → `fuseDecodeAttention` →
`foldFusedAttentionKvConcat`; hint-gated, never serialized) and the Vulkan flat-layout pass,
builds the pool, assigns per-node backends, folds tiny GPU islands onto the CPU
(`Hint::GpuIslandFold`), partitions into segments, and compiles them (all in `session.cpp`).
The unified cache is flushed at teardown via `Session::updateCache()` (called from
`~Session()`), not at `plan()` time, and only when it actually changed during the
session. Member **declaration order is load-bearing for teardown**:
`backends_` is declared first so it is destroyed *last*, after `segments_` and
`pool_` have released their device buffers — the `VulkanContext` lives inside
`VulkanBackend`.

`IOTensor` is the host-side I/O struct (name, shape, dtype, raw bytes) handed to
`Session::run(inputs, outputs)`, which binds inputs into the pool, runs segments in
order, optionally dumps layers, and copies graph outputs back out (host, NCHW,
each output's declared dtype — an fp16-declared output returns raw fp16 bytes). `Session::tensor(name)`, `nodeBackends()`, `fallbackOps()`, and `fallbackReasons()` exist for
debugging and fallback reporting (`vknn_compile --support-report` writes the same per-node
assignment offline).

Engine-resident state and decode acceleration also live on `Session`: `linkOutputToInput()`
keeps an output→input pair device-resident across runs (the KV-cache path — ranged copies of the
previous run's output apply at the start of each run, no host round-trip), `setOutputArgMax()`
replaces a full output download with an on-device argmax (8-byte readback), `setOutputRow()`
downloads a single row of a flat output (the prefill-logits case), and `configureDecodeChain()`
records `Config::decodeChainSteps` decode iterations into one command-buffer sequence with on-GPU
token feedback — each bit-identical to its unlinked / single-step equivalent.

### 2.6 Profiler (`include/vknn/profiler.h`)

`Profiler` collects one `OpRecord` per executed op when `Config::profile` is set:
op `name`/`type`, `backend` string, `cpuMs` (CPU wall) and `gpuMs` (GPU timestamp;
`< 0` means not measured), `dispatch` dims, `bytesIO`, and `fellBack`. It can emit a
sorted table (`printTable`), JSON (`toJson`), or a Chrome trace
(`writeChromeTrace`). The Vulkan backend fills `gpuMs` from per-node timestamp
queries scaled by `timestampPeriod` (39.0625 ns on the target device); the CPU backend
fills `cpuMs` and sets `fellBack = isFallback`.

---

## 3. The `NC4HW4` layout and why

`TensorFormat::NC4HW4` (`include/vknn/tensor_format.h`) is the Vulkan backend's
internal layout: channels are packed into `vec4` blocks. A logical NCHW tensor with
`C` channels becomes `ceil(C/4)` channel-blocks of 4, i.e. its packed element count is

```cpp
// vk_backend.h
inline int64_t packedElems(const Shape& shape) {
  NCHW x = NCHW::from(shape);
  return x.n * cBlocks(x.c) * 4 * x.h * x.w;   // cBlocks(c) = (c + 3) / 4
}
```

Why pack this way:

- **vec4 = the GPU's natural width.** Mobile-GPU compute lanes load and ALU
  `vec4`s efficiently; packing 4 channels per element keeps memory accesses
  coalesced and lets every kernel work in `vec4` granularity. The backend never uses
  `VK_KHR_cooperative_matrix` (the capability is queried but no kernel consumes it), so GEMM and
  conv run on subgroup + `vec4` math rather than tensor-core-style matrix ops.
- **Channel-major-in-blocks suits CNN access patterns.** Conv accumulates over
  input channels; grouping channels into blocks of 4 makes the inner loop a tidy
  `vec4` reduction.
- **fp16 storage, fp32 accumulate.** With `precision = low` (fp16) the packed buffers are
  16-bit (`shaderFloat16` and 16-bit storage are both supported), but kernels
  accumulate in fp32 to preserve accuracy. On MobileNetV2 this yields cosine
  `0.999965` vs the fp32 path.
- **Selective fp32 (`precision = normal`).** Tensors whose names match the built-in
  geometry-tail set (or `Config::fp32Tensors`) keep fp32 *storage* under fp16 compute:
  `markFp32` (load-time, after the layout pass) sets `TensorDesc::storeFp32`, the producing
  node selects its fp32 kernel variant via a per-node `env.useFp16`, and `ConvertDtype` nodes
  bridge the fp16/fp32 frontier. Empty matches are a no-op, so `normal` == `low` for models
  without the named tensors. `high` is full fp32.

Host data is always plain NCHW fp32; the conversion to/from `NC4HW4` happens only at
segment boundaries via the `pack` / `unpack` compute shaders (§4.3, §5).

---

## 4. Segments: pre-recorded Vulkan command buffers *and* CPU fallback

The segment model lets the engine both pre-record a single static GPU command
buffer *and* drop transparently to the CPU for any op the GPU cannot run, with no
special-casing in the core dispatch loop.

### 4.1 Backend assignment and partitioning (`session.cpp`)

Each node is assigned to the **highest-priority backend whose
`supportsNode(graph, node, dt)` returns true** (shape-aware; defaults to the
type-only `supports(op, dt)`). If the primary backend declines an op (e.g. the GPU
lacks a kernel, or the op is listed in `Config::disableVkOps`), a throttled
fallback warning is logged and the node falls through to the next backend
(ultimately CPU). If *no* backend supports an op, planning throws
`Status::Unsupported`.

The topo-ordered node list is then sliced into **maximal contiguous runs of the same
backend index** — the segments:

```cpp
for (size_t n = 0; n < graph_.nodes.size(); ++n) {
  if (parts.empty() ||
      nodeBackendIdx_[n] != nodeBackendIdx_[parts.back().front()])
    parts.push_back({});
  parts.back().push_back((int)n);
}
```

An all-Vulkan model is one segment. Forcing two ops to CPU
(`Config::disableVkOps = "Add,GlobalAveragePool"`) fragments MobileNetV2 into 23
Vulkan/CPU segments, and the output remains cosine `1.000000` because boundaries
reconcile residency.

### 4.2 Boundary sets

For each segment, `buildBucket()` computes `boundaryInputs` (tensors consumed by the
segment but produced *outside* it — graph inputs, initializers, or another
segment's output) and `boundaryOutputs` (tensors produced here and consumed by
another segment or that are graph outputs). These two sets are the *only* places
host↔device traffic can happen; everything internal to a segment stays in
device-native layout.

### 4.3 Residency reconciliation at boundaries

Because the `host` side of every `RtTensor` is canonical NCHW fp32, it is the
universal handoff format. The CPU backend operates directly on `host`
(`cpu::allocOut` sets `hostValid = true`, `deviceValid = false`), so its
`toHost`/`toDevice` are the default no-ops. The Vulkan `VulkanSegment::run`
(`src/backend/vulkan/vk_backend.cpp`) does the reconciliation explicitly:

```cpp
void run(ExecContext& ctx) override {
  // boundary INPUTS: attach device buffer; pack host→device if host is the
  // only valid copy (i.e. produced by a CPU segment or freshly bound input)
  for (TensorId tid : boundaryInputs) {
    RtTensor& rt = ctx.t(tid);
    ... rt.device->buffer = buffers_[tid];
    bool alreadyHere = rt.deviceValid && rt.device->buffer == buffers_[tid];
    if (rt.hostValid && !alreadyHere) {   // repack unless THIS segment's buffer holds it
      VulkanBackend::packToBuffer(buffers_[tid].get(), rt, useFp16_);  // NCHW→NC4HW4
      rt.deviceValid  = true;
      rt.deviceFormat = TensorFormat::NC4HW4;
    }
  }

  for (VkCommandBuffer cmd : cmds_)                  // one PRE-RECORDED cmd buf per CHUNK
    wall += be_->runner().submitAndWait(cmd);        // one vkQueueSubmit + full fence wait per chunk

  // boundary OUTPUTS: unpack device→host so the next (possibly CPU) segment can read
  for (TensorId tid : boundaryOutputs) {
    ... VulkanBackend::unpackFromBuffer(buffers_[tid].get(), rt, useFp16_);  // NC4HW4→NCHW
    rt.deviceValid  = true;
    rt.deviceFormat = TensorFormat::NC4HW4;
  }
  ...
}
```

The `hostValid && !alreadyHere` guard is the residency check: a tensor is packed unless this
segment's own buffer already holds the current values (a tensor produced by an earlier GPU
segment is `deviceValid` but points at that segment's buffer, so it is repacked here). Internal activations never touch the
host. A CPU segment in the middle of a Vulkan model therefore requires no special
handling: the Vulkan segment before it unpacks its outputs to host, the CPU segment
reads and writes host, and the next Vulkan segment packs them back.

### 4.4 Pre-recorded command buffers

Because each plan is a single fully-static shape (batch 1 by default, or a declared
shape / bucket — see [limitations.md §1](limitations.md)) and the graph is fully
planned, a `VulkanSegment`
allocates device buffers for all its activation tensors, calls `op->prepare()` on
each op (which builds pipelines and prepacks+uploads weights), and then
**records its command buffers once, at load time**, in `VulkanSegment::record()` — the recording
splits into a new command buffer (chunk) every `maxSubmitNodes` (default 500) nodes or
`maxSubmitBindings` (default 1024) recorded push-descriptor writes, so no single submission runs
long enough to trip the GPU watchdog:

```cpp
void record() {
  cmd_ = be_->runner().allocate();
  be_->runner().begin(cmd_);
  vkCmdResetQueryPool(cmd_, queryPool_, ...);
  for (size_t k = 0; k < nodeIdx.size(); ++k) {
    if (needBarrier)                          // buffer-level hazard tracking (RAW/WAW/WAR):
      vk::computeBarrier(cmd_);               // a barrier only when required, so independent
                                              // branches overlap on the GPU
    if (queryPool_)                           // per-node timestamps only while profiling
      vkCmdWriteTimestamp(cmd_, ..._TOP_OF_PIPE_..., queryPool_, k*2);
    ops_[k]->record(cmd_, node, env_);        // bind pipeline + push descriptors + dispatch
    if (queryPool_)
      vkCmdWriteTimestamp(cmd_, ..._BOTTOM_OF_PIPE_..., queryPool_, k*2 + 1);
  }
  be_->runner().end(cmd_);
}
```

At run time `run()` re-submits the pre-recorded chunks — one `vkQueueSubmit` + full fence wait
per chunk (batching every chunk into one submit lets the driver watchdog reset a long submission
and zero its unexecuted tail) — with no per-op host round-trips; re-recording happens only when
the recorded stream must change: a boundary buffer rebind (dma-buf imports, GPU-convert
bindings) or a change to the resident links, registered argmax reductions, or the decode-chain
configuration. Ops bind their data with **push descriptors** (no
descriptor-set allocation churn), and each segment owns a timestamp `VkQueryPool`
(2 queries per node) for the profiler. The `VulkanOp` interface
(`prepare()` / `record()`) and the `VkOpEnv` (context, pipeline cache, weight cache,
`devBuf` activation-buffer lookup, `useFp16`, tuning level, command runner for
autotune benchmarks) keep each op self-contained.

On MobileNetV2 fp32, this gives 24.35 ms / 41 fps (GPU compute alone 12.1 ms by
timestamp; the rest is pack/unpack and host↔device transfer); fp16 is 22.0 ms /
45.4 fps. The CPU reference path is 672 ms / 1.5 fps with cosine `1.000000`.

The CPU counterpart, `CpuSegment::run`, follows the same model: it iterates its ops
calling `op->run(node, ctx)` against the host pool, timing each into the profiler and
propagating `isFallback`.

---

## 5. Shaders

Kernels are GLSL compute shaders in `shaders/` (~150 files): the layout kernels
(`pack`/`unpack`/`convert_layout`/`boundary_convert`), the conv family (below),
`matmul`/`matmul_tiled`, the `flat_*` row-major generic ops, softmax/layernorm
reductions, and the raster kernels — most precision-templated via
`precision.glsl` so the build emits both the fp32 and `_fp16` variants from one
source, plus shared `common.glsl`. The vendored glslang compiles them at build
time, and `tools/embed_spirv.py` embeds them into the static lib as SPIR-V,
reachable through `vknn::embeddedShaders()` (`src/backend/vulkan/vk_pipeline.h`),
so the runtime ships no loose shader files.

Conv selects among several strategies per shape: the general kernel (with a
register-blocked variant), the 1×1 pointwise family (`conv1x1`, strided
`conv1x1_s2`, split-K deep 1×1 — WTILE autotuned per shape), a 3×3 LDS kernel,
the tiled-GEMM Winograd (`tuneWino` decides Winograd-vs-direct and the F(2,3)/F(4,3) tile by
deterministic shape rules — numerics-affecting choices are never timing-raced — and times only
the bit-neutral tiles-per-thread parameter), and a specialized
depthwise kernel (`dwconv`). The general
conv exposes its `local_size_x` as a spec constant so the autotuner can pick a
workgroup size per conv signature.

---

## 6. Caller-owned DMA-BUF I/O (`include/vknn/ion.h`, `src/core/ion.cpp`)

I/O can be bound directly to caller-provided dma-buf fds, with no vknn-side host I/O
buffer or copy. vknn never allocates these buffers — the fd comes from the caller's
camera / gralloc / ION stack (on this device, an allocation from the `/dev/dma_heap/system`
heap, since `/dev/ion` is absent).

`vknn::IonBuffer` is a thin wrapper over a caller-provided fd:
`IonBuffer::wrapFd(int fd, size_t bytes, bool takeOwnership = false)` mmaps it for CPU
access. The high-level API binds model I/O to fds: `Tensor::fromDmaBuf(int fd, shape, name)`
binds a model **input** to a caller fd, and `Tensor::toDmaBuf(int fd, shape, name)` binds a
model **output** to a caller fd. `Model::run(const std::vector<Tensor>& inputs, const
std::vector<Tensor>& outputs = {})` reads each fd-bound input straight from the fd and writes
each fd-bound output straight into the fd. A bound output's returned `Tensor` carries no host
copy (its data is empty); unbound outputs come back as host tensors.

The Vulkan import primitive `vk::Buffer::importDmaBufFd`
(`VkImportMemoryFdInfoKHR`, handle type `DMA_BUF_BIT_EXT`, allowed types queried with
`vkGetMemoryFdPropertiesKHR`) is the mechanism behind `fromDmaBuf`/`toDmaBuf`: the GPU reads and
writes the dma-buf directly. A `layout`/`dtype` declared on `fromDmaBuf`/`toDmaBuf` that matches
the device-native boundary (or `TensorFormat::Auto`) binds the fd directly as the boundary
buffer; any other declaration keeps the pooled boundary buffer and a recorded `boundary_convert`
dispatch converts on the GPU. `Session::inputInfo()`/`outputInfo()` report the exact
device-native block to allocate (`deviceBytes`/`deviceFormat`/`deviceDtype`). A failed import
warns loudly (zero-copy unavailable) rather than reading undefined memory. Because the platform is UMA (all memory types are
`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`) there are no staging copies, and the path is
bit-exact against the staged path (`maxAbsErr 0`).

---

## 7. Caches (`config.cacheFile`)

One **self-validating, multi-variant MessagePack cache file** accelerates warm session creation
(ADR-0009; codec in `src/core/cache_codec.{h,cpp}`). `Runtime::load(path, cfg, cacheFile)` takes the
cache path as its third argument; empty (the default) resolves to `<model>.cache` next to the model
(e.g. `enc.vxm` → `enc.cache`), exposed as `Config::cacheFile`. For a session built from an in-memory
graph (no model path), the cache lives under `cacheDir` instead. Loading the file on a warm start skips
shader compilation, conv autotuning, and the Winograd weight transform; caching is always on
(`Config::noCache` disables it for cold-compile measurement).

The document is guarded as a whole by a format version, a **kernel hash** (md5 of all embedded SPIR-V,
`embeddedShadersHash()`), the device (vendor/device/driver + pipeline-cache UUID), and a model hash — any
mismatch discards and recomputes it. It holds one **variant** per cache-affecting configuration
(precision, `flatLayout`, `gpuIslandFold`, `matmulViewFold`, `ropeFusion`, `fusedAttention`,
`kvConcatFold`, `fp32Tensors`, and the Winograd/direct-conv hints); a matching variant is
reused and a new configuration appends one. Each variant bundles three blobs:

- **Vulkan pipeline cache** — the `VkPipelineCache` blob (`vk::PipelineCache`, created in
  `VulkanBackend`). Skips driver shader recompilation.
- **Prepacked-weights cache** — `WeightCache` (`vk_backend.h`): the weights already repacked into
  `NC4HW4`, keyed by op + role + shape (106 entries on MobileNetV2); warm runs skip the host repacking.
- **Autotune cache** — in the same `WeightCache`, an op-signature → chosen kernel-parameter
  table (`local_size_x`, WTILE, register blocking, Winograd tiles-per-thread) plus the `Tuning`
  level each entry was measured at (`tuned()` / `setTuned()`); a cached entry is reused when its
  recorded level ≥ the requested level and re-swept otherwise. Only bit-neutral parameters are
  ever timed — numerics-affecting choices are deterministic shape rules.

`Session::updateCache()` writes the file, but only when the cache changed during the session —
an unchanged warm run leaves the file untouched. It is called automatically from `~Session()`
(the cache is flushed at teardown), and is also public for manual checkpointing.

---

## 8. Backends

| Backend | Role | Notes |
| --- | --- | --- |
| **VulkanBackend** | primary | `NC4HW4` layout, one pre-recorded command buffer per static segment, push descriptors, timestamp queries, fp16 storage / fp32 accumulate, on-device autotuning. |
| **CpuBackend** | reference + fallback | scalar reference kernels for all ops, NEON kernels for `Add` and `Gemm`; operates directly on host NCHW. Cosine `1.000000` vs onnxruntime. |

---

## 9. Extensibility

All three plugin points use self-registration (no edits to core dispatch), and rely
on the whole-archive link of the static lib:

- **Add a CPU op:** subclass `vknn::CpuOp` (implement `run`) + `VKNN_REGISTER_CPU_OP(OpType, Class)`.
- **Add a Vulkan op:** subclass `vknn::VulkanOp` (implement `prepare()` / `record()`)
  + `VKNN_REGISTER_VK_OP(OpType, Class)` + a `.comp` shader in `shaders/`.
- **Add a backend:** subclass `vknn::Backend` + `VKNN_REGISTER_BACKEND(BackendKind, Class)`.

---

## 10. Source map

| Area | Path |
| --- | --- |
| Public headers | `include/vknn/` (`backend.h`, `session.h`, `tensor.h`, `graph.h`, `op.h`, `config.h`, `profiler.h`, `tensor_format.h`, `ion.h`, …) |
| ONNX import | `src/import/onnx/onnx_parser.cpp` |
| Graph passes | `src/import/` (`passes.h`; one `.cpp` per pass: `run_standard_passes.cpp`, `infer_shapes.cpp`, `const_fold.cpp`, `fuse_pointwise_chains.cpp`, `prune_dead_initializers.cpp`, …) |
| Session / planning | `src/core/session.cpp` |
| Core support | `src/core/` (`graph.cpp`, `op.cpp`, `config.cpp`, `profiler.cpp`, `backend_registry.cpp`, `ion.cpp`, `json.h`, `logging.cpp`) |
| Vulkan backend | `src/backend/vulkan/` (`vk_backend.cpp`, `vk_context`, `vk_buffer`, `vk_command`, `vk_pipeline`, `ops/` — one `.cpp` per operator (Transpose and Slice share one)) |
| CPU backend | `src/backend/cpu/` (`cpu_backend.cpp`, `ops/` — one file per operator) |
| Shaders | `shaders/` (compiled by the vendored glslang — `third_party/glslang` — or a system `glslc` fallback, embedded via `tools/embed_spirv.py`) |
| Examples | `examples/` (`basics/` — `probe`, `backend_switch`, `op_check`, `readme_quickstart`; `vision/` — `classify`, `predict`, `predict_cache`, `image_bench`; `bench/` — `profile`, `microbench`; `io/` — `run_io`, `dmabuf_fd_io`, `zerocopy_simple`, `zerocopy_cache`; `llm/` — `chat`, `vlm`; `splatting/` — `yonosplat`) |
