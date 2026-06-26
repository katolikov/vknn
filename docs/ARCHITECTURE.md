# VKNN Architecture

`VKNN` (Vulkan Neural Network, namespace `vknn`) is a small C++17 inference runtime
that loads an ONNX CNN, lowers it to a backend-agnostic NCHW IR, optimizes it with
graph passes, partitions it into backend-specific *segments*, and executes those
segments — primarily on a Vulkan compute backend tuned for an AMD RDNA-class mobile
GPU, with a CPU backend as a seamless reference and
fallback.

This document describes the end-to-end pipeline, the core abstractions, the internal
`NC4HW4` tensor layout, the segment execution model (which gives both pre-recorded
Vulkan command buffers *and* transparent CPU fallback), and the cache subsystem. It
references real source under `include/vknn/` and `src/`.

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
 │ GRAPH PASSES   src/import/passes.{h,cpp}  — runStandardPasses(g, 1)     │
 │   inferShapes        (static batch = 1, fills dynamic dims)             │
 │   foldBatchNorm      (BN → scale/bias folded into Conv)                 │
 │   fuseActivations    (Clip/Relu → Node.fusedAct on Conv/Gemm)          │
 │   constFold          (evaluate shape-path subgraphs)                    │
 │   eliminateDeadNodes                                                    │
 │                                                                         │
 │   MobileNetV2: 105 → 65 nodes; 35 Clip/Relu fused; 5 shape nodes folded │
 └───────────────────────────────────────────────────────────────────────┘
       │   optimized Graph, then Graph::topoSort()
       ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ SESSION::plan()   src/core/session.cpp                                  │
 │   1. instantiate backends in priority order:                           │
 │        cfg.backend, cfg.fallback..., (CPU if allowCpuFallback)         │
 │      (skip unregistered / !available())                                 │
 │   2. allocate RtTensor pool, load initializers (host-resident)         │
 │   3. per-node backend assignment: first backend that supports(op,dt)   │
 │   4. partition topo-ordered nodes into maximal same-backend SEGMENTS   │
 │   5. backend->compileSegment(nodeIdx, g, cfg) for each segment         │
 │   6. compute per-segment boundaryInputs / boundaryOutputs              │
 │   7. backend->finalize()  (flush caches to disk)                       │
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
   → collect graph outputs (host, NCHW, fp32)
```

The canonical IR layout is **NCHW** throughout (`graph.h` header comment, and
`TensorFormat::kNCHW` is the default in `TensorDesc`). Only the Vulkan backend
re-packs into `NC4HW4` internally; the rest of the engine never sees it.

---

## 2. Core abstractions

### 2.1 Tensors: `TensorDesc`, `HostBuffer`, `RtTensor` (`include/vknn/tensor.h`)

There are two tensor representations, split by lifetime:

- **`TensorDesc`** — compile-time description living in `Graph::tensors`. Holds the
  logical NCHW `shape` (dynamic dims as `-1`), `dtype`, `format`
  (`TensorFormat::kNCHW` by default), and the `isInput` / `isOutput` /
  `isInitializer` role flags.

- **`RtTensor`** — runtime storage in the `Session` pool, indexed by `TensorId`.
  It models **dual residency**:

  ```cpp
  struct RtTensor {
    TensorId id = kNoTensor;
    Shape shape;
    DType dtype = DType::kFloat32;

    // host residency (canonical NCHW, fp32 for compute/IO)
    HostBuffer host;
    bool hostValid = false;

    // device residency (managed by a backend)
    std::shared_ptr<DeviceStorage> device;     // null until a backend allocates
    TensorFormat deviceFormat = TensorFormat::kUnknown;
    DType        deviceDtype  = DType::kFloat32;
    bool         deviceValid  = false;
  };
  ```

  The `host` side is always **canonical NCHW, fp32** (`HostBuffer` is a raw byte
  vector with `f32()` / `i64()` views). The `device` side is an opaque
  `DeviceStorage` (forward-declared in the core; the Vulkan backend defines it as a
  `std::shared_ptr<vk::Buffer>` in `vk_backend.h`), tagged with its own
  `deviceFormat` (e.g. `kNC4HW4`) and `deviceDtype` (e.g. fp16). The two `*Valid`
  flags are what the residency-reconciliation logic (§4.3) reads and writes so a
  tensor is only ever packed/unpacked/copied when actually needed.

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
  OpType type = OpType::kUnknown;
  std::string name;
  std::vector<TensorId> inputs, outputs;
  Attributes attr;
  ActType fusedAct = ActType::kNone;   // set by fuseActivations
  float   actLo = 0, actHi = 0;        // Clip bounds when fusedAct == kClip
};
```

`OpType` enumerates the supported ops: `kConv`, `kClip`, `kRelu`, `kAdd`,
`kGlobalAvgPool`, `kGemm`, `kReshape`, `kFlatten`, `kSoftmax`, `kBatchNorm`,
`kIdentity`, plus shape ops `kShape` / `kConstant` / `kGather` / `kUnsqueeze` /
`kConcat` (the shape ops are CPU-only and are const-folded away on the Vulkan
path). `ActType` (`kNone`/`kRelu`/`kRelu6`/`kClip`) is kept in sync with
`shaders/common.glsl`. `Attributes` is a typed `name → Attr` map with `geti` /
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
`BackendRegistry` singleton; `Session::plan()` calls `BackendRegistry::create(kind)`.
This works only because the static lib is linked whole-archive
(`$<LINK_LIBRARY:WHOLE_ARCHIVE,vknn>`), so the registrar globals are not stripped.

### 2.4 Config (`include/vknn/config.h`)

`vknn::Config` is an MNN-inspired struct, loadable from JSON
(`Config::fromJsonFile` / `fromJsonString` / `toJson`). Key fields:

- `backend` (`kVulkan`/`kCpu`), ordered `fallback` list, `allowCpuFallback`
  (CPU is the implicit final fallback).
- `precision` (`kFp32`/`kFp16`/`kAuto`; default `kFp16`), `power`, `cpuThreads`.
- `inputLayout` / `outputLayout` (`kNCHW`/`kNHWC`) — what the *user* supplies and
  wants back; the engine converts internally.
- `enableZeroCopy` (ION / dma-buf path, §6).
- Cache controls: `cacheDir`, `cachePipeline`, `cacheWeights`, `cacheTuning`.
- Diagnostics: `profile`, `verbosity`, `layerDump` / `layerDumpDir`.
- `tuning` (`kOff`/`kFast`/`kThorough`) — autotuning level.

### 2.5 Session / Runtime (`include/vknn/session.h`, `src/core/session.cpp`)

`Session` owns the planned graph, the active backends, the segments, the caches
(via the backends), and the `RtTensor` pool. `Runtime` is a thin façade
(`Runtime::load(onnxPath, cfg)` → `Session::createFromOnnx`).

The build flow is `createFromOnnx` → `importOnnx` → `create(Graph&&, cfg)` →
`plan()`. `plan()` runs the passes, instantiates backends, builds the pool, assigns
per-node backends, partitions into segments, compiles them, and finalizes caches
(all in `session.cpp`). Member **declaration order is load-bearing for teardown**:
`backends_` is declared first so it is destroyed *last*, after `segments_` and
`pool_` have released their device buffers (the `VulkanContext` lives inside
`VulkanBackend`).

`IOTensor` is the host-side I/O struct (name, shape, dtype, raw bytes) handed to
`Session::run(inputs, outputs)`, which binds inputs into the pool, runs segments in
order, optionally dumps layers, and copies graph outputs back out (host, NCHW,
fp32). `Session::tensor(name)` and `nodeBackends()` exist for debugging and
fallback reporting.

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

`TensorFormat::kNC4HW4` (`include/vknn/tensor_format.h`) is the Vulkan backend's
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

- **vec4 = the GPU's natural width.** RDNA-class compute lanes load and ALU
  `vec4`s efficiently; packing 4 channels per element keeps memory accesses
  coalesced and lets every kernel work in `vec4` granularity. This matters
  especially because the device exposes **no `VK_KHR_cooperative_matrix`**, so GEMM
  and conv are done with subgroup + `vec4` math rather than tensor-core-style
  matrix ops.
- **Channel-major-in-blocks suits CNN access patterns.** Conv accumulates over
  input channels; grouping channels into blocks of 4 makes the inner loop a tidy
  `vec4` reduction.
- **fp16 storage, fp32 accumulate.** With `precision = fp16` the packed buffers are
  16-bit (`shaderFloat16`, 16-bit storage are both supported), but kernels still
  accumulate in fp32 to preserve accuracy. On MobileNetV2 this yields cosine
  `0.999965` vs the fp32 path.

Host data is always plain NCHW fp32; the conversion to/from `NC4HW4` happens only at
segment boundaries via the `pack` / `unpack` compute shaders (§4.3, §5).

---

## 4. Segments: pre-recorded Vulkan command buffers *and* CPU fallback

The segment model is the heart of the design. It is what lets the engine both
pre-record a single static GPU command buffer *and* drop transparently to the CPU
for any op the GPU can't run — with no special-casing in the core dispatch loop.

### 4.1 Backend assignment and partitioning (`session.cpp`)

Each node is assigned to the **highest-priority backend whose `supports(op, dt)`
returns true**. If the primary backend declines an op (e.g. the GPU lacks a kernel,
or `VKNN_DISABLE_VK_OPS` is set), a throttled fallback warning is logged and the
node falls through to the next backend (ultimately CPU). If *no* backend supports an
op, planning throws `Status::kUnsupported`.

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

So an all-Vulkan model is one big segment; forcing two ops to CPU
(`VKNN_DISABLE_VK_OPS="Add,GlobalAveragePool"`) fragments MobileNetV2 into 23
Vulkan/CPU segments — yet the output is still cosine `1.000000`, because boundaries
reconcile residency.

### 4.2 Boundary sets

For each segment, `plan()` computes `boundaryInputs` (tensors consumed by the
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
(`src/backends/vulkan/vk_backend.cpp`) does the reconciliation explicitly:

```cpp
void run(ExecContext& ctx) override {
  // boundary INPUTS: attach device buffer; pack host→device if host is the
  // only valid copy (i.e. produced by a CPU segment or freshly bound input)
  for (TensorId tid : boundaryInputs) {
    RtTensor& rt = ctx.t(tid);
    ... rt.device->buffer = buffers_[tid];
    if (rt.hostValid && !rt.deviceValid) {
      VulkanBackend::packToBuffer(buffers_[tid].get(), rt, useFp16_);  // NCHW→NC4HW4
      rt.deviceValid  = true;
      rt.deviceFormat = TensorFormat::kNC4HW4;
    }
  }

  double wall = be_->runner().submitAndWait(cmd_);   // submit the PRE-RECORDED cmd buf

  // boundary OUTPUTS: unpack device→host so the next (possibly CPU) segment can read
  for (TensorId tid : boundaryOutputs) {
    ... VulkanBackend::unpackFromBuffer(buffers_[tid].get(), rt, useFp16_);  // NC4HW4→NCHW
    rt.deviceValid  = true;
    rt.deviceFormat = TensorFormat::kNC4HW4;
  }
  ...
}
```

The `hostValid && !deviceValid` guard is exactly the residency check: a tensor is
packed only if the host is its sole valid copy. Internal activations never touch the
host. This is why a CPU segment in the middle of a Vulkan model "just works" — the
Vulkan segment before it unpacks its outputs to host, the CPU segment reads/writes
host, and the next Vulkan segment packs them back.

### 4.4 Pre-recorded command buffers

Because batch is static (1) and the graph is fully planned, a `VulkanSegment`
allocates device buffers for all its activation tensors, calls `op->prepare()` on
each op (which builds pipelines and prepacks+uploads weights), and then
**records the command buffer once, at compile time**, in `VulkanSegment::record()`:

```cpp
void record() {
  cmd_ = be_->runner().allocate();
  be_->runner().begin(cmd_);
  vkCmdResetQueryPool(cmd_, queryPool_, ...);
  for (size_t k = 0; k < nodeIdx.size(); ++k) {
    vkCmdWriteTimestamp(cmd_, ..._TOP_OF_PIPE_..., queryPool_, k*2);
    ops_[k]->record(cmd_, node, env_);          // bind pipeline + push descriptors + dispatch
    vkCmdWriteTimestamp(cmd_, ..._BOTTOM_OF_PIPE_..., queryPool_, k*2 + 1);
    vk::computeBarrier(cmd_);
  }
  be_->runner().end(cmd_);
}
```

At run time `run()` only re-submits `cmd_` (`submitAndWait`) — no re-recording, no
per-op host round-trips. Ops bind their data with **push descriptors** (no
descriptor-set allocation churn) and each segment owns a timestamp `VkQueryPool`
(2 queries per node) for the profiler. The `VulkanOp` interface
(`prepare()` / `record()`) and the `VkOpEnv` (context, pipeline cache, weight cache,
`devBuf` activation-buffer lookup, `useFp16`, tuning level, command runner for
autotune benchmarks) make each op self-contained.

On MobileNetV2 fp32 this gives 24.35 ms / 41 fps (GPU compute alone 12.1 ms by
timestamp; the rest is pack/unpack and host↔device transfer); fp16 is 22.0 ms /
45.4 fps. The CPU reference path is 672 ms / 1.5 fps with cosine `1.000000`.

The CPU counterpart, `CpuSegment::run`, is the trivial version of the same model: it
just iterates its ops calling `op->run(node, ctx)` against the host pool, timing each
into the profiler and propagating `isFallback`.

---

## 5. Shaders

Kernels are GLSL compute shaders in `shaders/`: `pack`, `unpack`, `conv`, `dwconv`,
`avgpool`, `fc`, `add`, each with an `_fp16` variant, plus shared `common.glsl`.
They are compiled by `glslc` at build time and embedded into the static lib as
SPIR-V by `tools/embed_spirv.py`, reachable through
`vknn::embeddedShaders()` (`src/backends/vulkan/vk_pipeline.h`), so the runtime ships
no loose shader files.

Conv uses two strategies: a general `group == 1` kernel (covering both 1×1 pointwise
and 3×3 strided convs) and a specialized depthwise kernel (`dwconv`). The general
conv exposes its `local_size_x` as a spec constant so the autotuner can pick a
workgroup size per conv signature.

---

## 6. ION / DMA-BUF zero-copy (`include/vknn/ion.h`, `src/core/ion.cpp`)

On the target device `/dev/ion` no longer exists, so zero-copy uses **DMA-BUF heaps**
(`/dev/dma_heap/system`) via `DMA_HEAP_IOCTL_ALLOC`, importing the resulting dma-buf
fd into Vulkan with `VkImportMemoryFdInfoKHR` (handle type `DMA_BUF_BIT_EXT`). Two
modes are exposed by `vknn::IonBuffer`: **Mode A** allocates a heap buffer, **Mode B**
(`wrapFd`) wraps an externally provided fd; the import itself is
`vk::Buffer::importDmaBufFd`. Because the platform is UMA (all memory types are
`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`) there are no staging copies. Verified
on-device in both modes, bit-exact against the staged path (`maxAbsErr 0`). Enabled
via `Config::enableZeroCopy`.

---

## 7. Caches (`config.cacheDir`)

Three content/configuration-keyed caches make warm session creation dramatically
faster (cold first run + autotune 445 ms; cold no-tune ~152 ms; warm all-caches
68 ms — up to 6.5× faster warm):

- **Vulkan pipeline cache** — `VkPipelineCache` persisted to
  `cacheDir/pipeline.bin` (`vk::PipelineCache`, created in `VulkanBackend` when
  `cachePipeline`). Skips driver shader recompilation.
- **Prepacked-weights cache** — `WeightCache` (`vk_backend.h`): a content-keyed,
  length-prefixed blob of weights already repacked into `NC4HW4`, keyed by
  op + role + shape. On MobileNetV2 this is 106 entries; warm runs skip the host
  repacking. Controlled by `cacheWeights`.
- **Autotune cache** — stored in the same `WeightCache` as an
  op-signature → chosen `local_size_x` table (`tuned()` / `setTuned()`); 20 conv
  workgroup-size entries for MobileNetV2. Controlled by `cacheTuning` and the
  `tuning` level.

All three are flushed in `Backend::finalize()` (`VulkanBackend::saveCaches`) at the
end of `Session::plan()`.

---

## 8. Backends

| Backend | Role | Notes |
| --- | --- | --- |
| **VulkanBackend** | primary | `NC4HW4` layout, one pre-recorded command buffer per static segment, push descriptors, timestamp queries, fp16 storage / fp32 accumulate, on-device autotuning. |
| **CpuBackend** | reference + fallback | scalar reference kernels for all ops, NEON kernels for `Add` and `Gemm`; operates directly on host NCHW. Cosine `1.000000` vs onnxruntime. |

---

## 9. Extensibility

All three plugin points use self-registration (no edits to core dispatch), relying
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
| Graph passes | `src/import/passes.{h,cpp}` |
| Session / planning | `src/core/session.cpp` |
| Core support | `src/core/` (`graph.cpp`, `op.cpp`, `config.cpp`, `profiler.cpp`, `backend_registry.cpp`, `ion.cpp`, `json.h`, `logging.cpp`) |
| Vulkan backend | `src/backends/vulkan/` (`vk_backend.cpp`, `vk_context`, `vk_buffer`, `vk_command`, `vk_pipeline`, `vk_ops.cpp`) |
| CPU backend | `src/backends/cpu/` (`cpu_backend.cpp`, `ops_basic.cpp`, `ops_conv.cpp`, `ops_shape.cpp`) |
| Shaders | `shaders/` (compiled by `glslc`, embedded via `tools/embed_spirv.py`) |
| Examples | `examples/` (`probe`, `classify`, `profile`, `ion_zerocopy`, `backend_switch`, `op_check`) |
