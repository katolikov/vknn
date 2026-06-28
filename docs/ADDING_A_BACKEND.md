# Adding a Backend to VKNN

This guide describes how to add a new execution backend (a piece of hardware
or runtime that can run inference) to **VKNN**. The contract is small: subclass
`vknn::Backend`, implement a handful of virtual methods, register the class with
`VKNN_REGISTER_BACKEND`, and make it selectable from `vknn::Config`. The core
dispatch, the importer, and the graph passes require no edits — self-registration
plus whole-archive linking wires everything together at startup.

Two backends in the tree serve as references: `src/backend/cpu/`
(`CpuBackend` — the NCHW-native reference path and automatic fallback) and
`src/backend/vulkan/` (`VulkanBackend` — a full GPU backend with packing, pipelines,
and a pre-recorded command buffer). The CPU backend is the smaller of the two and the
best starting point to read.

---

## 1. Where a backend sits in the pipeline

The runtime pipeline is:

```
ONNX file
  -> dependency-free protobuf parser
  -> backend-agnostic NCHW IR (vknn::Graph)
  -> graph passes (inferShapes, foldBatchNorm, fuseActivations, constFold, eliminateDeadNodes)
  -> Session partitions the topo-ordered nodes into maximal same-backend "segments"
  -> backends compile + run each segment
```

A backend never sees ONNX, protobuf, or the raw graph passes. It gets handed:

1. **Capability questions** via `supports(OpType, DType)` (and the shape-aware
   `supportsNode(...)`) so the `Session` can decide which nodes to assign to it.
2. **A contiguous run of node indices** via `compileSegment(...)`, which it turns
   into an executable `Segment`.
3. **Tensor residency hooks** (`toHost` / `toDevice`) that the runtime calls at
   segment boundaries to move data on/off the device.

Everything the backend needs at run time arrives through `ExecContext`.

---

## 2. The `Backend` interface

From `include/vknn/backend.h`:

```cpp
class Backend {
public:
  virtual ~Backend() = default;
  virtual BackendKind kind() const = 0;
  virtual const char* name() const = 0;
  /// Whether the backend is usable on this device (false => skip in selection).
  virtual bool available() const = 0;
  /// Capability query used for per-op backend assignment / fallback decisions.
  virtual bool supports(OpType t, DType dt) const = 0;
  /// Shape-aware capability query; defaults to the type-only check.
  virtual bool supportsNode(const Graph& g, const Node& nd, DType dt) const {
    return supports(nd.type, dt);
  }

  /// Ensure tensor `rt` has valid host data (NCHW canonical). Default: assume host already valid.
  virtual void toHost(RtTensor& rt, ExecContext& ctx) {}
  /// Ensure tensor `rt` is resident on this backend (e.g. uploaded+packed). Default: no-op.
  virtual void toDevice(RtTensor& rt, ExecContext& ctx) {}

  /// Compile a contiguous run of nodes (indices into graph.nodes) into a Segment.
  virtual std::unique_ptr<Segment> compileSegment(const std::vector<int>& nodeIdx,
                                                  Graph& g, const Config& cfg) = 0;

  /// Called once after all segments are compiled (flush pipeline/weight/tuning caches to disk).
  virtual void finalize() {}
};
```

### `kind()` — `BackendKind`

Returns the enum tag identifying this backend. `BackendKind` is declared in
`include/vknn/config.h` (the values are `kVulkan` and `kCpu`). This tag is the key
the backend registers under and the value `config.backend` selects. A new backend
means a new `BackendKind` enumerator there.

```cpp
BackendKind kind() const override { return BackendKind::kCpu; }
```

### `name()` — human-readable label

Used in logs and profiler tags. Keep it short.

```cpp
const char* name() const override { return "CPU"; }
```

### `available()` — usable on this device?

Return `false` to remove the backend from selection entirely (e.g. the required
driver/extension is missing, or on a host build). When `available()`
is `false`, the `Session` skips the backend without calling `supports()` or
`compileSegment()`. The CPU backend is always available; `VulkanBackend::available()`
returns `true` only after it creates a Vulkan instance/device and finds a
compute queue. A backend probes its dependencies here (or in the
constructor) and returns `true` only if it can run.

### `supports(OpType, DType)` / `supportsNode(...)` — per-op capability

The `Session` calls these for every node when assigning backends. Return `true`
only for the `(OpType, DType)` pairs you can execute. The partitioner groups
runs of consecutive nodes that the same backend supports into one segment; nodes
you decline are handed to the next backend in the fallback list (typically
`VulkanBackend`, then `CpuBackend`).

`supports()` is the coarse, type-only check. Override `supportsNode(graph, node, dt)`
when support depends on the node's attributes or shapes — `VulkanBackend` uses it to
accept a `Concat` only on a 4-aligned channel axis, or a `Binary` only for layouts its
kernels can broadcast. `OpType` lives in `include/vknn/op.h` and `DType` in
`include/vknn/tensor.h`. This is where a backend encodes "Conv and Gemm in fp16 but
not Softmax", which shapes how the graph is partitioned.

### `compileSegment(...)` — turn nodes into a `Segment`

Called once per assigned segment, at session-creation time. It receives the
node indices (into `g.nodes`), the mutable `Graph`, and the `Config`, and returns
a `std::unique_ptr<Segment>` whose `run()` is invoked at inference time.

This is where backends do their expensive, one-time work:

- **VulkanBackend** builds pipelines, pre-packs weights to NC4HW4, plans an
  activation-liveness buffer layout, and pre-records *one* command buffer for the whole
  (static) segment.
- **CpuBackend** instantiates the per-node ops.

### `toHost` / `toDevice` — tensor residency at boundaries

The execution model keeps each tensor resident on whichever backend last
touched it and reconciles residency only at segment boundaries. Before a segment
runs, the runtime calls `toDevice` on its boundary inputs and `toHost` on the
prior producer's outputs as needed, so a tensor produced on the GPU and consumed
on the CPU is moved exactly once.

- `toHost(rt, ctx)` must leave `rt` with valid host data in the **canonical
  NCHW** layout. Default is a no-op (assume host data already valid) — correct
  for a pure-CPU backend.
- `toDevice(rt, ctx)` must make `rt` resident on your backend (upload, re-pack to
  your internal layout, etc.). Default is a no-op.

The Vulkan backend overrides both: `toDevice` uploads and packs NCHW -> NC4HW4
(or imports a caller-provided dma-buf fd), and `toHost` unpacks NC4HW4 -> NCHW back to
host memory. A backend whose native layout *is* NCHW and that runs on the host
leaves both at their defaults, as `CpuBackend` does.

### `finalize()` — flush caches

Called once after all segments are compiled. Use it to persist anything to
reuse on the next cold start. The Vulkan backend bundles its `VkPipelineCache`,
the prepacked-weights cache, and the autotune cache into the unified `config.cacheFile`
here, which turns a cold start into a faster warm start. A backend leaves it
empty until it has something worth caching.

---

## 3. The `Segment` abstraction

A `Segment` is the unit of execution: a contiguous run of nodes that all belong
to one backend, compiled into a runnable object.

```cpp
class Segment {
public:
  virtual ~Segment() = default;
  virtual void run(ExecContext& ctx) = 0;
  Backend* backend = nullptr;
  bool isFallback = false;   // true if this CPU segment exists because the primary backend
                             // could not run these ops (drives the fallback warning + profiler tag)
  std::vector<int> nodeIdx;
  // tensor ids this segment consumes from outside / produces for outside (boundary set)
  std::vector<TensorId> boundaryInputs;
  std::vector<TensorId> boundaryOutputs;
};
```

`compileSegment` returns a subclass of `Segment` and fills in:

- `backend` — pointer back to the owning backend (used for boundary `toHost`/`toDevice`).
- `nodeIdx` — the nodes this segment covers (usually just the `nodeIdx` you were handed).
- `boundaryInputs` / `boundaryOutputs` — the tensor ids consumed from / produced
  for the outside world. The runtime uses these to drive residency reconciliation
  at segment edges, so only the tensors that actually cross a backend boundary
  get moved.
- `isFallback` — set by the CPU backend when a segment only exists because the
  primary backend declined those ops; it drives the fallback warning and the
  profiler tag. A normal backend leaves it `false`.

`run(ExecContext&)` is the hot path and does as little per-call work as
possible — all setup belongs in `compileSegment`. The Vulkan backend's `run()`
submits one pre-recorded command buffer and not much else.

### `ExecContext`

```cpp
struct ExecContext {
  std::vector<RtTensor>* pool = nullptr;   // indexed by TensorId
  const Graph* graph = nullptr;
  const Config* config = nullptr;
  Profiler* profiler = nullptr;
  RtTensor& t(TensorId id) { return (*pool)[id]; }
};
```

Everything `run()` needs is here: the tensor pool (`ctx.t(id)` returns the
`RtTensor` for a `TensorId`), the immutable `Graph`, the `Config`, and an
optional `Profiler`. Operators read their inputs and write their outputs through
`ctx.t(...)`.

---

## 4. Registration: `VKNN_REGISTER_BACKEND`

A backend self-registers via a static initializer. No central list, no factory
table to edit.

```cpp
struct BackendRegistrar {
  BackendRegistrar(BackendKind k, BackendRegistry::Factory f) {
    BackendRegistry::instance().registerBackend(k, std::move(f));
  }
};
#define VKNN_REGISTER_BACKEND(KIND, TYPE)                    \
  static ::vknn::BackendRegistrar _vx_backend_reg_##TYPE(    \
      KIND, []() -> std::unique_ptr<::vknn::Backend> {       \
        return std::unique_ptr<::vknn::Backend>(new TYPE()); \
      })
```

A backend registers itself with a single line at file scope:

```cpp
VKNN_REGISTER_BACKEND(BackendKind::kCpu, CpuBackend);
```

`VKNN_REGISTER_BACKEND(KIND, TYPE)` defines a file-static `BackendRegistrar` whose
constructor runs before `main()` and inserts a factory lambda under `KIND`. When
the `Session` needs a backend, it calls `BackendRegistry::instance().create(k)`.

> **Why the static initializer runs.** VKNN is a static library, so the
> linker would normally drop object files that hold only unreferenced static
> initializers. The build links the library whole-archive
> (`$<LINK_LIBRARY:WHOLE_ARCHIVE,vknn>` in CMake), which forces every object —
> and so every `VKNN_REGISTER_BACKEND` initializer — to be retained. The same
> mechanism makes `VKNN_REGISTER_CPU_OP` and `VKNN_REGISTER_VK_OP`
> work. If a backend in a new `.cpp` never registers,
> the file must be part of the `vknn` target and whole-archive linking
> must be in effect.

---

## 5. Config selection

A backend is chosen through `vknn::Config` (JSON-backed; see `include/vknn/config.h`):

- `backend` — the primary `BackendKind` (`VULKAN` | `CPU`).
- `fallback` — an ordered list of `BackendKind`s tried when the primary declines
  an op.
- `allowCpuFallback` — whether the CPU backend is allowed as the final fallback.

At session creation the `Session`:

1. instantiates the primary backend via the registry and checks `available()`;
2. for each node, asks `supportsNode(...)` of the primary, then walks the
   `fallback` list;
3. partitions the topo-ordered nodes into maximal same-backend segments;
4. calls `compileSegment` per segment and `finalize` once at the end.

Selecting Vulkan with a CPU fallback is purely config:

```jsonc
{
  "backend": "VULKAN",
  "fallback": ["CPU"],
  "allowCpuFallback": true
}
```

The same machinery handles partial fallback: when the primary backend declines a few
ops in the middle of a graph, those nodes become a small CPU segment and the rest stays
on the GPU. Forcing a few ops to CPU exercises the boundary path and
confirms it still yields cosine 1.000000 against the reference.

---

## 6. Minimal skeleton for a new backend

A backend file has this shape:

```cpp
#include "vknn/backend.h"
#include "vknn/logging.h"

namespace vknn {
namespace {

class MySegment : public Segment {
public:
  void run(ExecContext& ctx) override {
    // Hot path: execute the pre-compiled nodes for this segment.
    // Read inputs / write outputs via ctx.t(id).
  }
};

class MyBackend : public Backend {
public:
  MyBackend() { /* probe device / driver here */ }

  BackendKind kind() const override { return BackendKind::kMyBackend; }  // add to config.h
  const char* name() const override { return "MyBackend"; }
  bool available() const override { return /* deps present? */ true; }

  bool supports(OpType t, DType dt) const override {
    // Return true only for the (op, dtype) pairs you can run.
    return false;
  }

  void toDevice(RtTensor& rt, ExecContext& ctx) override {
    // Upload / re-pack rt to your device layout. Default no-op if NCHW host is fine.
  }
  void toHost(RtTensor& rt, ExecContext& ctx) override {
    // Bring rt back to canonical NCHW host data.
  }

  std::unique_ptr<Segment> compileSegment(const std::vector<int>& nodeIdx,
                                          Graph& g, const Config& cfg) override {
    auto seg = std::make_unique<MySegment>();
    seg->backend = this;
    seg->nodeIdx = nodeIdx;
    // Compute boundaryInputs / boundaryOutputs; build pipelines / pack weights here.
    return seg;
  }

  void finalize() override { /* flush caches into cfg.cacheFile */ }
};

VKNN_REGISTER_BACKEND(BackendKind::kMyBackend, MyBackend);

}  // namespace
}  // namespace vknn
```

Add the source to the `vknn` CMake target, add the `BackendKind` enumerator and
its JSON spelling in `include/vknn/config.h`, and the backend is selectable with no
other changes.

---

## 7. Pattern: an offline-compiled / whole-graph accelerator

Both in-tree backends compile *on device, from the IR, at session creation* —
the Vulkan backend JITs SPIR-V pipelines, the CPU backend instantiates per-node ops.
Some accelerators (vendor NPU SDKs, DSPs) work differently: they consume a
**pre-compiled model artifact** produced by an offline, host-side toolchain, and the
on-device runtime only loads and executes it.

The backend plug-in path handles that shape too, with no core changes — only the
mapping of the abstract methods differs:

- **`available()`** — confirm the runtime libraries load *and* their API version
  matches what you built against; return `false` otherwise.
- **`supports()` / `supportsNode()`** — return `true` for the ops the offline compile
  covers. Because such toolchains compile a whole (sub)graph ahead of time, the natural
  unit is a large segment, not per-op JIT.
- **`compileSegment()`** — rather than compiling on device, locate (or receive) the
  precompiled artifact for this segment, load it through the runtime API, and wrap the
  execution handle in a `Segment` whose `run()` binds I/O buffers and dispatches.
- **`toDevice()` / `toHost()`** — bridge `RtTensor` host NCHW data to the accelerator's
  expected I/O buffer layout. On a UMA device whose memory is
  `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` and that supports
  `VK_EXT_external_memory_dma_buf`, a zero-copy path shares dma-buf-backed buffers
  the same way the Vulkan backend imports dma-buf fds — avoiding host&harr;device
  copies at segment boundaries.
- **`finalize()`** — nothing to JIT-cache (the artifact is already compiled); this
  is where session/handle caches go.

**Only this one file changes.** The segment execution model,
residency reconciliation, config selection, and fallback are all in place, so a
backend — JIT or offline-compiled — drops in as a single `.cpp`.

---

## 8. Checklist

- [ ] Add a `BackendKind` enumerator (and its JSON spelling) in `include/vknn/config.h`.
- [ ] Subclass `vknn::Backend`; implement `kind`, `name`, `available`, `supports` (and `supportsNode` if shape-dependent), `compileSegment`.
- [ ] Override `toHost` / `toDevice` if your native layout is not host NCHW.
- [ ] Subclass `vknn::Segment`; do all setup in `compileSegment`, keep `run()` lean; fill `backend`, `nodeIdx`, `boundaryInputs`, `boundaryOutputs`.
- [ ] Override `finalize()` if you have caches to bundle into `config.cacheFile`.
- [ ] `VKNN_REGISTER_BACKEND(BackendKind::kYours, YourBackend);` at file scope.
- [ ] Add the `.cpp` to the `vknn` CMake target (whole-archive linking does the rest).
- [ ] Select via `config.backend` / `config.fallback`; verify `available()` and `supports()` partition the graph as expected.

See also: `include/vknn/backend.h` (the contract), `src/backend/cpu/` (a small NCHW
backend), and `src/backend/vulkan/` (a full GPU backend).
