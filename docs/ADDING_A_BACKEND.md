# Adding a Backend to vxrt

This guide explains how to add a new execution backend (a new piece of hardware
or runtime that can run inference) to **vxrt**. The contract is small: subclass
`vx::Backend`, implement a handful of virtual methods, register the class with
`VX_REGISTER_BACKEND`, and make it selectable from `vx::Config`. No edits to the
core dispatch, the importer, or the graph passes are needed — self-registration
plus whole-archive linking wires everything together at startup.

The worked reference throughout is the ENN backend stub,
`src/backends/enn/enn_backend.cpp`, which proves the entire plug-in path in ~65
lines while declining all ops. At the end we describe the real Samsung
ENN/NNC offline flow a full implementation would have to follow.

---

## 1. Where a backend sits in the pipeline

The runtime pipeline is:

```
ONNX file
  -> dependency-free protobuf parser
  -> backend-agnostic NCHW IR (vx::Graph)
  -> graph passes (inferShapes, foldBatchNorm, fuseActivations, constFold, eliminateDeadNodes)
  -> Session partitions the topo-ordered nodes into maximal same-backend "segments"
  -> backends compile + run each segment
```

A backend never sees ONNX, protobuf, or the raw graph passes. It is handed:

1. **Capability questions** via `supports(OpType, DType)` so the `Session` can
   decide which nodes to assign to it.
2. **A contiguous run of node indices** via `compileSegment(...)`, which it turns
   into an executable `Segment`.
3. **Tensor residency hooks** (`toHost` / `toDevice`) that the runtime calls at
   segment boundaries to move data on/off the device.

Everything the backend needs at run time arrives through `ExecContext`.

---

## 2. The `Backend` interface

From `include/vx/backend.h`:

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
`include/vx/config.h` (current values include `kVulkan`, `kCpu`, `kEnn`). This
tag is the key under which the backend is registered and the value
`config.backend` selects. Adding a brand-new backend means adding a new
`BackendKind` enumerator there.

The ENN stub:

```cpp
BackendKind kind() const override { return BackendKind::kEnn; }
```

### `name()` — human-readable label

Used in logs and profiler tags. Keep it short. The stub deliberately advertises
that it is not a real implementation:

```cpp
const char* name() const override { return "ENN(stub)"; }
```

### `available()` — usable on this device?

Return `false` to remove the backend from selection entirely (e.g. the required
driver/extension is missing, or you are on the host build). When `available()`
is `false`, the `Session` skips the backend without ever calling `supports()` or
`compileSegment()`.

Note the ENN stub returns `true` even though it is a stub. That is intentional:
it makes the backend *selectable* so the plug-in path is exercised end to end,
and then it declines every op so the configured fallback actually runs the
graph:

```cpp
// Selectable + instantiable (so the plug-in path is exercised). It declines all ops, so
// the configured fallback (Vulkan/CPU) executes the graph.
bool available() const override { return true; }
```

A real backend typically probes its dependencies here (or in the constructor)
and returns `true` only if it can actually run something.

### `supports(OpType, DType)` — per-op capability

The `Session` calls this for every node when assigning backends. Return `true`
only for the `(OpType, DType)` pairs you can execute. The partitioner groups
runs of consecutive nodes that the same backend supports into one segment; nodes
you decline are handed to the next backend in the fallback list (typically
`VulkanBackend`, then `CpuBackend`).

The stub declines everything, which is why it never executes a single op:

```cpp
bool supports(OpType, DType) const override { return false; }
```

`OpType` and `DType` are defined in `include/vx/graph.h` and
`include/vx/tensor.h` respectively. A real backend's `supports()` is where you
encode "I can do Conv and Gemm in fp16 but not Softmax", which directly shapes
how the graph is partitioned.

### `compileSegment(...)` — turn nodes into a `Segment`

Called once per assigned segment, at session-creation time. You receive the
node indices (into `g.nodes`), the mutable `Graph`, and the `Config`. You return
a `std::unique_ptr<Segment>` whose `run()` will be invoked at inference time.

This is where backends do their expensive, one-time work:

- **VulkanBackend** builds pipelines, pre-packs weights to NC4HW4, and
  pre-records *one* command buffer for the whole (static) segment.
- **CpuBackend** simply instantiates the per-node ops.

The stub cannot build a segment, so it fails loudly with a precise diagnostic
rather than returning something half-working:

```cpp
std::unique_ptr<Segment> compileSegment(const std::vector<int>&, Graph&,
                                        const Config&) override {
  throw Error(Status::kUnsupported,
              "ENN backend requires an NNC-format model; no on-device NNC compiler / public "
              "headers available on this device (see LIMITATIONS.md). Use VULKAN or CPU.");
}
```

In practice `compileSegment` is never reached for the stub, because
`supports()` returns `false` for every op, so the partitioner never assigns it a
segment. The throw is the safety net for the "selected but nothing to run" case.

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
(or imports a dma-buf for zero-copy), and `toHost` unpacks NC4HW4 -> NCHW back to
host memory. A backend whose native layout *is* NCHW and that runs on the host
can leave both at their defaults.

### `finalize()` — flush caches

Called once after all segments are compiled. Use it to persist anything you want
to reuse on the next cold start. The Vulkan backend flushes its `VkPipelineCache`,
the prepacked-weights cache, and the autotune cache to `config.cacheDir` here —
this is what turns a ~152 ms cold (no-tune) start into a ~68 ms warm start. Most
new backends will leave it empty until they have something worth caching.

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

Your `compileSegment` returns a subclass of `Segment` and is expected to fill in:

- `backend` — pointer back to the owning backend (used for boundary `toHost`/`toDevice`).
- `nodeIdx` — the nodes this segment covers (usually just the `nodeIdx` you were handed).
- `boundaryInputs` / `boundaryOutputs` — the tensor ids consumed from / produced
  for the outside world. The runtime uses these to drive residency reconciliation
  at segment edges, so only the tensors that actually cross a backend boundary
  get moved.
- `isFallback` — set by the CPU backend when a segment only exists because the
  primary backend declined those ops; it drives the fallback warning and the
  profiler tag. A normal backend leaves it `false`.

`run(ExecContext&)` is the hot path. It should do as little per-call work as
possible — all setup belongs in `compileSegment`. The Vulkan backend's `run()`
essentially submits one pre-recorded command buffer.

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

## 4. Registration: `VX_REGISTER_BACKEND`

A backend self-registers via a static initializer. No central list, no factory
table to edit.

```cpp
// --------------------------- Backend registry ---------------------------
class BackendRegistry {
 public:
  using Factory = std::function<std::unique_ptr<Backend>()>;
  static BackendRegistry& instance();
  void registerBackend(BackendKind k, Factory f);
  bool has(BackendKind k) const;
  std::unique_ptr<Backend> create(BackendKind k) const;
 private:
  std::map<BackendKind, Factory> factories_;
};

struct BackendRegistrar {
  BackendRegistrar(BackendKind k, BackendRegistry::Factory f) {
    BackendRegistry::instance().registerBackend(k, std::move(f));
  }
};
#define VX_REGISTER_BACKEND(KIND, TYPE)                                      \
  static ::vx::BackendRegistrar _vx_backend_reg_##TYPE(                      \
      KIND, []() -> std::unique_ptr<::vx::Backend> {                         \
        return std::unique_ptr<::vx::Backend>(new TYPE());                   \
      })
```

The stub registers itself with a single line at file scope:

```cpp
VX_REGISTER_BACKEND(BackendKind::kEnn, EnnBackend);
```

`VX_REGISTER_BACKEND(KIND, TYPE)` defines a file-static `BackendRegistrar` whose
constructor runs before `main()` and inserts a factory lambda under `KIND`. When
the `Session` needs a backend, it calls `BackendRegistry::instance().create(k)`.

> **Why the static initializer actually runs.** vxrt is a static library, so the
> linker would normally drop object files containing only unreferenced static
> initializers. The build links the library whole-archive
> (`$<LINK_LIBRARY:WHOLE_ARCHIVE,vxrt>` in CMake), which forces every object —
> and therefore every `VX_REGISTER_BACKEND` initializer — to be retained. This
> is the same mechanism that makes `VX_REGISTER_CPU_OP` and `VX_REGISTER_VK_OP`
> work. If you add a backend in a new `.cpp` and it never seems to register,
> confirm the file is part of the `vxrt` target and that whole-archive linking
> is in effect.

---

## 5. Config selection

A backend is chosen through `vx::Config` (JSON-backed; see `include/vx/config.h`):

- `backend` — the primary `BackendKind` (`VULKAN` | `CPU` | `ENN`).
- `fallback` — an ordered list of `BackendKind`s tried when the primary declines
  an op.
- `allowCpuFallback` — whether the CPU backend is allowed as the final fallback.

At session creation the `Session`:

1. instantiates the primary backend via the registry and checks `available()`;
2. for each node, asks `supports(OpType, DType)` of the primary, then walks the
   `fallback` list;
3. partitions the topo-ordered nodes into maximal same-backend segments;
4. calls `compileSegment` per segment and `finalize` once at the end.

So selecting ENN with a Vulkan fallback is purely config:

```jsonc
{
  "backend": "ENN",
  "fallback": ["VULKAN", "CPU"],
  "allowCpuFallback": true
}
```

With the stub, `backend: "ENN"` is accepted, the backend is instantiated and
probed, `supports()` declines every op, and the entire graph runs on `VULKAN`
(then `CPU` for anything Vulkan declines). This is exactly the fallback path the
core already exercises — e.g. forcing `VXRT_DISABLE_VK_OPS="Add,GlobalAveragePool"`
produces a mix of Vulkan and CPU segments and still yields cosine 1.000000
output. The ENN stub is just another producer of "decline" answers in that same
machinery.

---

## 6. Minimal skeleton for a new backend

Putting the pieces together, a new backend file looks like this:

```cpp
#include "vx/backend.h"
#include "vx/logging.h"

namespace vx {
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

  BackendKind kind() const override { return BackendKind::kMyBackend; } // add to config.h
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

  void finalize() override { /* flush caches to cfg.cacheDir */ }
};

VX_REGISTER_BACKEND(BackendKind::kMyBackend, MyBackend);

}  // namespace
}  // namespace vx
```

Add the source to the `vxrt` CMake target, add the `BackendKind` enumerator and
its JSON spelling in `include/vx/config.h`, and the backend is selectable with no
other changes.

---

## 7. The real ENN / NNC offline flow

The stub exists because two pieces are missing on the device, not because the
hardware is incapable. On the probed Galaxy S26 (Exynos 2600, `s5e9965`) the
backend `dlopen`-probes the ENN runtime and finds 4 of 5 libraries present —
`libenn_public_api_cpp.so`, `libenn_model.so`, `libenn_user_driver_gpu.so`,
`libenn_user_driver_unified.so` — yet still declines all ops:

```cpp
const char* kEnnLibs[] = {
    "libenn_public_api_cpp.so", "libenn_engine.so", "libenn_model.so",
    "libenn_user_driver_gpu.so", "libenn_user_driver_unified.so",
};
// ...
void probe() {
  int found = 0;
  std::string present;
  for (const char* lib : kEnnLibs) {
    void* h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (h) { ++found; present += std::string(lib) + " "; dlclose(h); }
  }
  // logs "probed 4/5 runtime libs present ... but NNC model + public headers
  // are unavailable on-device, so ENN execution is stubbed (falls back)."
}
```

The two blockers (documented in `docs/adr/0007` and `LIMITATIONS.md`):

1. **No public ENN C++ headers.** The `libenn_public_api_cpp.so` symbols are
   present, but without the matching headers there is no supported way to declare
   the API surface and call it.
2. **No on-device NNC compiler.** ENN does not consume ONNX at run time. It
   consumes a pre-compiled **`.nnc`** model, and the tool that produces `.nnc` is
   an offline Samsung SDK component that is not on the device.

A full implementation would follow the offline ENN/NNC flow rather than the
JIT-from-ONNX model the Vulkan backend uses:

```
source ONNX
  -> Samsung NNC compiler (offline, host-side SDK tool)   <-- not on device
     [graph conversion, quantization, NPU/GPU op scheduling]
  -> .nnc model file (pre-compiled for the target s5e9965 NPU/GPU)
  -> ship .nnc alongside the app
  -> ENN runtime on device (libenn_public_api_cpp.so + user drivers)
     [load .nnc, allocate I/O, execute on NPU/GPU]
```

Concretely, a real `EnnBackend` would:

- **`available()`** — confirm the ENN runtime libs load *and* their headers/API
  version match what we built against; return `false` otherwise.
- **`supports()`** — return `true` for ops that the offline NNC compile actually
  covers. Because NNC compiles a whole (sub)graph ahead of time, the natural unit
  is a large segment, not per-op JIT.
- **`compileSegment()`** — rather than compiling on device, locate (or have been
  shipped) the `.nnc` artifact corresponding to this segment, load it through the
  ENN public API, and wrap the ENN execution handle in a `Segment` subclass whose
  `run()` binds I/O buffers and dispatches.
- **`toDevice()` / `toHost()`** — bridge `RtTensor` host NCHW data to ENN's
  expected I/O buffer layout. On this UMA device, memory is
  `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`, and the platform supports
  `VK_EXT_external_memory_dma_buf` + AHardwareBuffer, so a future zero-copy ENN
  path could share dma-buf-backed buffers the same way the Vulkan backend already
  imports dma-buf fds — avoiding host<->device copies at segment boundaries.
- **`finalize()`** — nothing to JIT-cache (the `.nnc` is already compiled), but
  this is where you would drop any ENN session/handle caches.

The key architectural point: **only this one file changes.** Because the backend
plug-in path, the segment execution model, residency reconciliation, config
selection, and fallback are all already in place and exercised by the stub,
replacing the body of `src/backends/enn/enn_backend.cpp` with a real ENN runtime
binding — and shipping the offline-compiled `.nnc` models — is sufficient to make
ENN a first-class backend.

---

## 8. Checklist

- [ ] Add a `BackendKind` enumerator (and its JSON spelling) in `include/vx/config.h`.
- [ ] Subclass `vx::Backend`; implement `kind`, `name`, `available`, `supports`, `compileSegment`.
- [ ] Override `toHost` / `toDevice` if your native layout is not host NCHW.
- [ ] Subclass `vx::Segment`; do all setup in `compileSegment`, keep `run()` lean; fill `backend`, `nodeIdx`, `boundaryInputs`, `boundaryOutputs`.
- [ ] Override `finalize()` if you have caches to flush to `config.cacheDir`.
- [ ] `VX_REGISTER_BACKEND(BackendKind::kYours, YourBackend);` at file scope.
- [ ] Add the `.cpp` to the `vxrt` CMake target (whole-archive linking does the rest).
- [ ] Select via `config.backend` / `config.fallback`; verify `available()` and `supports()` partition the graph as expected.

See also: `include/vx/backend.h` (the contract), `src/backends/enn/enn_backend.cpp`
(the worked stub), `src/backends/vulkan/` (a full backend), and `LIMITATIONS.md` +
`docs/adr/0007` for the ENN status rationale.
