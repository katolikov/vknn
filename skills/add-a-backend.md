# How to add a backend

Goal: make a new piece of hardware or runtime executable as a backend, selectable from `Config`, with
no edits to core dispatch. Deep dive (with worked snippets and the offline-compiled-accelerator
pattern): [../docs/ADDING_A_BACKEND.md](../docs/ADDING_A_BACKEND.md).

## The contract

Subclass `vknn::Backend` (see `include/vknn/backend.h`) and implement:

- `kind()` / `name()` — your `BackendKind` tag (add the enumerator + JSON spelling to
  `include/vknn/config.h`) and a short label.
- `available()` — return `false` to be skipped entirely (missing driver/extension, host build).
- `supports(OpType, DType)` — the coarse per-op capability check. Override `supportsNode(graph, node,
  dt)` when support depends on attributes/shapes (e.g. a Concat axis, a broadcast layout).
- `compileSegment(nodeIdx, graph, cfg)` — do all expensive one-time work here (build pipelines, pack
  weights, plan buffers, pre-record a command buffer) and return a `Segment`.
- `toHost` / `toDevice` — move a tensor to/from your device layout at segment boundaries. Leave them
  as no-ops if your native layout is host NCHW.
- `finalize()` — flush any caches to `cfg.cacheDir` (optional).

A `Segment` subclass holds the compiled work; its `run(ExecContext&)` is the hot path and should do as
little as possible. Fill `backend`, `nodeIdx`, `boundaryInputs`, `boundaryOutputs`.

## Register it

One line at file scope; whole-archive linking retains the static initializer:

```cpp
VKNN_REGISTER_BACKEND(BackendKind::kYours, YourBackend);
```

Add the `.cpp` to the `vknn` CMake target (the source globs pick it up on reconfigure).

## Select it

Purely `Config` — no code changes elsewhere:

```jsonc
{ "backend": "YOURS", "fallback": ["VULKAN", "CPU"], "allowCpuFallback": true }
```

The `Session` instantiates the primary backend, checks `available()`, asks `supportsNode` per node,
partitions consecutive same-backend nodes into segments, and falls back through the list for declined
ops. Forcing a few ops to CPU is a quick way to exercise the boundary path and confirm cosine 1.0.

## Two shapes of backend

- **JIT-from-IR** (like the in-tree Vulkan + CPU backends): compile on device, from the IR, at session
  creation.
- **Offline-compiled** (a vendor NPU/DSP SDK): consume a pre-compiled artifact built by a host-side
  toolchain; `compileSegment` loads/binds it instead of JITing. The plug-in path supports both — see
  the pattern section in the deep-dive doc.
