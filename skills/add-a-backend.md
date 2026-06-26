# How to add a backend

A backend runs a piece of hardware or runtime, selectable from `Config`, with no edits to core
dispatch. For the deep dive — worked snippets and the offline-compiled-accelerator pattern — see
[../docs/ADDING_A_BACKEND.md](../docs/ADDING_A_BACKEND.md).

## The contract

Subclass `vknn::Backend` (see `include/vknn/backend.h`) and implement:

- `kind()` / `name()` — the `BackendKind` tag (add the enumerator + JSON spelling to
  `include/vknn/config.h`) and a short label.
- `available()` — returns `false` to be skipped entirely (missing driver/extension, host build).
- `supports(OpType, DType)` — the coarse per-op capability check. Override `supportsNode(graph, node,
  dt)` when support depends on attributes/shapes (e.g. a Concat axis, a broadcast layout).
- `compileSegment(nodeIdx, graph, cfg)` — performs the expensive one-time work (build pipelines, pack
  weights, plan buffers, pre-record a command buffer) and returns a `Segment`.
- `toHost` / `toDevice` — move a tensor to/from the device layout at segment boundaries. They are
  no-ops when the native layout is host NCHW.
- `finalize()` — flushes any caches to `cfg.cacheDir` (optional).

A `Segment` subclass holds the compiled work; its `run(ExecContext&)` is the hot path and stays
minimal. Fill `backend`, `nodeIdx`, `boundaryInputs`, `boundaryOutputs`.

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
ops. Forcing a few ops to CPU exercises the boundary path and confirms cosine 1.0.

## Two shapes of backend

- **JIT-from-IR** (like the in-tree Vulkan + CPU backends): compile on device, from the IR, at session
  creation.
- **Offline-compiled** (a vendor NPU/DSP SDK): consume a pre-compiled artifact built by a host-side
  toolchain; `compileSegment` loads and binds it instead of JITing. The plug-in path handles both — see
  the pattern section in the deep-dive doc.
