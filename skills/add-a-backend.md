# How to add a backend

A backend runs a piece of hardware or runtime, selectable from `Config`, with no edits to core
dispatch. For the deep dive — worked snippets and the offline-compiled-accelerator pattern — see
[../docs/adding-a-backend.md](../docs/adding-a-backend.md).

## The contract

Subclass `vknn::Backend` (see `include/vknn/backend.h`) and implement:

- `kind()` / `name()` — the `BackendKind` tag (add the enumerator to `include/vknn/backend_kind.h` and its string spelling to
  `backendName`/`backendFromStr` in `src/core/config.cpp`) and a short label.
- `available()` — returns `false` to be skipped entirely (missing driver/extension, host build).
- `configure(cfg)` — applies the session `Config` before planning (e.g. the debug op-disable list);
  called once per session create, before `supportsNode()` drives assignment. Default no-op.
- `supports(OpType, DType)` — the coarse per-op capability check. Override `supportsNode(graph, node,
  dt, whyNot)` when support depends on attributes/shapes (e.g. a Concat axis, a broadcast layout);
  on refusal, fill `*whyNot` (may be null) with a short stable reason for the fallback diagnostics.
- `compileSegment(nodeIdx, graph, cfg)` — performs the expensive one-time work (build pipelines, pack
  weights, plan buffers, pre-record a command buffer) and returns a `Segment`.
- `toHost` / `toDevice` — move a tensor to/from the device layout at segment boundaries. They are
  no-ops when the native layout is host NCHW.
- `finalize()` — called once after all segments are compiled; flushes any caches to `cfg.cacheFile`
  (optional).

A `Segment` subclass holds the compiled work; its `run(ExecContext&)` is the hot path and stays
minimal. Fill `backend`, `compiledGraph`, and `nodeIdx`; the `Session` computes `boundaryInputs`/`boundaryOutputs`
and sets `isFallback` after `compileSegment` returns. Optional `Segment` virtuals (resident
output-to-input links, on-device argmax/row readback, decode chains) default to `Unsupported`; the
`Session` then uses host-side paths.

## Register it

One line at file scope; whole-archive linking retains the static initializer. The generated factory
constructs `new YourBackend(cfg)`, so the class takes the session `Config` in its constructor:

```cpp
VKNN_REGISTER_BACKEND(BackendKind::kYours, YourBackend);
```

Add a `file(GLOB_RECURSE ...)` entry for the new `src/backend/<name>/*.cpp` directory to the `vknn`
sources in the root `CMakeLists.txt` — the existing globs cover only `src/core`, `src/import`,
`src/layout`, `src/backend/cpu`, and (Vulkan builds) `src/backend/vulkan`.

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
