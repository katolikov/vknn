# ADR-0006: Segment-based execution model + automatic CPU fallback

## Status
Accepted (2026-06-24)

## Context
We need (a) pre-recorded Vulkan command buffers for the static graph (perf), and (b) per-op
fallback to CPU when the active backend can't run an op (correctness + the M5 requirement),
without the core engine special-casing any backend.

## Decision
The Session assigns each topologically-ordered node to the highest-priority backend whose
`supports(op, dtype)` is true (CPU is the universal final fallback), then partitions the node
list into **maximal runs of consecutive same-backend nodes** ("segments"). Each backend compiles
its segments into a `Segment` object:
- The **Vulkan** segment allocates one device buffer per activation tensor, prepares ops (compiles
  pipelines, prepacks/uploads weights), and **pre-records a single command buffer** for the whole
  segment with barriers + timestamp queries. `run()` packs boundary inputs, submits once, unpacks
  boundary outputs.
- The **CPU** segment just runs its ops immediately.

**Tensor residency** is reconciled at segment boundaries: a Vulkan segment uploads (packs
NCHW→NC4HW4) its boundary inputs that are only host-valid, and downloads (unpacks NC4HW4→NCHW)
its boundary outputs so the next CPU segment / final output sees them. That's the whole
cross-backend hand-off — no op needs to know about other backends.

A throttled WARNING is emitted per fallen-back op type, and the profiler tags fallback ops.

## Consequences
- A fully-Vulkan graph is one segment → one command buffer → minimal host↔device sync.
- Disabling any Vulkan op (`VKNN_DISABLE_VK_OPS`) transparently splits the graph and falls back to
  CPU with correct output (verified on device: 23 segments, cosine still 1.000000).
- Adding a backend never touches core dispatch (ADR-0002 registry + this model).
