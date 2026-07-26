# ADR-0008: Disk caches (pipeline / weights / autotune) for fast warm sessions

## Status
Superseded by [ADR-0009](0009-cache-tuning-redesign.md) (2026-07-05). The three caches described here
still exist; ADR-0009 replaces the hand-rolled `VKNNCAC1` container with a self-validating, multi-variant
MessagePack file and collapses the `cache-mode` + `tuning` knobs into one `tuning` (none/fast/heavy).

Accepted (2026-06-24)

## Context
Session creation cold-cost is dominated by (a) SPIR-V→ISA pipeline compilation, (b) weight
prepacking to NC4HW4, and (c) workgroup-size autotuning. All are deterministic for a fixed
model+device and are paid once.

## Decision
Three caches are bundled into **one unified per-model cache file** (`<model>.cache`, container magic
`VKNNCAC1`), keyed implicitly by device+driver via the pipeline cache blob and by op-signature for the
others:
1. **`VkPipelineCache`** blob — the driver reuses compiled pipelines.
2. **Prepacked-weights cache** — the exact NC4HW4-packed weight/bias float blobs,
   content-keyed by `nodeName#w` / `#b`; skips the host repack loops on warm starts.
3. **Autotune cache** — chosen `local_size_x` per conv op-signature. The
   general-conv shader uses a **specialization constant** for `local_size_x`; on a cache miss
   (when `tuning != off`) the op benchmarks candidates ({64,128,256}, or {32,64,128,256} thorough)
   on-device and stores the fastest. Warm starts load the table and skip benchmarking.

`Runtime::load(path, cfg, cacheFile)` resolves the cache path — the argument wins, then
`Config::cacheFile`, then `<model>.cache` next to the model. A session built from an in-memory graph
has no model path to anchor a default, so it caches only when `Config::cacheFile` is set;
`Runtime::cacheFileIn(dir, model)` names one inside a shared directory, and missing parent
directories are created on the first write.

The file is written twice over a session's life, both times only when its bytes actually changed:
`Session::flushNewCacheWork()` at the end of every creation path, so a cold load's autotune sweep is
on disk before the first run, and `Session::updateCache()` from `~Session()` for whatever the run
phase added. Without the creation-time flush a process killed while a model is resident (an Android
app in the background) repeats the whole sweep on its next load. The creation-time flush costs one
size query when nothing new was produced: it runs only when the weight cache is dirty or the driver's
pipeline blob grew since the last save.

## Consequences
- On device (MobileNetV2 fp16): cold session **445 ms** (first run incl. full autotune),
  **warm 68 ms** → up to **6.5×** faster; autotuning lifts inference to **22.0 ms / 45.4 fps**.
- The cache file is safe to delete and regenerate. The pipeline cache is the dominant warm-start win;
  weights/tuning caches add incremental savings and make tuning a one-time cost.
- Benchmark-based tuning of additional axes (tile sizes, vectorization) is a documented extension.
