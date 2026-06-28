# ADR-0008: Disk caches (pipeline / weights / autotune) for fast warm sessions

## Status
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

`Runtime::load(path, cfg, cacheFile)` resolves the cache path — an empty `cacheFile` defaults to
`<model>.cache` next to the model. `Session::updateCache()` writes the file from `~Session()`, and
only when the cache actually changed during the session (an unchanged warm run leaves the file
untouched). `config.cacheDir` is the fallback location for a session built from an in-memory graph,
which has no model path to anchor the cache file.

## Consequences
- On device (MobileNetV2 fp16): cold session **445 ms** (first run incl. full autotune),
  **warm 68 ms** → up to **6.5×** faster; autotuning lifts inference to **22.0 ms / 45.4 fps**.
- The cache file is safe to delete and regenerate. The pipeline cache is the dominant warm-start win;
  weights/tuning caches add incremental savings and make tuning a one-time cost.
- Benchmark-based tuning of additional axes (tile sizes, vectorization) is a documented extension.
