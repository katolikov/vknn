# ADR-0008: Disk caches (pipeline / weights / autotune) for fast warm sessions

## Status
Accepted (2026-06-24)

## Context
Session creation cold-cost is dominated by (a) SPIR-V→ISA pipeline compilation, (b) weight
prepacking to NC4HW4, and (c) workgroup-size autotuning. All are deterministic for a fixed
model+device and should be paid once.

## Decision
Three disk caches under `config.cacheDir` (keyed implicitly by device+driver via the pipeline
cache blob, and by op-signature for the others):
1. **`VkPipelineCache`** serialized to `pipeline.bin` — the driver reuses compiled pipelines.
2. **Prepacked-weights cache** (`weights.bin`) — the exact NC4HW4-packed weight/bias float blobs,
   content-keyed by `nodeName#w` / `#b`; skips the host repack loops on warm starts.
3. **Autotune cache** (in `weights.bin`) — chosen `local_size_x` per conv op-signature. The
   general-conv shader uses a **specialization constant** for `local_size_x`; on a cache miss
   (when `tuning != off`) the op benchmarks candidates ({64,128,256}, or {32,64,128,256} thorough)
   on-device and stores the fastest. Warm starts load the table and skip benchmarking.

`Backend::finalize()` flushes all caches after planning.

## Consequences
- Verified on device (MobileNetV2 fp16): cold session **445 ms** (first run incl. full autotune),
  **warm 68 ms** → up to **6.5×** faster; autotuning lifted inference to **22.0 ms / 45.4 fps**.
- Caches are safe to delete (regenerated). The pipeline cache is the dominant warm-start win;
  weights/tuning caches add incremental savings and make tuning a one-time cost.
- Benchmark-based tuning of additional axes (tile sizes, vectorization) is a documented extension.
