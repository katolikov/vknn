# ADR-0009: Self-validating multi-variant MessagePack cache; one tuning knob

## Status
Accepted (2026-07-05). Supersedes [ADR-0008](0008-caches-and-autotuning.md).

## Context
ADR-0008 bundled the pipeline / prepacked-weight / autotune caches into a hand-rolled `VKNNCAC1`
container guarded only by an 8-byte magic. Two user knobs governed it: `cache-mode` (off/tune/full, what
to persist) and `tuning` (off/fast/thorough, autotune effort). Three gaps drove this redesign:
- **Weak validation.** Only the `VkPipelineCache` blob self-validates (driver UUID). The prepacked-weight
  and autotune blobs had no guard beyond the per-model filename, so a shader/code change, a driver
  update, or a wrong device could silently reuse stale artifacts.
- **Confusing surface.** `cache-mode` and `tuning` overlapped; users had to reason about both.
- **Single-config.** One file held one configuration; switching precision recompiled from scratch.

## Decision
1. **One knob.** `Config::tuning` = `none | fast | heavy` (autotune effort only; legacy `off`→none,
   `thorough`→heavy). `cache-mode`/`CacheMode` and `Hint::Tuning` are removed. Caching is **always on**;
   `Config::noCache` (CLI `--no-cache`) is a debug escape hatch for cold-compile measurement.
2. **MessagePack format.** The cache is a MessagePack document (official `msgpack-c`, vendored as a
   submodule) behind `src/core/cache_codec.{h,cpp}` — host-testable, inspectable with standard tooling.
3. **Self-validating whole-file guards.** format version + **kernel hash** (md5 of all embedded SPIR-V,
   baked by `tools/embed_spirv.py` as `embeddedShadersHash()`) + device (vendor/device/driver +
   pipeline-cache UUID) + model hash. Any mismatch discards the file and recomputes.
4. **Multi-variant.** The document holds one variant per cache-affecting configuration (precision,
   `flatLayout`, `gpuIslandFold`, `fp32Tensors`, and the conv-kernel hints). A matching variant is reused;
   a new configuration appends a variant, leaving the others intact. `tuning` effort is not part of the
   key — it only governs whether missing autotune entries get measured.
5. **`--no-flat` / `--no-fold-islands`** move behind `Hint::FlatLayout` / `Hint::GpuIslandFold` (default
   On = fastest), replacing the `noFlatOps` / `foldGpuIslands` config bools.

`run()` remains pure execution — all compilation, prepacking, and autotuning happen at load.

## Consequences
- The cache auto-heals on a device/driver/shader/model change; there is nothing to invalidate by hand.
- Switching precision back and forth reuses cached variants instead of recompiling; the file grows by one
  variant per configuration used.
- Legacy `VKNNCAC1` files fail the MessagePack parse and are transparently recomputed and rewritten. Old
  `config.json` `cacheMode` keys are ignored with a warning; old `tuning` values still parse via aliases.
- `tuning` selects conv kernels, which can differ by one fp16 ULP (outputs stay cos ≈ 1.0, same argmax) —
  it is not a bit-exact knob, unlike `precision` and `priority`. Verified warm==cold bit-exact and
  multi-variant reuse bit-exact on both target devices.
