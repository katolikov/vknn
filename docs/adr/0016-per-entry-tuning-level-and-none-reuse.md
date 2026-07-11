# ADR-0016: Per-entry autotune level; `--tuning none` reuses a cached tune

## Status
Accepted (2026-07-11). Amends [ADR-0009](0009-cache-tuning-redesign.md) point 4.

## Context
ADR-0009 made caching always-on and stated that `tuning` effort "is not part of the key — it only
governs whether missing autotune entries get measured." A warm load at the same level already reuses the
stored tune (measured: resnet50 `--tuning heavy` cold-vs-warm session load **5220 ms → 1549 ms**, a
3.4x load-time win, reusing 29 tuning entries with no re-measurement). Two gaps remained in that design:

- **Gap A — `--tuning none` ignored a valid cached tune.** Each pick site returned its hardcoded default
  kernel when `tuning == None` *before* consulting the cache, so a warm `--tuning none` re-ran on the
  default kernels even when a fast/heavy tune for this exact device was on disk. `none` conflated "run no
  new measurement" with "ignore any stored measurement."
- **Gap B — the tuning level was not part of any cache identity.** A cache written at `fast` and then
  loaded at `heavy` reused the fast entries and never explored the heavier candidate set (conv
  local-size `{32,64,128,256}` vs `{64,128,256}`; OCB `{2,3}` vs `{2}`). Measured: after a `fast` cold
  load, a `heavy` load reused the fast entries in ~826 ms instead of re-sweeping — `heavy` silently
  delivered `fast`-quality kernels.

Making the level part of the *whole-file* guard would fix Gap B but nuke the entire cache (and re-pack
every weight) on any level change — the wrong granularity.

## Decision
1. **Per-entry tuning level.** Each autotune entry records the `Tuning` level it was measured at, stored
   in an **append-only** `"tunelvl"` companion map in the MessagePack variant (parallel to `tune`). An
   older reader ignores the key; an older cache omits it and the decoder leaves the level map empty. The
   level is **not** part of the variant key or the whole-file guard — raising the level re-measures only
   the affected entries, never the weights.
2. **Cache-first pick.** Every pick site (`conv` local-size / WTILE / OCB / conv-gemm / Winograd,
   `conv_gemm`, `matmul`) consults the cache before falling back to its default (`VkOpEnv::reuseTuned`):
   - Reuse a cached entry under `Tuning::None` at any level — **`none` runs no new sweep but honors a
     stored measurement**.
   - Otherwise reuse only when the entry's measured level is `>=` the requested level; a lower-level
     entry (fast cached, heavy requested) is **re-swept** with the heavier candidate set and its stored
     level upgraded.
   - A legacy entry with no recorded level reads back as `Fast` (the production default nearly every
     pre-existing cache was measured at), so warm `fast` loads keep their benefit and only `heavy`
     re-sweeps it once.
   Value-specific eligibility gates (the OCB `kChoiceLds3x3` shape gate, conv-gemm residual/overflow,
   matmul tile-fits/bounds) still apply to a reused entry.
3. **`--no-cache` is unchanged.** With no cache read, `--tuning none` produces the deterministic default
   kernels — the byte-gate configuration. Every CNN / yonosplat determinism gate pins
   `--tuning none --no-cache`, so those gates are unaffected.

## Consequences
- **`--tuning none` on a warm device reuses the cached fast/heavy tune** (device-tuned kernels, faster
  warm inference) rather than the default kernels. Verified on both target GPUs: `--tuning none` with a
  heavy cache present reproduces the heavy output, not the default output; `--tuning none --no-cache`
  still reproduces the deterministic default output byte-for-byte (`resnet50` / `mobilenetv2`).
- **`heavy` now re-sweeps a `fast` cache.** Verified: after a `fast` cold load, a `heavy` load is a full
  re-sweep (~2650 ms, weights already cached) rather than an ~826 ms fast reuse, and a second `heavy`
  load then reuses the upgraded entries (~1266 ms).
- **Reusing a given cache is deterministic** (same cache file → same picks → byte-identical output).
- Cross-device safety is unchanged: the per-signature `gpuTag` prefix and the whole-file device guard
  still keep one device's picks off another.
- The `"tunelvl"` field is append-only; the per-variant MessagePack map count moves from 15 to 16.
