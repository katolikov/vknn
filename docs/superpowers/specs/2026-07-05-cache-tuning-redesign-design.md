# Cache & Tuning redesign — design (sub-project ① of 3)

Date: 2026-07-05
Status: approved (design), pending spec review
Scope: this spec covers **sub-project ①** only. ② converter UX and ③ docs pass are separate specs.

## Context

VKNN warm-starts from a per-model `<model>.cache`. Today two user knobs govern it:
`--cache-mode off|tune|full` (`CacheMode`, what to persist) and `--tuning off|fast|thorough`
(`Hint::Tuning`, conv-autotune effort). The cache file (`src/backend/vulkan/vk_backend.cpp:33-131`,
`486-586`) is a hand-rolled binary container (magic `VKNNCAC1` = VkPipelineCache blob + a
length-prefixed weight/tune blob). Its only integrity check is the 8-byte magic: the VkPipelineCache
blob self-validates in the driver (UUID), but the prepacked-weight and autotune blobs have **no**
device/model/code guard beyond the per-model filename and device-keyed tune signatures. Autotuning
already runs at **load** (`op->prepare()` from `compileSegment`), not in `run()`.

## Goals

1. Collapse the two knobs into one: `--tuning none|fast|heavy` = autotune effort only. Remove
   `--cache-mode` and the `CacheMode` concept entirely.
2. Caching is always-on, comprehensive, and **self-validated**: a load either safely reuses a cache
   or transparently recomputes and rewrites it. Never a user knob (a `--no-cache` debug flag remains
   for cold-compile measurement).
3. **Multi-variant** cache: one file holds independently-keyed variants (per cache-affecting config),
   so switching e.g. precision back and forth never recompiles the other variant.
4. Cache integrity key = format version + **kernel-md5** (hash of all embedded SPIR-V) + device sig +
   model hash; per-variant key = the config that changes compiled artifacts.
5. Cache on disk is **MessagePack** (serialize + deserialize), inspectable with standard msgpack
   tooling. Library vendored as a `third_party` submodule.
6. Fold the obscure `--no-flat` / `--no-fold-islands` toggles behind the `Hint` mechanism with
   fastest defaults (same config surface as the cache key).
7. `run()` stays pure execution — all compile/tune/pack happens at load.

## Non-goals

- Converter UX (sub-project ②) and the docs pass (sub-project ③).
- Changing kernels, autotune search spaces, or numerical behavior. The redesign is bit-exact.

## Design

### 1. The `tuning` knob

- New `include/vknn/tuning.h`: `enum class Tuning { None, Fast, Heavy }` + `tuningFromStr`
  (`"none"→None`, `"fast"→Fast`, `"heavy"→Heavy`; **backward-compat aliases** `"off"→None`,
  `"thorough"→Heavy`). Default `Fast`.
- `Config::tuning` (default `Tuning::Fast`) replaces both `Config::cacheMode` and the old
  `Hint::Tuning`. CLI `--tuning none|fast|heavy` in `run_io`.
- `conv.cpp` reads `env.tuning` derived from `Config::tuning`: `None`→ no measurement (default
  kernel), `Fast`→ current fast candidate set, `Heavy`→ current thorough set. (The three
  `env.tuning == Mode::NoTune/Thorough` branches in `conv.cpp:193,209,216,250,312,347,380` map 1:1.)
- **Removed**: `include/vknn/cache_mode.h`, `CacheMode`, `Config::cacheMode`, the
  `cachesPipeline/cachesTuning/cachesWeights` predicates, `--cache-mode`, and `Hint::Tuning` from
  `hint.h` (the `Mode` reuse for tuning goes away; Winograd/DirectConv hints stay).
- JSON: parse `"tuning"` into `Config::tuning`; a legacy `"cacheMode"` key is accepted-and-ignored
  with a one-line `VKNN_WARN` (so old `config.json` / `benchmark/configs/*.json` still load).

### 2. Caching always-on

- When a cache path is resolvable (`<model>.cache`, or `Runtime::load(cacheFile)`), the backend
  always reads a valid cache and always writes back any newly-built variant. In-memory graphs (no
  path) stay memory-only, as today.
- `Config` gains `bool noCache = false` (debug; CLI `--no-cache`) to disable all cache I/O for
  cold-compile measurement. Not serialized to JSON by default.

### 3. Cache format (MessagePack) + validation + multi-variant

Vendored lib: **`ludocode/mpack`** (single `mpack.c`/`mpack.h`, lean, first-class `bin` blobs) as a
`third_party/mpack` submodule; `msgpack/msgpack-c` is the batteries-included alternative — final pick
confirmed at spec review. Wrapped behind a small `vk_cache_codec.{h,cpp}` so the rest of the backend
sees typed structs, not the msgpack API.

Document schema (logical):

```
{ format:      u32,                       // bump on any structural change; != current -> whole file invalid
  kernelHash:  str,                       // md5 of all embedded SPIR-V (see §4)
  device:      { vendorId:u32, deviceId:u32, driverVersion:u32, pipelineCacheUUID:bin },
  model:       str,                       // hash of the compiled graph (env.modelTag today)
  variants: [
    { key:     { precision:str, flat:bool, foldIslands:bool, fp32Tensors:str,
                 winograd:int, winogradVariant:int, winogradUnit:int, directConv3x3:int },
      pipeline: bin,                       // serialized VkPipelineCache blob
      weights:  { <weightName>: bin },     // prepacked / Winograd-transformed weights (fp16|fp32 as built)
      tune:     { <sig>: i32 } } ]         // per-variant conv autotune table
}
```

**Load** (`loadUnified`):
1. Parse msgpack. Parse failure (e.g. a legacy `VKNNCAC1` file) → treat as absent.
2. If `format != kCacheFormat`, `kernelHash != embeddedShadersHash()`, `device != this device`, or
   `model != this model` → discard the whole document, start empty (full recompute + rewrite).
3. Else compute `currentKey` from the effective `Config` and find the variant with `key ==
   currentKey`. Found → feed its `pipeline` into the `VkPipelineCache`, its `weights` into the
   `WeightCache`, its `tune` into the tune table. Missing → this variant is built fresh at load and
   **appended**; the other variants are preserved untouched.

**Save** (`saveCaches`/`finalize`): re-serialize the full document (all variants, the current one
updated) via msgpack; keep the existing "skip the write if the serialized bytes are unchanged"
optimization.

Rationale for keys: `kernelHash` + `device` + `model` guard the whole file (all variants share SPIR-V,
GPU, and graph). Only the per-variant `key` fields change the compiled pipeline / prepacked weights.
`tuning` effort is **not** in the key — it only governs whether missing tune entries get measured; the
tune table is stored per variant and accumulates best-known values.

The per-variant `key` = every config field the survey classifies as affecting compiled artifacts
(`config_struct.h`): `precision`, `noFlatOps`, `foldGpuIslands`, `fp32Tensors`, and the conv-kernel
hints `Winograd`, `WinogradVariant`, `WinogradUnit`, `DirectConv3x3`. (`priority`, submit-chunking,
caches, and all debug/IO fields do not affect compiled artifacts and are excluded.)

### 4. Kernel hash

`tools/embed_spirv.py` already generates `vknn_embedded_shaders.cpp` (`embeddedShaders()` name→SPIR-V
map, consumed at `vk_pipeline.cpp:38`). Extend it to compute an md5 over every SPIR-V blob in
name-sorted order and emit `const char *embeddedShadersHash();` (declared in `vk_pipeline.h`). This is
the "md5sum of all kernels": any `.comp`/`.glsl` change alters the embedded SPIR-V → the hash changes
→ every stale cache is invalidated on next load. Deterministic and zero runtime cost.

### 5. `--no-flat` / `--no-fold-islands` → hints

Add `Hint::FlatOps` and `Hint::FoldIslands` (`Mode::On`/`Off`, default `On` = fastest), matching the
existing `Winograd` hint pattern. The hints are the single source of truth: the `Config::noFlatOps` and
`Config::foldGpuIslands` bools are removed, and the (few) read sites switch to
`cfg.hint(Hint::FlatOps, (int) Mode::On) == (int) Mode::On` (same for `FoldIslands`). `run_io` keeps
`--no-flat` / `--no-fold-islands` as advanced flags that set the hints to `Off`. Defaults unchanged
(flat on, fold on), so default behavior is the fastest path. Debug flags (`disableVkOps`,
`dumpTensors`, `layerDump`, ...) are untouched. The variant key reads these two from the hints.

### 6. Load/run split

Autotune + pipeline build + weight prepack already occur in `prepare()` at load. Add a lightweight
guard (a `bool sealed_` flag on the segment env set after load) so an accidental
measure/compile/prepack in `run()` asserts in debug builds. No functional change; documents the
invariant.

### 7. Migration

Legacy `VKNNCAC1` caches fail the msgpack parse → treated as absent → recomputed and rewritten as
msgpack. No manual migration or version-straddling reader. `format` starts at 2 to signal the break.

## Components / boundaries

- `include/vknn/tuning.h` — the `Tuning` enum + parser (mirrors `precision.h` / `priority.h`).
- `vk_cache_codec.{h,cpp}` (new) — typed cache document ⇄ MessagePack bytes; the only file that
  touches the msgpack API. Testable host-side with no device.
- `WeightCache` + the pipeline/weight handling in `vk_backend.cpp` — reshaped around the typed
  document and the variant lookup; the `VkPipelineCache` and prepack logic are unchanged internally.
- `tools/embed_spirv.py` + `vk_pipeline.{h,cpp}` — the kernel hash.
- `config_struct.h` / `config.cpp` / `hint.h` / `run_io.cpp` — the knob + hint surface.

## Testing

- **Host unit** (`tests/`): `vk_cache_codec` round-trips a document with binary blobs, float weights,
  nested maps, and multiple variants (serialize→deserialize→equal); validation logic
  (format/kernelHash/device/model mismatch → invalid; variant key match/miss).
- **Device (both 960 + 940)**:
  - warm load == cold load, bit-exact, on the CNN suite + a large model.
  - precision A → B → A: second A reuses its variant (no recompile), B built once; file holds both.
  - simulated `embeddedShadersHash()` bump → whole cache invalidated + rebuilt.
  - `--tuning none|fast|heavy` all bit-exact (only load time differs).
  - `--no-flat` / `--no-fold-islands` via hints reproduce today's behavior; keyed as distinct variants.
- **Regression**: full CNN suite bit-exact fused==nofuse, both devices; `run()` timing unchanged.

## Risks

- msgpack lib choice/build wiring (submodule + CMake) — contained to `vk_cache_codec` + `third_party`.
- Variant-key completeness: missing a cache-affecting field → stale reuse. Mitigation: derive the key
  from the survey's category-(a) list and add a host test that changing each keyed field yields a new
  variant.
- Cache file growth with many variants — acceptable (variants are per-config, few in practice); the
  kernelHash/device/model guards already bound it to one model on one device+build.
- Backward-compat JSON: old `cacheMode`/`tuning` values in configs — handled by ignore-with-warn and
  the `off/thorough` aliases.

## File-by-file change list (implementation preview)

- add `include/vknn/tuning.h`; remove `include/vknn/cache_mode.h`
- `include/vknn/config_struct.h`: drop `cacheMode`, `noFlatOps`, `foldGpuIslands`; add `tuning`,
  `noCache`; `hint.h`: drop `Hint::Tuning`, add `Hint::FlatOps`, `Hint::FoldIslands`
- `src/core/config.cpp`: parse/serialize `tuning`; ignore-with-warn legacy `cacheMode`
- `src/backend/vulkan/vk_cache_codec.{h,cpp}` (new); rework cache read/write + variant lookup in
  `vk_backend.cpp`; `WeightCache` serialize/load move to the codec
- `tools/embed_spirv.py` + `vk_pipeline.{h,cpp}`: `embeddedShadersHash()`
- `examples/run_io.cpp`: `--tuning none|fast|heavy`, `--no-cache`; `--no-flat`/`--no-fold-islands`→hints
- `third_party/<mpack|msgpack-c>` submodule + `.gitmodules` + `CMakeLists.txt`
- `tests/`: cache codec + validation unit tests
```
