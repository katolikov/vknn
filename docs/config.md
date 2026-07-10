# Configuration (`vknn::Config`)

`vknn::Config` is the single struct that controls backend selection, precision,
caching, zero-copy, profiling, and autotuning for a `vknn::Session`. It is defined
in [`include/vknn/config.h`](../include/vknn/config.h) and parsed/serialized in
[`src/core/config.cpp`](../src/core/config.cpp).

The engine reads no environment variables. Every knob, including the research/debug ones, is
a `Config` field or a `Config::setHint(Hint, value)` (MNN-style). The defaults select the
best-and-fast configuration.

A `Config` can be built three ways:

- Default-constructed in C++ (`vknn::Config cfg;`) and field-assigned.
- Loaded from a JSON file: `Config::fromJsonFile(path)`.
- Loaded from a JSON string: `Config::fromJsonString(json)`.

JSON parsing is lenient. Every field is optional, and any field absent from the
JSON keeps its struct default. An unparseable or non-object document yields a
fully default `Config`. A missing file path logs a warning and returns defaults
rather than throwing.

---

## Fields

All defaults below are the C++ member initializers in `struct Config`.

| Field | JSON type | Accepted values | Default | Meaning |
|---|---|---|---|---|
| `backend` | string | `"VULKAN"`, `"CPU"` (case-insensitive: `vulkan`/`cpu` also accepted) | `"VULKAN"` | Primary backend the planner prefers for each node. Unrecognized strings fall back to `CPU`. |
| `fallback` | array of string | same tokens as `backend` | `["CPU"]` | Ordered list of backends to try when the primary declines a node. CPU is always an implicit final fallback regardless of this list. Providing the key replaces the whole list. |
| `allowCpuFallback` | bool | `true` / `false` | `true` | If `false`, nodes that no listed backend accepts are an error instead of silently running on CPU. |
| `precision` | string | `"low"`, `"normal"`, `"high"` (aliases `"fp16"`→low, `"mixed"`→normal, `"fp32"`→high; unknown → low) | `"low"` | Quality tier for the Vulkan backend. `low` = fp16 storage + fp32 accumulation everywhere. `normal` = fp16, but a built-in geometry-tail set (`mixedPrecisionFp32Tensors()`) is kept fp32 — selective fp32 (a no-op for models without those tensors). `high` = full fp32 storage. Under `low`/`normal` every fp16 narrowing store rounds to nearest even (the `RoundingModeRTE` SPIR-V execution mode), so per-store error is unbiased and does not accumulate a directional drift across depth. See `fp32Tensors` to override the `normal` set. |
| `maxSubmitNodes` | int | ≥ 0 | `500` | Split a GPU segment larger than this into chunks of this many nodes, each its own submit, so no single submit trips the GPU watchdog. `0` disables chunking. Only the very large YoNoSplat-class transformer needs it; results are numerically identical. |
| `decodeChainSteps` | int | ≥ 1 | `1` | Decode iterations the decode bucket's GPU segment records as one command-buffer chain (`Session::configureDecodeChain`): one submit + one fence per this many greedy tokens, with on-device feedback (argmax id → `input_ids`, position + 1, mask slot) between iterations. `1` records the single-step stream unchanged; only an explicitly configured bucket chains. The token stream is bit-identical to the single-step loop; chained decode is argmax-only. |
| `freeWeightsAfterUpload` | bool | `true` / `false` | `true` | Free host weight buffers after they are uploaded to the device, reclaiming the full weight blob. `run()` never reads graph initializers, so this is safe; needed to fit large (e.g. 965M-param) models on-device. |
| `priority` | string | `"low"`, `"normal"`, `"high"` | `"normal"` | GPU queue scheduling priority (Vulkan `VK_KHR/EXT_global_priority`). `normal` reproduces the default device-creation path; `low`/`high` request the matching queue tier. Scheduling only — never changes numerical output; an inert no-op on a device without a global-priority extension. |
| `cacheFile` | string | filesystem path | `""` → `<model>.cache` | Per-model MessagePack cache holding the compiled pipelines, prepacked/Winograd weights, and conv autotune table. Empty resolves to `<model>.cache` next to the model. Caching is always on: a warm start reloads it (skipping shader compilation, weight prepacking, and autotuning), and it auto-heals when stale. See [Caching](#caching). |
| `cacheDir` | string | filesystem path | `"vknn_cache"` | Fallback location for the cache when the session is built from an in-memory graph (no model path to anchor `cacheFile`). |
| `noCache` | bool | `true` / `false` | `false` | Debug: skip all cache read/write, recompiling + re-tuning on every load (for cold-compile measurement). |
| `profile` | bool | `true` / `false` | `false` | Enable the per-op profiler (GPU timestamp queries + CPU timing); the table is available via `session.profiler()`. |
| `verbosity` | int | `0`, `1`, `≥2` | `1` | Log level. `0` → Warn, `1` → Info, `≥2` → Debug. Applied by `Config::applyLogLevel()`. |
| `layerDump` | bool | `true` / `false` | `false` | Dump every layer's output tensor to disk for debugging. |
| `layerDumpDir` | string | filesystem path | `"vknn_dump"` | Destination directory for layer dumps (used only when `layerDump` is `true`). |
| `tuning` | string | `"none"`, `"fast"`, `"heavy"` (aliases `"off"`→none, `"thorough"`→heavy) | `"fast"` | Load-time conv autotune effort. `none` uses the default kernel (no per-shape measurement), `fast` does a quick candidate sweep, `heavy` an exhaustive one. Effort only — never changes numerical output beyond kernel-selection fp rounding (outputs stay cos ≈ 1.0, same argmax); the chosen kernels are cached and reused on a warm start. |
| `cpuThreads` | int | ≥ 1 | `4` | Worker threads the CPU backend partitions its hot output loops across (Conv, ConvGemm, FusedDwPw, Gemm, MatMul, the elementwise family). Only loops with disjoint per-iteration outputs and no cross-iteration accumulation are split, so results are bit-identical for any thread count — the CPU backend stays the byte oracle for the GPU path. `1` runs every loop inline. A fixed default rather than a probe of the host, so a plan runs the same way on every machine. |
| `flatLayout` | bool | `true` / `false` | `true` | Flat row-major GPU layout pass that keeps generic head ops (Transpose/Slice/Concat/Binary/Softmax) on the GPU. On by default (fastest). `false` (CLI `--no-flat`) forces NC4HW4 / CPU paths — advanced. Backed by `Hint::FlatLayout`. |
| `gpuIslandFold` | bool | `true` / `false` | `true` | Fold tiny GPU op-islands between CPU segments onto the CPU (fewer boundary round-trips). On by default (fastest). `false` (CLI `--no-fold-islands`) keeps every supported op on the GPU — verification runs use it so the fallback count is meaningful. Backed by `Hint::GpuIslandFold`. |
| `timing` | bool | `true` / `false` | `false` | Print per-stage timing (pack / submit+gpu / unpack, plus `Session::run` bind/segments/collect). |
| `debugSegments` | bool | `true` / `false` | `false` | Trace per-segment and per-CPU-op execution. |
| `disableVkOps` | string | e.g. `"Add,Conv"` | `""` | Comma list of op types forced onto the CPU backend (exercises the CPU-fallback path). Entries match whole op-type names: `"Conv"` does not disable `ConvTranspose`. |
| `dumpTensors` | string | e.g. `"layer3"` | `""` | Comma list of tensor-name substrings to dump to disk after a run. |
| `fp32Tensors` | string (C++ only, not serialized to JSON) | e.g. `"/enc/MatMul_,-camera_head"` | `""` | Advanced override of the selective-fp32 set: tensor-name substrings (leading `-` excludes) kept in fp32 storage under fp16 compute. Empty + `precision:"normal"` uses the built-in geometry-tail preset; a non-empty value replaces it (and also applies under `precision:"low"`). |
| `inputShapes` | `map<string, Shape>` (C++ only, not serialized to JSON) | input tensor name → full concrete shape, e.g. `{{"pixel_values",{1,3,224,224}}}` | `{}` (empty) | Declared concrete shapes for graph inputs on the **ONNX-load path** (`createFromOnnx` / `Model::load` from `.onnx`), keyed by input name. Each listed input has its dynamic (negative) dims resolved from the declared shape; an input absent here falls back to `batch = 1` on an unnamed or batch-named leading axis, and any other dynamic axis with no declaration is a hard error (never a silent freeze to 1). Empty = the fixed-shape / batch-only path (byte-identical to before). **Ignored for a `.vxm` session** — a `.vxm` already has its shapes baked at compile time; set them there with `vknn_compile --dim` / `--shape` / `--bucket` (see below). |
| `dimBindings` | `map<string, int64_t>` (C++ only, not serialized to JSON) | ONNX `dim_param` name → value, e.g. `{{"past_sequence_length",256},{"sequence_length",1}}` | `{}` (empty) | Symbolic-dimension bindings for the **ONNX-load path**, keyed by the ONNX `dim_param` name. Every dynamic input axis whose `dim_param` — a bare symbol, an integer literal, or a compound expression like `past_sequence_length + sequence_length` — resolves entirely from these bindings is filled automatically, so a many-input dynamic model (a with-past decoder's 51 inputs) needs a couple of bindings instead of one `inputShapes` entry per tensor. `inputShapes` (a per-tensor concrete shape) overrides a binding for that tensor; an unnamed or batch-named (`N`/`B`/`*batch*`) leading axis still falls back to `batch`, while a leading axis with any other `dim_param` name must be bound (an unbound one is a hard error, never a silent freeze to 1). Empty = the batch-only path. **Ignored for a `.vxm` session**; set the equivalent with `vknn_compile --dim`. |

### The `vknn_compile` flags

`vknn_compile <model.onnx> <out.vxm> [flags]` runs the import passes ahead of time and
writes a pre-optimized `.vxm`. Its flags cover optimization level, weight precision,
declared shapes, the quantization path, and the support report. A fixed-shape,
default-optimization model needs none of them. The multi-graph form
`vknn_compile <out.vxm> --graph "FILE.onnx[;segments]" [--graph ...]` takes **no positional
input model** — each `--graph` occurrence names its own source file (see the `--graph` row).

| `vknn_compile` flag | Meaning |
|---------------------|---------|
| `--fp16` | Store weights as fp16 in the `.vxm` (default fp32). Halves the file and the runtime host repack; the run-time compute precision is a separate `Config::precision` knob. |
| `-O0` / `-O1` / `-O2` / `-O3` (or `--opt N`) | Optimization level, default `-O1`. `-O0` = no optional fusion (reference); `-O1` = the general pointwise fusion (the bit-exact production set); `-O2`/`-O3` = + the experimental Squeeze-Excite and depthwise+1×1 fusions. |
| `-Os` | A superset of `-O3`: all of its fusion plus calibration-free int4 weight quantization. Per-op-class defaults quantize MatMul weights to int4 with a scale group and keep the activation-salient outlier columns fp16, while Conv/Gemm weights load-dequant to fp16; a per-layer relative-error guard keeps a hostile layer fp16, and native int4 GPU MatMul kernels run the quantized weights. Emits a **VXM5** container — a subtag over the exact VXM3/VXM4 body, so a non-quantized compile stays byte-identical VXM3/4. The Qwen instruct model is ≈2.4× smaller as int4. Accepts a `.vxm` as input to re-quantize an already-compiled model; a multi-bucket `.vxm` re-quantizes every bucket over the shared initializer pool. |
| `--quant-samples N` | Calibration samples for the int4 min-MSE scale-group search + bias correction (`-Os`). `0` = weight-only (uniform column weighting, no calibration) — required for a multi-bucket compile so every bucket's scales stay identical and the payloads content-dedup. |
| `--quant-group N` / `--quant-outliers N` / `--conv-group N` / `--conv-outliers N` | Int4 quantization tuning knobs over the per-op-class defaults (`-Os`): the scale-group width and the count of activation-salient outlier columns kept fp16, for MatMul weights (`--quant-*`) and Conv/Gemm weights (`--conv-*`). |
| `--calib F0[,F1,...]` | Calibration sample files for the `-Os` min-MSE scale search + bias correction — one sample file per graph-input occurrence. |
| `--[no-]fuse-pointwise` / `--[no-]fuse-se` / `--[no-]fuse-dwpw` / `--[no-]lower-conv` | Override a single fusion/lowering pass on top of the `-O` level. |
| `--strict-fuse` | Round every fused step so `fused == unfused` byte-identical (the byte-verification mode). The default fast mode fp32-chains each fused unit and rounds once per stored stream — faster, and at least as accurate as the unfused graph. |
| `--no-dequantize` | Keep `QuantizeLinear` / `DequantizeLinear` / QLinear ops instead of folding them to float. Default (off) compiles a quantized checkpoint to a plain float graph (see [limitations.md §6](limitations.md)). |
| `--batch N` | Leading-axis (batch) fallback for a dynamic batch dim. Default `1`. |
| `--dim NAME=VALUE` (repeatable) | Bind a symbolic input dimension by its ONNX `dim_param` name, e.g. `--dim sequence_length=1 --dim past_sequence_length=256`. Every dynamic input axis whose `dim_param` (a bare symbol, an integer literal, or a compound like `past_sequence_length + sequence_length`) resolves from the bindings is filled automatically — a couple of `--dim` replace one `--shape` per tensor on a many-input decoder. Populates `Config::dimBindings`. `--shape` overrides `--dim` for a given tensor. |
| `--list-dims` | Import the model and print its free symbolic input dimensions (the `dim_param` names to bind with `--dim`), then exit without compiling. |
| `--shape NAME=D0xD1x...` (repeatable) | Declare one graph input's full concrete shape, resolving every dynamic axis of that input, e.g. `--shape pixel_values=1x3x224x224`. Populates the same map as `Config::inputShapes`; overrides `--dim` for that tensor. |
| `--bucket "NAME=D0x...;dim:NAME2=VALUE;..."` (repeatable) | Declare **one plan bucket** per occurrence: the model is compiled once per bucket over a fresh import, and the buckets share one content-deduped initializer pool in a multi-bucket `.vxm`. A bucket segment is either a per-tensor `NAME=D0xD1x...` shape or a `dim:NAME=VALUE` symbolic binding (so prefill vs decode plans differ only by their bound `sequence_length` / `past_sequence_length`). `--batch` / `--shape` / `--dim` are the shared fallback under every bucket. With no `--bucket`, exactly one bucket is written (a legacy single-graph `.vxm`, byte-identical to before). `--bucket` requires an ONNX input. |
| `--graph "FILE.onnx[;NAME=D0x...;dim:NAME2=VALUE;...]"` (repeatable) | **Multi-graph form**: each occurrence compiles **one bucket from its own source file**, with its own shape/dim segments (the `--bucket` segment syntax) over the shared `--batch`/`--shape`/`--dim` fallback. Occurrences may repeat a file at different shapes (a decoder's prefill + decode plans) or name different files (a vision tower + its decoder); every bucket lands in one `.vxm` over a content-deduped initializer pool, and the device weight pool shares GPU copies by payload content, so N buckets over one weight set cost one copy ([ADR-0014](adr/0014-multi-graph-vxm.md)). The only positional argument is the output `.vxm`; mutually exclusive with `--bucket`; a `.vxm` cannot be a graph source. See [running-a-vlm.md](running-a-vlm.md) for a five-bucket VLM compile. |
| `--support-report <out.json>` | Write the per-node backend assignment (node, op, backend, and the refusal reason for every CPU node) from the engine's own capability model (`vkSupportSurvey` — the exact gate the device runs), then continue the compile. Consumed by `tools/check_model_support.py --engine-report`. On a multi-bucket/multi-graph compile, one report per bucket: `out.json` for bucket 0, then `out.bucketN.json`. |

A multi-bucket `.vxm` streams its buckets at load (host memory peaks at one bucket's
weights, not the file total); `Session::run()` selects the bucket by the bound input
**names + shapes** and rejects an unmatched key with `Status::InvalidArgument`. A
one-graph multi-shape session keeps the legacy positional single-input forgiveness; a
multi-graph session dispatches named inputs strictly. `Session::bucketKeys()` and the
indexed `inputInfo(bucket)` / `outputInfo(bucket)` describe each bucket. An
ONNX-loaded session can add a bucket at run time with `Session::prepareShapes(shapes)`
(a `.vxm` session returns `Status::Unsupported`). See
[limitations.md §1](limitations.md) for the full contract.

Run-time-only knobs are `Config` fields set on the **session**, not compile flags: they
are in the field table above and reach the CLI through the runner examples. `vknn_classify`
exposes `--precision`, `--tuning`, `--winograd`, `--profile`, and `--layer-dump`
([below](#how-the-classify-example-exposes-config-flags)); `vknn_run_io` adds
`--disable-vk-ops`, `--no-fold-islands`, and `--no-flat` for exercising the fallback and
layout paths.

### Enum reference

The string tokens map onto these enums (from `config.h` / `tensor_format.h`):

```cpp
enum class BackendKind { Vulkan, Cpu };
enum class Precision   { Low, Normal, High };  // "low" fp16 | "normal" fp16 + selective fp32 | "high" fp32
enum class Priority    { Low, Normal, High };  // GPU queue global-priority tier (scheduling only)
enum class Tuning      { None, Fast, Heavy };  // load-time conv autotune effort
enum class TensorFormat : uint8_t { NCHW, NHWC, NC4HW4, Auto, Unknown };  // Auto: declared-boundary zero-copy sentinel (bytes already device-native)
```

### Conv kernel knobs — `Config::setHint(Hint, Mode)`

Every conv kernel-selection knob goes through one MNN-style hint API: `setHint(Hint, Mode)`. The
defaults are the production kernels; normal use needs none of these. There are no environment variables.

```cpp
enum class Hint {
  Winograd        = 0,  // 3x3 Winograd selection      (Auto / On / Off)
  WinogradVariant = 1,  // Winograd matmul impl         (TiledGemm / Fused / FusedSplit / FullyFused / SubgroupGemm)
  WinogradUnit    = 2,  // Winograd output tile         (F23 / F43)
  DirectConv3x3   = 3,  // direct 3x3 kernel            (DirectAuto / RegisterTiled / LdsHalo)
  FlatLayout      = 4,  // flat-layout GPU pass         (On / Off, default On)
  GpuIslandFold   = 5,  // fold tiny GPU islands to CPU (On / Off, default On)
  MatMulViewFold  = 6,  // MatMul operand-view fold at load  (On / Off, default On)
  RopeFusion      = 7,  // rotate-half RoPE chain fusion at load (On / Off, default On)
  FusedAttention  = 8,  // single-query decode-attention fusion at load (On / Off, default On)
  KvConcatFold    = 9,  // per-token KV-cache Concat fold into split-source attention (On / Off, default Off — its rows-only present output is incompatible with the engine-resident KV link; opt in for a host-cache decode loop)
};
// One Mode enum holds every value; the Hint picks the knob, the Mode the value. (Autotune effort is
// a top-level Config::tuning field, not a Hint.)
enum class Mode {
  Auto = 0, On = 1, Off = 2,                                                  // Hint::Winograd, FlatLayout, GpuIslandFold, MatMulViewFold, RopeFusion, FusedAttention, KvConcatFold
  TiledGemm = 0, Fused = 1, FusedSplit = 2, FullyFused = 3, SubgroupGemm = 4, // Hint::WinogradVariant
  F23 = 0, F43 = 4,                                                           // Hint::WinogradUnit
  DirectAuto = 0, RegisterTiled = 1, LdsHalo = 2,                             // Hint::DirectConv3x3
};
cfg.tuning = Tuning::Heavy;                   // maximum autotuning (a Config field, not a hint)
cfg.setHint(Hint::WinogradUnit, Mode::F43);   // force F(4,3) Winograd
int v = cfg.hint(Hint::WinogradUnit);         // read back (0 if unset)
```

`MatMulViewFold`, `RopeFusion`, and `FusedAttention` are load-time graph-fusion passes that never change a
compiled `.vxm`; each is keyed into the plan-cache variant, so flipping the hint reselects (or recomputes) a
variant instead of ever serving a stale plan.

In JSON, the common knobs have named keys (`"winograd": "off"`, `"tuning": "heavy"`, `"flatLayout": false`);
the raw hint form is an array indexed by the `Hint` value, one entry per hint in enum order
(Winograd, WinogradVariant, WinogradUnit, DirectConv3x3, FlatLayout, GpuIslandFold, MatMulViewFold,
RopeFusion, FusedAttention, KvConcatFold) — e.g. `"hints": [2, 0, 0, 0, 2, 1]` sets the first six and
leaves the rest at their defaults.

`NC4HW4` is the internal Vulkan packed layout (channels in vec4 blocks); the engine I/O is `NCHW`.

---

## Caching

Warm starts load a per-model cache (`cacheFile`, default `<model>.cache`) so a second run skips shader
compilation, conv autotuning, and weight prepacking. Caching is always on — it is not a user knob.

- **Format** — MessagePack (compact, self-describing binary; inspect with any msgpack tool, e.g.
  `python3 -c "import msgpack; print(msgpack.unpackb(open('m.cache','rb').read(), strict_map_key=False, raw=True).keys())"`).
- **Self-validating** — the file records a format version, an md5 of all embedded SPIR-V kernels, the
  device (vendor/device/driver + pipeline-cache UUID), and a model hash. If any differ (a driver update,
  a shader change, a different GPU, a different model) the whole file is discarded and recomputed — there
  is nothing to invalidate by hand.
- **Multi-variant** — one file holds an independent entry per cache-affecting configuration (`precision`,
  `flatLayout`, `gpuIslandFold`, `fp32Tensors`, and the conv-kernel hints). Switching precision back and
  forth reuses each variant instead of recompiling; a new configuration appends a new variant.
- **`tuning`** sets only the load-time autotune effort for entries not yet measured; the chosen kernels
  are stored in the variant and reused. To re-tune at a higher effort, delete the cache (or use `noCache`).
- **`noCache`** (CLI `--no-cache`) skips all cache I/O for cold-compile measurement.

`run()` performs no compilation or tuning — every one-time cost is paid at load.

---

## JSON example

A complete config file (`config.json`). Every key here is optional; the example
lists all of them, with non-default values where useful:

```json
{
  "backend": "VULKAN",
  "fallback": ["CPU"],
  "allowCpuFallback": true,
  "precision": "low",
  "priority": "normal",
  "maxSubmitNodes": 500,
  "maxSubmitBindings": 1024,
  "decodeChainSteps": 1,
  "cacheFile": "enc.cache",
  "cacheDir": "vknn_cache",
  "noCache": false,
  "tuning": "fast",
  "freeWeightsAfterUpload": true,
  "flatLayout": true,
  "gpuIslandFold": true,
  "timing": false,
  "profile": false,
  "verbosity": 1,
  "layerDump": false,
  "layerDumpDir": "vknn_dump",
  "debugSegments": false,
  "disableVkOps": "",
  "dumpTensors": "",
  "winograd": "auto",
  "winogradVariant": 0,
  "winogradUnit": 0,
  "directConv3x3": 0
}
```

This is the exact shape `Config::toJson()` produces, so configs round-trip:
serialize a configured `Config`, edit the JSON, reload it.

---

## Loading a config

```cpp
#include "vknn/config.h"
#include "vknn/session.h"
using namespace vknn;

// From a file (returns defaults + a warning if the path is missing):
Config cfg = Config::fromJsonFile("config.json");

// Or from an in-memory string:
Config cfg2 = Config::fromJsonString(R"({ "backend": "CPU", "precision": "fp32" })");

// Field assignment also works directly:
Config cfg3;
cfg3.backend   = BackendKind::Vulkan;
cfg3.precision = Precision::Low;
cfg3.tuning    = Tuning::Heavy;

// Apply the log level implied by verbosity:
cfg.applyLogLevel();

// Hand the config to the runtime:
auto session = Runtime::load("assets/mobilenetv2.onnx", cfg);
```

`Runtime::load(path, cfg[, cacheFile])` resolves `cfg.cacheFile` (empty →
`<model>.cache` next to the model), then dispatches on extension: `.vxm` →
`Session::createFromVxm`, anything else → `Session::createFromOnnx`. It returns
a `std::unique_ptr<Session>`.

---

## How the `classify` example exposes config flags

[`examples/vision/classify.cpp`](../examples/vision/classify.cpp) (`vknn_classify`) layers
file config and CLI overrides. It loads `--config` first (if given), then
lets individual flags override specific fields:

```cpp
Config cfg;
if (!cfgpath.empty()) cfg = Config::fromJsonFile(cfgpath);   // base from JSON
cfg.backend   = backendFromStr(backend);                      // --backend overrides
cfg.precision = precisionFromStr(precision);                  // --precision overrides (low|normal|high)
cfg.cacheDir  = argval(argc, argv, "--cache", cfg.cacheDir.c_str());  // --cache overrides
if (hasflag(argc, argv, "--profile")) cfg.profile = true;     // --profile sets flag
if (hasflag(argc, argv, "--layer-dump")) {                    // --layer-dump DIR
  cfg.layerDump = true;
  cfg.layerDumpDir = argval(argc, argv, "--layer-dump", cfg.layerDumpDir.c_str());
}
```

The CLI flags and the config fields they touch:

| Flag | Config field set | Default |
|---|---|---|
| `--config PATH` | loads the whole `Config` via `fromJsonFile` | (none) |
| `--backend NAME` | `backend` (`vulkan`/`cpu`) | `vulkan` |
| `--precision P` | `precision` (`low`/`normal`/`high`; `fp16`/`fp32` aliases) | `low` |
| `--cache DIR` | `cacheDir` | struct default |
| `--profile` | `profile = true` | off |
| `--layer-dump DIR` | `layerDump = true`, `layerDumpDir = DIR` | off |
| `--winograd MODE` | `setHint(Hint::Winograd)` (`auto`/`on`/`off`) | `auto` |
| `--tuning LEVEL` | `tuning` (`none`/`fast`/`heavy`) | `fast` |

Flags that are not config fields (model/input handling and benchmarking):
`--model PATH`, `--input PATH`, `--shape N,C,H,W`, `--golden PATH`,
`--show-graph`, `--bench N`.

Precedence: `--config` loads first, then the explicit flags overwrite
whatever the JSON set. `--backend`, `--precision`, `--winograd`, and `--tuning`
always assign their field or hint (they carry CLI defaults), so a `backend`,
`precision`, `winograd`, or `tuning` value in the JSON file is overridden by the
flag defaults unless the matching flag is passed. This applies when mixing
`--config` with these flags.

Example invocations:

```bash
# Pure flags, no file:
vknn_classify --backend vulkan --precision fp16 --profile

# File config plus a cache override:
vknn_classify --config config.json --cache /data/local/tmp/vxrt/cache

# CPU reference run with a golden comparison:
vknn_classify --backend cpu --precision fp32 --golden assets/golden.bin
```
