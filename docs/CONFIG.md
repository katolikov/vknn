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
| `precision` | string | `"low"`, `"normal"`, `"high"` (aliases `"fp16"`→low, `"mixed"`→normal, `"fp32"`→high; unknown → low) | `"low"` | Quality tier for the Vulkan backend. `low` = fp16 storage + fp32 accumulation everywhere. `normal` = fp16, but a built-in geometry-tail set (`mixedPrecisionFp32Tensors()`) is kept fp32 — selective fp32 (a no-op for models without those tensors). `high` = full fp32 storage. See `fp32Tensors` to override the `normal` set. |
| `maxSubmitNodes` | int | ≥ 0 | `500` | Split a GPU segment larger than this into chunks of this many nodes, each its own submit, so no single submit trips the GPU watchdog. `0` disables chunking. Only the very large YoNoSplat-class transformer needs it; results are numerically identical. |
| `freeWeightsAfterUpload` | bool | `true` / `false` | `true` | Free host weight buffers after they are uploaded to the device, reclaiming the full weight blob. `run()` never reads graph initializers, so this is safe; needed to fit large (e.g. 965M-param) models on-device. |
| `cacheFile` | string | filesystem path | `""` → `<model>.cache` | Unified per-model cache file bundling the pipeline-cache blob and the prepacked-weight + autotune blob (container magic `VKNNCAC1`). Empty resolves to `<model>.cache` next to the model. Loading it on a warm start skips shader compilation, conv autotuning, and the Winograd weight transform. |
| `cacheDir` | string | filesystem path | `"/data/local/tmp/vxrt/cache"` | Fallback location for the unified cache when the session is built from an in-memory graph (no model path to anchor `cacheFile`). |
| `cacheMode` | string | `"off"`, `"tune"`, `"full"` | `"full"` | What a warm start reloads from `cacheFile`. `off` recomputes everything every load; `tune` keeps the cheap, deterministic blobs (compiled `VkPipelineCache` + the conv autotune table) but re-uploads weights; `full` also keeps the content-keyed prepacked-weights blob for the fastest warm load (and the largest cache file). |
| `profile` | bool | `true` / `false` | `false` | Enable the per-op profiler (GPU timestamp queries + CPU timing); the table is available via `session.profiler()`. |
| `verbosity` | int | `0`, `1`, `≥2` | `1` | Log level. `0` → Warn, `1` → Info, `≥2` → Debug. Applied by `Config::applyLogLevel()`. |
| `layerDump` | bool | `true` / `false` | `false` | Dump every layer's output tensor to disk for debugging. |
| `layerDumpDir` | string | filesystem path | `"/data/local/tmp/vxrt/dump"` | Destination directory for layer dumps (used only when `layerDump` is `true`). |
| `tuning` | string | `"off"`, `"fast"`, `"thorough"` | `"fast"` | Autotuning level for conv workgroup-size search (sets `Hint::Tuning`). `off` uses defaults, `fast` does a quick search, `thorough` searches more candidates. |
| `winograd` | string | `"auto"`, `"on"`, `"off"` | `"auto"` | 3×3 Winograd F(2,3) selection (sets `Hint::Winograd`). `auto` measures the tiled-GEMM Winograd against the direct kernel per shape and keeps the faster; `on` forces it; `off` always uses the direct kernel. Forcing `on`/`off` skips the per-shape timing, so the kernel choice (and the output bits) is deterministic run-to-run. `auto` requires `tuning` != `off`. |
| `noFlatOps` | bool | `true` / `false` | `false` | Disable the flat-layout GPU pass (forces NC4HW4 / CPU paths). Diagnostic. |
| `timing` | bool | `true` / `false` | `false` | Print per-stage timing (pack / submit+gpu / unpack, plus `Session::run` bind/segments/collect). |
| `debugSegments` | bool | `true` / `false` | `false` | Trace per-segment and per-CPU-op execution. |
| `disableVkOps` | string | e.g. `"Add,Conv"` | `""` | Comma list of op types forced onto the CPU backend (exercises the CPU-fallback path). |
| `dumpTensors` | string | e.g. `"layer3"` | `""` | Comma list of tensor-name substrings to dump to disk after a run. |
| `fp32Tensors` | string | e.g. `"/enc/MatMul_,-camera_head"` | `""` | Advanced override of the selective-fp32 set: tensor-name substrings (leading `-` excludes) kept in fp32 storage under fp16 compute. Empty + `precision:"normal"` uses the built-in geometry-tail preset; a non-empty value replaces it (and also applies under `precision:"low"`). |

### Enum reference

The string tokens map onto these enums (from `config.h` / `tensor_format.h`):

```cpp
enum class BackendKind { Vulkan, Cpu };
enum class Precision   { Low, Normal, High };  // "low" fp16 | "normal" fp16 + selective fp32 | "high" fp32
enum class CacheMode   { Off, Tune, Full };  // Full = also cache prepacked weights (default)
enum class TensorFormat : uint8_t { NCHW, NHWC, NC4HW4, Unknown };
```

### Conv kernel knobs — `Config::setHint(Hint, Mode)`

Every conv kernel-selection knob goes through one MNN-style hint API: `setHint(Hint, Mode)`. The
defaults are the production kernels; normal use needs none of these. There are no environment variables.

```cpp
enum class Hint {
  Winograd        = 0,  // 3x3 Winograd selection      (Auto / On / Off)
  Tuning          = 1,  // autotune effort             (NoTune / Fast / Thorough)
  WinogradVariant = 2,  // Winograd matmul impl         (TiledGemm / Fused / FusedSplit / FullyFused / SubgroupGemm)
  WinogradUnit    = 3,  // Winograd output tile         (F23 / F43)
  DirectConv3x3   = 4,  // direct 3x3 kernel            (DirectAuto / RegisterTiled / LdsHalo)
};
// One Mode enum holds every value; the Hint picks the knob, the Mode the value.
enum class Mode {
  Auto = 0, On = 1, Off = 2,                                                  // Hint::Winograd
  NoTune = 0, Fast = 1, Thorough = 2,                                         // Hint::Tuning
  TiledGemm = 0, Fused = 1, FusedSplit = 2, FullyFused = 3, SubgroupGemm = 4, // Hint::WinogradVariant
  F23 = 0, F43 = 4,                                                           // Hint::WinogradUnit
  DirectAuto = 0, RegisterTiled = 1, LdsHalo = 2,                             // Hint::DirectConv3x3
};
cfg.setHint(Hint::Tuning, Mode::Thorough);   // maximum autotuning
cfg.setHint(Hint::WinogradUnit, Mode::F43);  // force F(4,3) Winograd
int v = cfg.hint(Hint::WinogradUnit);        // read back (0 if unset)
```

In JSON, the common knobs have named keys (`"winograd": "off"`, `"tuning": "thorough"`); the raw form
is an array indexed by the `Hint` value, `"hints": [2, 1, 0, 0, 0]` (Winograd, Tuning, WinogradVariant,
WinogradUnit, DirectConv3x3).

`NC4HW4` is the internal Vulkan packed layout (channels in vec4 blocks); the engine I/O is `NCHW`.

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
  "maxSubmitNodes": 500,
  "cacheFile": "enc.cache",
  "cacheDir": "/data/local/tmp/vxrt/cache",
  "cacheMode": "full",
  "freeWeightsAfterUpload": true,
  "noFlatOps": false,
  "timing": false,
  "profile": false,
  "verbosity": 1,
  "layerDump": false,
  "layerDumpDir": "/data/local/tmp/vxrt/dump",
  "debugSegments": false,
  "disableVkOps": "",
  "dumpTensors": "",
  "fp32Tensors": "",
  "winograd": "auto",
  "tuning": "fast",
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
cfg3.cacheMode = CacheMode::Full;
cfg3.setHint(Hint::Tuning, Mode::Thorough);

// Apply the log level implied by verbosity:
cfg.applyLogLevel();

// Hand the config to the runtime:
auto session = Runtime::load("assets/mobilenetv2.onnx", cfg);
```

`Runtime::load(onnxPath, cfg)` is a thin façade over
`Session::createFromOnnx(path, cfg)` and returns a `std::unique_ptr<Session>`.

---

## How the `classify` example exposes config flags

[`examples/classify.cpp`](../examples/classify.cpp) (`vknn_classify`) layers
file config and CLI overrides. It loads `--config` first (if given), then
lets individual flags override specific fields:

```cpp
Config cfg;
if (!cfgpath.empty()) cfg = Config::fromJsonFile(cfgpath);   // base from JSON
cfg.backend   = backendFromStr(backend);                      // --backend overrides
cfg.precision = (precision == "fp32") ? Precision::High      // --precision overrides
                                       : Precision::Low;
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
| `--precision P` | `precision` (`fp32`/`fp16`) | `fp16` |
| `--cache DIR` | `cacheDir` | struct default |
| `--profile` | `profile = true` | off |
| `--layer-dump DIR` | `layerDump = true`, `layerDumpDir = DIR` | off |

Flags that are not config fields (model/input handling and benchmarking):
`--model PATH`, `--input PATH`, `--shape N,C,H,W`, `--golden PATH`,
`--show-graph`, `--bench N`.

Precedence: `--config` loads first, then the explicit flags overwrite
whatever the JSON set. `--backend` and `--precision` always assign
`cfg.backend`/`cfg.precision` (they carry CLI defaults), so a `backend` or
`precision` value in the JSON file is overridden by the flag defaults unless
the matching flag is passed. This applies when mixing `--config` with these two
flags.

Example invocations:

```bash
# Pure flags, no file:
vknn_classify --backend vulkan --precision fp16 --profile

# File config plus a cache override:
vknn_classify --config config.json --cache /data/local/tmp/vxrt/cache

# CPU reference run with a golden comparison:
vknn_classify --backend cpu --precision fp32 --golden assets/golden.bin
```
