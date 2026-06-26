# Configuration (`vknn::Config`)

`vknn::Config` is the single struct that controls backend selection, precision,
caching, zero-copy, profiling, and autotuning for a `vknn::Session`. It is defined
in [`include/vknn/config.h`](../include/vknn/config.h) and parsed/serialized in
[`src/core/config.cpp`](../src/core/config.cpp).

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
| `precision` | string | `"fp32"`, `"fp16"`, `"auto"` (also `"FP16"`, `"low"` → fp16) | `"fp16"` | Compute precision for the Vulkan backend. `fp16` = fp16 storage + fp32 accumulation. `fp32` = full precision. `auto` lets the backend choose. |
| `power` | string | `"normal"`, `"high"`, `"low"` | `"normal"` | Power/performance hint passed to the backend scheduler. |
| `cpuThreads` | int | ≥ 1 | `4` | Worker thread count for the CPU backend (NEON/scalar kernels). |
| `inputLayout` | string | `"NCHW"`, `"NHWC"` | `"NCHW"` | Layout of the input tensors you supply. The engine converts to its internal layout as needed. Any value other than `"NHWC"` is treated as `NCHW`. |
| `outputLayout` | string | `"NCHW"`, `"NHWC"` | `"NCHW"` | Layout you want output tensors returned in. Same parsing rule as `inputLayout`. |
| `enableZeroCopy` | bool | `true` / `false` | `false` | Enable DMA-BUF zero-copy I/O (imports a dma-buf fd into Vulkan via `VkImportMemoryFdInfoKHR`). Requires the platform DMA-BUF heap support. |
| `cacheDir` | string | filesystem path | `"/data/local/tmp/vxrt/cache"` | Directory holding the pipeline cache, prepacked-weights cache, and autotune cache. |
| `cachePipeline` | bool | `true` / `false` | `true` | Persist/reuse the `VkPipelineCache` on disk under `cacheDir`. |
| `cacheWeights` | bool | `true` / `false` | `true` | Persist/reuse the content-keyed prepacked-weights cache under `cacheDir`. |
| `cacheTuning` | bool | `true` / `false` | `true` | Persist/reuse the autotune cache (conv workgroup sizes) under `cacheDir`. |
| `profile` | bool | `true` / `false` | `false` | Enable the per-op profiler (GPU timestamp queries + CPU timing); the table is available via `session.profiler()`. |
| `verbosity` | int | `0`, `1`, `≥2` | `1` | Log level. `0` → Warn, `1` → Info, `≥2` → Debug. Applied by `Config::applyLogLevel()`. |
| `layerDump` | bool | `true` / `false` | `false` | Dump every layer's output tensor to disk for debugging. |
| `layerDumpDir` | string | filesystem path | `"/data/local/tmp/vxrt/dump"` | Destination directory for layer dumps (used only when `layerDump` is `true`). |
| `tuning` | string | `"off"`, `"fast"`, `"thorough"` | `"fast"` | Autotuning level for conv workgroup-size search. `off` uses defaults, `fast` does a quick search, `thorough` searches more candidates. |

### Enum reference

The string tokens map onto these enums (from `config.h` / `tensor_format.h`):

```cpp
enum class BackendKind { kVulkan, kCpu };
enum class Precision   { kFp32, kFp16, kAuto };
enum class PowerHint   { kNormal, kHigh, kLow };
enum class TuningLevel { kOff, kFast, kThorough };
enum class TensorFormat : uint8_t { kNCHW, kNHWC, kNC4HW4, kUnknown };
```

`NC4HW4` is the internal Vulkan packed layout (channels in vec4 blocks) and is
not a valid `inputLayout`/`outputLayout` value — only `NCHW` and `NHWC` are
accepted as I/O layouts.

---

## JSON example

A complete config file (`config.json`). Every key here is optional; the example
lists all of them, with non-default values where useful:

```json
{
  "backend": "VULKAN",
  "fallback": ["CPU"],
  "allowCpuFallback": true,
  "precision": "fp16",
  "power": "high",
  "cpuThreads": 4,
  "inputLayout": "NCHW",
  "outputLayout": "NCHW",
  "enableZeroCopy": true,
  "cacheDir": "/data/local/tmp/vxrt/cache",
  "cachePipeline": true,
  "cacheWeights": true,
  "cacheTuning": true,
  "profile": false,
  "verbosity": 1,
  "layerDump": false,
  "layerDumpDir": "/data/local/tmp/vxrt/dump",
  "tuning": "fast"
}
```

This is the exact shape `Config::toJson()` produces, so it round-trips:
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
cfg3.backend   = BackendKind::kVulkan;
cfg3.precision = Precision::kFp16;
cfg3.tuning    = TuningLevel::kThorough;

// Apply the log level implied by verbosity:
cfg.applyLogLevel();

// Hand the config to the runtime:
auto session = Runtime::load("assets/mobilenetv2.onnx", cfg);
```

`Runtime::load(onnxPath, cfg)` is a thin façade over
`Session::createFromOnnx(path, cfg)` and returns a `std::unique_ptr<Session>`.

---

## How the `classify` example exposes config flags

[`examples/classify.cpp`](../examples/classify.cpp) (`vknn_classify`) shows how
file config and CLI overrides layer. It loads `--config` first (if given), then
lets individual flags override specific fields:

```cpp
Config cfg;
if (!cfgpath.empty()) cfg = Config::fromJsonFile(cfgpath);   // base from JSON
cfg.backend   = backendFromStr(backend);                      // --backend overrides
cfg.precision = (precision == "fp32") ? Precision::kFp32      // --precision overrides
                                       : Precision::kFp16;
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

Watch the precedence: `--config` loads first, then the explicit flags overwrite
whatever the JSON set. `--backend` and `--precision` always assign
`cfg.backend`/`cfg.precision` (they carry CLI defaults), so a `backend` or
`precision` value in the JSON file gets overridden by the flag defaults unless
you pass the matching flag. Watch for this when mixing `--config` with these two
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
