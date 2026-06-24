# vxrt

**vxrt** (Vulkan-eXynos RunTime, namespace `vx`) is a small, dependency-free C++17 inference
runtime for CNNs on Samsung Exynos / Xclipse mobile GPUs. It imports an ONNX model with a
hand-rolled protobuf parser, lowers it to a backend-agnostic NCHW IR, runs graph passes
(shape inference, BatchNorm folding, activation fusion, constant folding, dead-node
elimination), partitions the result into maximal same-backend segments, and executes each
segment on a pluggable backend. The primary backend is Vulkan compute (NC4HW4 packed layout,
one pre-recorded command buffer per static segment, push descriptors, fp16 storage with fp32
accumulation, and true DMA-BUF zero-copy import); a scalar+NEON CPU backend provides the
reference path and automatic fallback. Everything is verified end-to-end on a Galaxy S26
against onnxruntime goldens.

## Verified results

MobileNetV2, fixed input image (top-1 class 258 = Samoyed), Galaxy S26 / Exynos 2600 /
Xclipse 960. Inference timings are wall-clock for one `Session::run`; cosine and max-abs-err
are versus the onnxruntime CPU golden.

| Backend | Latency | FPS | Cosine vs ORT | maxAbsErr |
|---|---|---|---|---|
| CPU (scalar + NEON) | 672 ms | 1.5 | 1.000000 | — |
| Vulkan fp32 | 24.35 ms | 41.0 | 1.000000 | 1.3e-5 |
| Vulkan fp16 | 22.0 ms | 45.4 | 0.999965 | 0.08 |

GPU compute time alone (timestamp queries, Vulkan fp16) is **12.1 ms**: Conv 10.7, Gemm 0.9,
Add 0.26, GlobalAveragePool 0.09, Reshape 0.11. The remaining wall time is host&harr;device
pack/unpack.

**Session creation** (Vulkan fp16), showing the effect of the on-disk caches:

| Session | Time | Notes |
|---|---|---|
| Cold (first run + autotune) | 445 ms | empty cache, `tuning=fast` |
| Cold (no tuning) | ~152 ms | empty cache, `tuning=off` |
| Warm (all caches hit) | 68 ms | pipeline + prepacked-weights + autotune caches |

Warm session creation is up to **6.5x** faster than a cold tuned build.

## Quickstart

**Prerequisites**

- Android NDK **r27** (`27.0.12077973`); set `ANDROID_NDK` if it lives elsewhere.
- `glslc` (shaderc) on `PATH` — compiles the GLSL compute shaders at build time.
- `ninja` and CMake (&ge; 3.21).
- Python 3 with `onnxruntime`, `numpy`, `onnx`, `pillow` — for the golden generator.
- `adb` with a connected arm64-v8a device for the on-device scripts.

```sh
# 1. Generate the onnxruntime golden (and per-layer goldens with --layers).
#    Drop your model at assets/mobilenetv2.onnx first.
python3 scripts/get_golden.py --image assets/cat.jpg --layers

# 2. Build the static lib + examples + tests for Android arm64-v8a.
scripts/build_android.sh                  # honors ANDROID_NDK / ANDROID_API / BUILD_DIR

# 3. Push to device and run vx_classify (backend + precision are positional args).
scripts/run_on_device.sh vulkan fp16      # also: cpu fp32, enn fp16, ...
```

`scripts/bench.sh [N]` runs the full latency sweep (Vulkan fp16/fp32, CPU) plus the
cold-vs-warm session-creation comparison.

## Example binaries

All build under `build-android/` (and on the host where Vulkan is unavailable). Each links the
static lib whole-archive so self-registering backends and operators survive.

| Binary | Source | What it does |
|---|---|---|
| `vx_probe` | `examples/probe.cpp` | Enumerates the device's Vulkan compute caps (driver, fp16/int8, subgroup size, queues, extensions). |
| `vx_classify` | `examples/classify.cpp` | Loads ONNX, runs an input, prints top-5; optional golden cosine/top-1 check, `--bench`, `--profile`, `--layer-dump`. |
| `vx_profile` | `examples/profile.cpp` | Runs with the profiler on: per-op timing table, JSON, and a Chrome trace. |
| `vx_ion_zerocopy` | `examples/ion_zerocopy.cpp` | DMA-BUF zero-copy import into Vulkan — Mode A (`IonBuffer::alloc`) and Mode B (`wrapFd`), both verified against the staged path. |
| `vx_backend_switch` | `examples/backend_switch.cpp` | Selects the backend via `Config` (VULKAN / CPU / ENN) with no other change; shows ENN consulted first then falling back. |
| `vx_op_check` | `examples/op_check.cpp` | Foundation check: GPU elementwise add vs CPU, plus pipeline-cache round-trip. |

Minimal use of the public API:

```cpp
#include "vx/session.h"
using namespace vx;

Config cfg;
cfg.backend   = BackendKind::kVulkan;   // fallback {kCpu} is implicit
cfg.precision = Precision::kFp16;

auto session = Runtime::load("mobilenetv2.onnx", cfg);

std::vector<IOTensor> in(1), out;
in[0].name  = "input";
in[0].shape = {1, 3, 224, 224};
in[0].data.resize(1 * 3 * 224 * 224 * sizeof(float));
// ... fill in[0].f32() with NCHW fp32 ...

session->run(in, out);   // out[0].f32() holds the NCHW result
```

## Repo layout

```
include/vx/            public headers (session, config, backend, op, tensor, ion, graph, profiler, ...)
src/core/              session, graph, passes glue, config/JSON, profiler, ion (dma-buf), logging
src/import/onnx/       dependency-free ONNX protobuf parser
src/import/passes.*    graph passes (inferShapes, foldBatchNorm, fuseActivations, constFold, eliminateDeadNodes)
src/backends/vulkan/   VulkanBackend: context, buffers, command/pipeline, NC4HW4 ops, autotune
src/backends/cpu/      CpuBackend: scalar reference + NEON Add/Gemm; basic/conv/shape ops
src/backends/enn/      EnnBackend: documented ENN-probing stub (declines ops -> fallback)
src/layout/            layout-conversion helpers (NCHW <-> NC4HW4 / NHWC)
shaders/               GLSL compute (.comp) + common.glsl; compiled by glslc, embedded into the lib
examples/              the six example binaries above
tests/                 GoogleTest: test_core.cpp, test_integration.cpp -> vx_tests
scripts/               build_android, run_on_device, bench, gen_docs, get_golden
tools/                 embed_spirv.py (SPIR-V -> vx::embeddedShaders()), compare_layers.py
docs/                  ARCHITECTURE, ADDING_*, CONFIG, LIMITATIONS, DEVICE_REPORT, adr/
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — import &rarr; IR &rarr; passes &rarr; segments &rarr; backends, and the NC4HW4 compute path.
- [docs/ADDING_AN_OPERATOR.md](docs/ADDING_AN_OPERATOR.md) — add a CPU or Vulkan op via the self-registration macros.
- [docs/ADDING_A_BACKEND.md](docs/ADDING_A_BACKEND.md) — implement and register a new `Backend`.
- [docs/CONFIG.md](docs/CONFIG.md) — every `vx::Config` field and its JSON form.
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md) — supported operators, known gaps, ENN stub status.
- [docs/DEVICE_REPORT.md](docs/DEVICE_REPORT.md) — full probed capabilities of the target device.
- [docs/adr/](docs/adr/) — architecture decision records (0001 language/build, 0002 Vulkan loader + embedded SPIR-V, 0003 UMA memory, 0004 NC4HW4 layout, 0005 ION via DMA-BUF, 0006 segment execution + fallback, 0007 ENN stub).

### Extending

Self-registration works because the static lib is linked whole-archive
(`$<LINK_LIBRARY:WHOLE_ARCHIVE,vxrt>`); no edits to core dispatch are needed.

- **CPU op:** subclass `vx::CpuOp` and `VX_REGISTER_CPU_OP(OpType::kFoo, FooCpuOp)`.
- **Vulkan op:** subclass `vx::VulkanOp` (implement `prepare()` / `record()`), add a GLSL `.comp` under `shaders/`, and `VX_REGISTER_VK_OP(OpType::kFoo, FooVulkanOp)`.
- **Backend:** subclass `vx::Backend` and `VX_REGISTER_BACKEND(BackendKind::kFoo, FooBackend)`.

## Tests and API docs

The unit/integration tests build and run **on the host** (no Vulkan / NEON required):

```sh
cmake -S . -B build-host -G Ninja \
  -DVXRT_ENABLE_VULKAN=OFF -DVXRT_ENABLE_NEON=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-host --target vx_tests
./build-host/vx_tests
```

`vx_tests` covers the core IR/passes (`tests/test_core.cpp`) and the import-to-run pipeline on
the CPU backend (`tests/test_integration.cpp`). The same binary also builds for the device.

Generate the Doxygen API reference:

```sh
scripts/gen_docs.sh        # -> docs/api/html/index.html (uses docs/Doxyfile)
```
