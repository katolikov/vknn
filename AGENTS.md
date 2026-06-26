# AGENTS.md — orientation for contributors (human or AI)

Read this first. It maps the repo, the conventions that are easy to get wrong, and where to go for
depth. The deeper guides live in [`docs/`](docs/) and [`skills/`](skills/).

## What VKNN is

**VKNN** (*Vulkan Neural Network*, namespace `vknn`) is a small, dependency-free C++17 inference
engine that runs neural networks on Android arm64 GPUs via Vulkan compute. It imports ONNX with a
hand-rolled protobuf parser, lowers to a backend-agnostic NCHW IR, runs graph passes (shape
inference, BatchNorm folding, activation/residual fusion, constant folding, dead-node elimination),
partitions into maximal same-backend **segments**, and runs each segment on a backend: **Vulkan**
(NC4HW4 packed layout, one pre-recorded command buffer per static segment, fp16 storage + fp32
accumulation) or **CPU** (scalar + NEON reference and automatic fallback). It runs image CNNs,
YOLOv8n detection, and a 965M-parameter transformer encoder (YoNoSplat) plus a from-scratch Vulkan
3D Gaussian Splatting rasterizer. See [README.md](README.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Build & test

`./build.sh` is the **only** build entry point.

```sh
./build.sh                 # host build: CPU backend + IR + ONNX import + tools + tests (no Vulkan)
./build.sh --android       # Android arm64-v8a build (Vulkan backend, NDK toolchain)
./build.sh --clear         # wipe the build dir first (clean build); combines with the others
./build.sh --convert       # build only the model compiler (vknn_compile)
./build.sh --docs          # build the static documentation site -> docs/site/index.html
./build-host/vknn_tests    # run the host unit/integration tests
```

Host artifacts land in `build-host/`, Android in `build-android/`. Override the NDK with
`ANDROID_NDK=...` and the API level with `ANDROID_API=...`.

## Repo layout

```
include/vknn/          public headers (model, session, config, backend, op, tensor, graph, ...)
src/core/              session, graph, passes glue, config/JSON, profiler, ion (dma-buf), logging
src/import/onnx/       dependency-free ONNX protobuf parser
src/import/passes.*    graph passes (inferShapes, foldBatchNorm, fuseActivations, constFold, ...)
src/backend/cpu/ops/  CPU operators — ONE OP PER FILE
src/backend/vulkan/   Vulkan backend: context/buffers/command/pipeline + ops/ (ONE OP PER FILE)
shaders/               GLSL compute (.comp) + common.glsl / precision.glsl; compiled by glslc, embedded
convert/compile.cpp    vknn_compile — the ONNX -> .vxm model compiler
examples/              tool/example binaries (built as vknn_*)
tests/                 GoogleTest -> vknn_tests
scripts/               build_android shim, run_on_device, bench, get_golden, yonosplat/
tools/                 embed_spirv.py, compare_layers.py
docs/ , skills/        reference docs + focused how-to guides
```

## The operator framework (and the one rule)

> **One operator per file.** Every backend op lives in its own source file under
> `src/backend/cpu/ops/<op>.cpp` and `src/backend/vulkan/ops/<op>.cpp`, with exactly one
> `VKNN_REGISTER_CPU_OP` / `VKNN_REGISTER_VK_OP` per file. Never put two ops in one file; split
> combined files if you find them. Shared helpers (e.g. `flat::Broadcast` in `flat_ops.h`) may be
> shared across the per-op files.

Adding an op touches: the `OpType` enum (`include/vknn/op.h`), the ONNX-name map
(`src/core/op.cpp`), a shape rule in `inferShapes` (`src/import/passes.cpp`), a CPU oracle
(`src/backend/cpu/ops/<op>.cpp`), and optionally a Vulkan op + GLSL shader gated by
`supportsNode`. The CMake globs use `CONFIGURE_DEPENDS`, so a new file is picked up on the next
configure (which `./build.sh` always runs). Full recipe: [skills/add-an-operator.md](skills/add-an-operator.md)
and [docs/ADDING_AN_OPERATOR.md](docs/ADDING_AN_OPERATOR.md). Backends:
[skills/add-a-backend.md](skills/add-a-backend.md) / [docs/ADDING_A_BACKEND.md](docs/ADDING_A_BACKEND.md).

## Run & validate on device

A connected arm64-v8a device over `adb` runs the Vulkan path. The device scratch dir is
`/data/local/tmp/vxrt/`.

```sh
adb push build-android/vknn_classify build-android/vknn_run_io /data/local/tmp/vxrt/
# image classifier: top-5 + golden cosine/top-1
adb shell /data/local/tmp/vxrt/vknn_classify --model M.onnx --input in.bin \
  --golden gold.bin --backend vulkan --precision fp16 --bench 20
# generic runner (any model: named inputs in, outputs dumped to a dir)
adb shell /data/local/tmp/vxrt/vknn_run_io M.vxm /data/local/tmp/vxrt/out in0.bin in1.bin
```

Check correctness with **cosine vs an onnxruntime golden**, and for any perf-sensitive change,
**measure runtime too** — beating MNN is a standing goal, so a change that keeps cosine but slows the
GPU is a regression. See [skills/compile-and-run-a-model.md](skills/compile-and-run-a-model.md) and
[docs/BENCHMARK.md](docs/BENCHMARK.md).

## Conventions

- **Formatting:** clang-format, Google base, 100-column, sorted + regrouped includes
  (`.clang-format`). Run `scripts/format.sh` (or `clang-format -i`) before committing.
- **Naming:** types `PascalCase`; methods stay `camelCase` (PascalCasing them collides with the
  standard library — `size`/`data`/`empty`/`max`).
- **No new third-party runtime deps.** ONNX import is hand-rolled; keep it that way.

## Commits & authorship

Commits are authored by the repo owner (`katolikov`), with plain messages. **Do not** add any AI
attribution — no `Co-Authored-By: Claude ...`, no "Generated with …", no mention of an assistant in
commit messages or PR bodies. Set `git config user.name`/`user.email` to the owner's identity before
committing.

## Gotchas (these have bitten before)

- **zsh does not word-split** an unquoted variable: `for f in $LIST` iterates once over the whole
  string. Use `printf '%s\n' $LIST | while read f; do ...; done` or `for f in ${(f)LIST}` in bulk-edit
  loops.
- **BSD `sed`** (macOS) has no `\b` word boundary. Use `[[:<:]]`/`[[:>:]]`, or a literal pattern.
- **Re-push the `vknn_*` binaries** to the device after **every** Android rebuild — a stale binary on
  the device silently invalidates a "fix".
- **Thermal throttling is real and fast.** Before each benchmark run, `adb shell sleep 12-14` to cool
  the GPU; never A/B two builds back-to-back. Use `VKNN_TIMING=1` for real submit+GPU time — the
  profiler's per-op sum is inflated by forced per-op barriers. `rm -rf` the model's cache dir before
  timing it.
- **Validate non-classifier / non-image models with `vknn_run_io`**, not `vknn_classify` (the latter
  assumes an image-classifier I/O shape).
- **Huge graphs OOM the host CPU path** (all activations stay live for one big segment) — validate
  large models on the device (fp16, GPU buffers), not on the host.
- **A new op `.cpp` not compiling?** Re-run `./build.sh` so CMake reconfigures and re-globs.
