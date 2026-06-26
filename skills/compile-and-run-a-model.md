# How to compile and run a model

Goal: take an ONNX model, compile it to an optimized `.vxm`, and run it on the device with the right
`Config`. For the public API see [../docs/CONFIG.md](../docs/CONFIG.md); for op support see
[../docs/OP_COVERAGE.md](../docs/OP_COVERAGE.md).

## 1. (Optional) Compile ONNX -> .vxm

`vknn_compile` runs the ONNX import + graph passes once and writes a backend-agnostic `.vxm` (weights
optionally fp16). Loading a `.vxm` later skips ONNX parsing and the passes — useful for large models.

```sh
./build.sh --convert                                   # builds vknn_compile only
./build-host/vknn_compile model.onnx model.vxm --fp16
```

Convert-time flags (these are **separate** from the runtime `Config`):

| Flag | Effect |
|---|---|
| `--fp16` | store weights as fp16 (≈half the file size; the GPU path is fp16 anyway) |
| `--no-fuse-swish` | disable folding `x * sigmoid(x)` into the producing Conv |
| `--fuse-se` | fuse the squeeze-excite tail (experimental; off by default) |
| `--fuse-dwpw` | fuse depthwise + pointwise (experimental; off by default) |
| `--dump-big` | log tensors larger than 50M elements after shape inference (diagnostic) |

You can also skip this step and load the `.onnx` directly — `Model::load` / `Runtime::load`
auto-detects `.onnx` vs `.vxm`.

## 2. Run on the device

Push the binary (re-push after every Android rebuild) and run. Two runners:

```sh
adb push build-android/vknn_classify build-android/vknn_run_io /data/local/tmp/vxrt/

# image classifier: top-5, golden cosine/top-1, --bench, --profile, --layer-dump
adb shell /data/local/tmp/vxrt/vknn_classify --model model.vxm --input in.bin \
  --golden gold.bin --backend vulkan --precision fp16 --bench 20

# generic runner: any model; named inputs in, each output dumped to a dir
adb shell mkdir -p /data/local/tmp/vxrt/out
adb shell /data/local/tmp/vxrt/vknn_run_io model.vxm /data/local/tmp/vxrt/out in0.bin in1.bin
```

`vknn_run_io` flags: `--backend vulkan|cpu`, `--precision fp16|fp32`, `--no-weight-cache`,
`--keep-weights`, `--opt-level N`, `--no-flat`, `--timing`.

## 3. Choose a Config (from code)

```cpp
vknn::Config cfg;
cfg.backend   = vknn::BackendKind::kVulkan;   // CPU is the implicit final fallback
cfg.precision = vknn::Precision::kFp16;        // kFp32 for a bit-exact path
cfg.cacheDir  = "/data/local/tmp/vxrt/cache";  // pipeline + weight + tuning caches
auto net = vknn::Model::load("model.vxm", cfg);
auto out = net.run(pixels);                    // pixels = std::vector<float>, NCHW
```

The simplest form, `vknn::Model::load("model.onnx")`, picks Vulkan-if-available + `Precision::kAuto`
(fp16 on GPU) and reads all shapes/names from the model.

## 4. Validate

Always compare against an **onnxruntime golden** (cosine ≥ 0.999 for fp16, 1.0 for fp32/CPU). Generate
goldens with `scripts/get_golden.py` (CNNs) or `scripts/yonosplat/gen_golden.py` (YoNoSplat). For any
perf-sensitive change, also record runtime (`--bench` / `VKNN_TIMING=1`) — keeping cosine but slowing
the GPU is a regression. Methodology + cooldown protocol: [../docs/BENCHMARK.md](../docs/BENCHMARK.md).
