# How to compile and run a model

Goal: take an ONNX model, compile it to an optimized `.vxm`, and run it on the device with the right
`Config`. For the public API see [../docs/CONFIG.md](../docs/CONFIG.md); for op support see
[../docs/OP_COVERAGE.md](../docs/OP_COVERAGE.md).

## 1. (Optional) Compile ONNX -> .vxm

`vknn_compile` runs the ONNX import + graph passes once and writes a backend-agnostic `.vxm` (weights
optionally fp16). Loading a `.vxm` later skips ONNX parsing and the passes, which pays off on large models.

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

Or skip this step and load the `.onnx` directly — `Model::load` / `Runtime::load`
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

## 3. Run from C++

The simplest form, `vknn::Model::load("model.onnx")`, picks Vulkan-if-available + `Precision::kAuto`
(fp16 on GPU) and reads all shapes/names from the model:

```cpp
vknn::Model net = vknn::Model::load("model.vxm");
vknn::Tensor out = net.run(pixels);   // pixels = std::vector<float>, NCHW
int cls = out.argmax();
```

Full control — Vulkan, fp16, **maximum autotuning**, all fusions, a **two-input → two-output** model:

```cpp
#include "vknn/model.h"
#include <cstdio>
#include <fstream>
#include <vector>

// Read a raw fp32 .bin into a float vector.
static std::vector<float> readBin(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  size_t n = f ? (size_t)f.tellg() / sizeof(float) : 0;
  std::vector<float> v(n);
  if (f) { f.seekg(0); f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float)); }
  return v;
}

int main() {
  vknn::Config cfg;
  cfg.backend   = vknn::BackendKind::kVulkan;    // run on the GPU (CPU is the implicit fallback)
  cfg.precision = vknn::Precision::kFp16;        // fp16 storage, fp32 accumulation
  cfg.tuning    = vknn::TuningLevel::kThorough;  // maximum autotuning (cached to cfg.cacheDir)
  cfg.optLevel  = 3;                             // all graph fusions (the default)

  vknn::Model net = vknn::Model::load("model.vxm", cfg);  // auto-detects .vxm vs .onnx
  if (!net) { fprintf(stderr, "failed to load model\n"); return 1; }

  auto in = net.inputs();  // two inputs; names + shapes come from the model
  vknn::Tensor a(readBin("in0.bin"), in[0].shape, in[0].name);
  vknn::Tensor b(readBin("in1.bin"), in[1].shape, in[1].name);

  std::vector<vknn::Tensor> outs = net.run({a, b});  // two inputs -> two outputs

  for (const vknn::Tensor& o : outs)
    printf("output '%s'  %s  max=%.4f\n", o.name().c_str(), o.shapeString().c_str(), o.max());

  if (const vknn::Tensor* y = vknn::findTensor(outs, net.outputs()[0].name))
    printf("argmax of '%s' = %lld\n", net.outputs()[0].name.c_str(), (long long)y->argmax());
  return 0;
}
```

Link the static lib **whole-archive** so the self-registering operators survive: drop the `.cpp`
into `examples/` and add its name to the `examples` list in `CMakeLists.txt`, which already handles this.
For finer control, the lower-level `vknn::Session` / `IOTensor` API (`include/vknn/session.h`) takes the
same `Config` and exposes per-tensor residency and DMA-BUF zero-copy.

## 4. Validate

Always compare against an **onnxruntime golden** (cosine ≥ 0.999 for fp16, 1.0 for fp32/CPU). Generate
goldens with `scripts/get_golden.py` (CNNs) or `scripts/yonosplat/gen_golden.py` (YoNoSplat). On any
perf-sensitive change, record runtime too (`--bench` / `VKNN_TIMING=1`) — holding cosine but slowing
the GPU is still a regression. Methodology and cooldown protocol: [../docs/BENCHMARK.md](../docs/BENCHMARK.md).
