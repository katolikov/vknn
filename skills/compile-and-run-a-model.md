# How to compile and run a model

Goal: take an ONNX model, compile it to an optimized `.vxm`, and run it on the device with the right
`Config`. For the public API see [../docs/config.md](../docs/config.md); for op support see
[../docs/op-coverage.md](../docs/op-coverage.md).

## 1. (Optional) Compile ONNX -> .vxm

`vknn_compile` runs the ONNX import and graph passes once and writes a backend-agnostic `.vxm` (weights
optionally fp16). Loading a `.vxm` skips ONNX parsing and the passes, which pays off on large models.

```sh
./build.sh --convert                                   # builds vknn_compile only
./build-host/vknn_compile model.onnx model.vxm --fp16
```

Convert-time flags are **separate** from the runtime `Config`:

| Flag | Effect |
|---|---|
| `--fp16` | store weights as fp16 (≈half the file size; the GPU path is fp16 anyway) |
| `--no-fuse-pointwise` | disable the pointwise-chain fusion (on at `-O1`+; folds swish `x * sigmoid(x)` diamonds into one SiLU step, plus general pointwise chains) |
| `--fuse-se` | fuse the squeeze-excite tail (experimental; off by default) |
| `--fuse-dwpw` | fuse depthwise + pointwise (experimental; off by default) |
| `--dump-big` | log tensors larger than 50M elements after shape inference (diagnostic) |
| `-O0`..`-O3` / `--opt N` | optimization level (default `-O1`); `-O2`+ additionally enables the experimental SE and dw+pw fusions |
| `-Os` | maximum preset: `-O3` + INT4 weight quantization (AWQ + min-MSE + bias correction; implies `--fp16`). Knobs: `--quant-bits 4\|8\|lut4`, `--quant-group N`, `--quant-outliers F`, `--quant-err F`, `--calib in0.bin,in1.bin` (one calibration sample per occurrence, raw files in graph-input order; absent -> synthetic samples) |
| `--batch N` | bind a dynamic batch-named leading axis (default 1) |
| `--dim NAME=VALUE` | bind an ONNX symbolic dimension (repeatable); an unbound non-batch dynamic axis is a hard error, never a silent 1 |
| `--shape NAME=D0xD1x...` | declare one input's full concrete shape (repeatable; overrides `--dim` for that tensor) |
| `--list-dims` | import, print each input's shape and the free dim symbols to bind with `--dim`, then exit without compiling |
| `--bucket "SEG;..."` | one shape bucket per occurrence (segments: `NAME=D0xD1x...` or `dim:NAME=VALUE`, `;`-separated); buckets share one initializer pool in the `.vxm` and the runtime dispatches each run to the matching bucket |
| `--graph "FILE.onnx[;SEG;...]"` | multi-graph form (single positional = the output: `vknn_compile out.vxm --graph ...`); each occurrence compiles one bucket from its own ONNX file into a single multi-bucket `.vxm` (e.g. vision tower + prefill + decoder) |
| `--support-report out.json` | write per-node backend assignment (GPU vs CPU + reason), computed by the same gate the device engine evaluates |

This step is optional: load the `.onnx` directly and `Model::load` / `Runtime::load`
auto-detects `.onnx` vs `.vxm`.

## 2. Run on the device

Push the binary (re-push after every Android rebuild) and run. Two runners are available:

```sh
adb push build-android/vknn_classify build-android/vknn_run_io /data/local/tmp/vxrt/

# image classifier: top-5, golden cosine/top-1, --bench, --profile, --layer-dump
adb shell /data/local/tmp/vxrt/vknn_classify --model model.vxm --input in.bin \
  --golden gold.bin --backend vulkan --precision low --bench 20

# generic runner: any model; named inputs in, each output dumped to a dir
adb shell mkdir -p /data/local/tmp/vxrt/out
adb shell /data/local/tmp/vxrt/vknn_run_io model.vxm /data/local/tmp/vxrt/out in0.bin in1.bin
```

`vknn_run_io` flags: `--backend vulkan|cpu`, `--precision low|normal|high`, `--priority low|normal|high`,
`--tuning none|fast|heavy`, `--winograd auto|on|off`, `--no-cache`, `--cache DIR`, `--keep-weights`,
`--timing`, `--profile`, `--repeat N`, `--bucket N` (run plan bucket N of a multi-bucket model, default 0),
`--cpu-threads N`, `--fp32-tensors NAMES`, `--dump NAMES`, `--disable-vk-ops NAMES`,
`--max-submit-nodes N` / `--max-submit-bindings N` (watchdog/TDR mitigation), `--layer-dump` /
`--layer-dump-dir DIR`, `--debug-segments`, plus the GPU-pass off switches `--no-flat`,
`--no-fold-islands`, `--no-matmul-view-fold`, `--no-rope-fusion`, `--no-fused-attention`,
`--no-kv-concat-fold`.

## 3. Run from C++

The simplest form, `vknn::Model::load("model.onnx")`, picks Vulkan-if-available + `Precision::Low`
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
  cfg.backend   = vknn::BackendKind::Vulkan;     // run on the GPU (CPU is the implicit fallback)
  cfg.precision = vknn::Precision::Low;         // fp16 storage, fp32 accumulation
  cfg.tuning    = vknn::Tuning::Heavy;          // maximum autotuning (cached to the model cache)

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
into `examples/` and add its name to the `examples` list in `CMakeLists.txt`, which handles this.
For finer control, the lower-level `vknn::Session` / `IOTensor` API (`include/vknn/session.h`) takes the
same `Config` and exposes per-tensor residency and DMA-BUF zero-copy.

## 4. Validate

Compare against an **onnxruntime golden** (cosine ≥ 0.999 for fp16, 1.0 for fp32/CPU). Generate
goldens with `scripts/get_golden.py` (CNNs) or `scripts/yonosplat/gen_golden.py` (YoNoSplat). On any
perf-sensitive change, record runtime too (`--bench` / `--timing`) — holding cosine but slowing
the GPU is a regression. Methodology and cooldown protocol: [../docs/benchmark.md](../docs/benchmark.md).
