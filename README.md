<h1 align="center">VKNN</h1>

<p align="center">
  <b>Vulkan Neural Network</b> — a small, dependency-free C++17 inference engine for neural networks on Android GPUs.
</p>

<p align="center">
  <img alt="C++17" src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white">
  <img alt="Vulkan compute" src="https://img.shields.io/badge/Vulkan-compute-A41E22?logo=vulkan&logoColor=white">
  <img alt="Android arm64-v8a" src="https://img.shields.io/badge/Android-arm64--v8a-3DDC84?logo=android&logoColor=white">
  <img alt="MIT license" src="https://img.shields.io/badge/license-MIT-blue">
  <img alt="no runtime deps" src="https://img.shields.io/badge/deps-none-success">
</p>

<p align="center">
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#compile-and-run-a-model">Compile &amp; run</a> ·
  <a href="#supported-operators">Operators</a> ·
  <a href="#documentation">Docs</a>
</p>

## About

**VKNN** (namespace `vknn`) runs neural networks on Android arm64 GPUs through **Vulkan compute**. It
imports an ONNX model with a hand-rolled protobuf parser, lowers it to an NCHW IR, runs graph passes
(shape inference, BatchNorm folding, activation/residual fusion, constant folding), partitions the
graph into maximal same-backend segments, and executes each on a pluggable backend. The primary
backend is Vulkan (NC4HW4 packed layout, one pre-recorded command buffer per segment, fp16 storage
with fp32 accumulation, caller-owned DMA-BUF I/O); a scalar + NEON **CPU backend** is the reference path and
the automatic fallback for ops the GPU declines. There are no third-party runtime dependencies —
only Vulkan and the C++ standard library. Every result is checked against an onnxruntime golden.

It runs image CNNs (ResNet-50, MobileNetV2/V3, EfficientNet, Inception, DenseNet, ShuffleNet),
detection (YOLOv8n), and a 965M-parameter transformer encoder (the YoNoSplat feed-forward 3D Gaussian
Splatting model) plus a from-scratch Vulkan 3DGS rasterizer — all on the GPU.

<p align="center">
  <img src="docs/images/vknn_gpu_outputs.png" alt="VKNN classifying a real photo on the Vulkan GPU" width="780">
</p>
<p align="center"><sub>The benchmark CNNs classifying a real photo on the Vulkan GPU (fp16), with top-5 ImageNet labels.</sub></p>

## Benchmarks

VKNN vs [MNN](https://github.com/alibaba/MNN) (Alibaba's production engine), same model, same device,
fp16, thermal-controlled medians — against both of MNN's GPU backends (Vulkan, and OpenCL with HEAVY
autotuning, its strongest path here):

| Model (fp16) | VKNN | MNN-Vulkan | MNN-OpenCL (HEAVY) | VKNN vs ORT |
|---|---|---|---|---|
| MobileNetV2 | 2.3 ms | 13.8 ms | 3.1 ms | cosine 0.99997 |
| MobileNetV3-Large | 2.8 ms | 17.0 ms | 6.4 ms | cosine 0.99954 |
| SqueezeNet 1.1 | 1.7 ms | 10.9 ms | 2.6 ms | cosine 0.99998 |
| EfficientNet-B0 | 4.3 ms | 19.9 ms | 9.3 ms | cosine 0.99983 |
| ResNet-50 | 10.3 ms | 18.3 ms | 10.3 ms | cosine 1.000000 |
| Inception-v3 | 15.5 ms | 25.6 ms | 19.6 ms | cosine 0.99998 |
| YOLOv8n (640×640) | 20.0 ms | ~73 ms | 24.5 ms | cosine 1.000000 |
| YoNoSplat encoder (965M params) | ~13.5 s | cannot convert | cannot convert | 6 outputs, cosine 0.999+ |

The VKNN figure is the full `run()` wall (it includes the host↔device copies); MNN's is inference-only.
Against MNN's absolute best (min over OpenCL-HEAVY, CPU-4-thread, Vulkan), VKNN is faster on **8 of 9**
models and at **parity on ResNet-50**. Methodology, per-stage timings, and the OpenCL-tuned comparison:
[docs/BENCHMARK.md](docs/BENCHMARK.md).

## Compile and run a model

```sh
./build.sh             # host build: CPU backend + IR + ONNX import + tools + tests (no Vulkan)
./build.sh --android   # full engine incl. the Vulkan backend (NDK r27 arm64-v8a)
```

Run an ONNX (or compiled `.vxm`) model on the device:

```sh
adb push build-android/vknn_classify model.onnx input.bin /data/local/tmp/vknn/
adb shell /data/local/tmp/vknn/vknn_classify --model model.onnx --input input.bin \
    --backend vulkan --precision low --bench 20
```

`--precision` is a quality tier: **`low`** (fp16 storage + fp32 accumulation), **`normal`** (fp16, but a
precision-critical geometry-tail set is kept fp32 — selective fp32, a no-op for models without it), or
**`high`** (full fp32). `vknn_compile` turns an ONNX model into an optimized `.vxm` (skips parsing/passes
at load; optional fp16 weights); `vknn_run_io` runs any multi-input/multi-output model. Full flow (compile, run,
YoNoSplat): [skills/compile-and-run-a-model.md](skills/compile-and-run-a-model.md).

**Or from C++.** `vknn::Model` reads each input/output name and shape from the model — you supply only
the data. Vulkan, fp16, maximum autotuning, all fusions; two inputs → two outputs:

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
  cfg.setHint(vknn::Hint::Tuning, vknn::Mode::Thorough); // maximum autotuning (cached in <model>.cache)

  vknn::Model net = vknn::Model::load("model.vxm", cfg);  // auto-detects .vxm vs .onnx
  if (!net) { fprintf(stderr, "failed to load model\n"); return 1; }

  // Names + shapes come from the model; you supply only the data.
  auto in = net.inputs();
  vknn::Tensor a(readBin("in0.bin"), in[0].shape, in[0].name);
  vknn::Tensor b(readBin("in1.bin"), in[1].shape, in[1].name);

  std::vector<vknn::Tensor> outs = net.run({a, b});  // two inputs -> two outputs

  for (const vknn::Tensor& o : outs)
    printf("output '%s'  %s  max=%.4f\n", o.name().c_str(), o.shapeString().c_str(), o.max());

  // ...or fetch a specific output by name:
  if (const vknn::Tensor* y = vknn::findTensor(outs, net.outputs()[0].name))
    printf("argmax of '%s' = %lld\n", net.outputs()[0].name.c_str(), (long long)y->argmax());
  return 0;
}
```

Link the static lib **whole-archive** so the self-registering operators/backends survive (drop the
`.cpp` in `examples/` and add it to the `examples` list in `CMakeLists.txt`, which already does this).
Everything is configured through `vknn::Config` — the engine reads no environment variables
([docs/CONFIG.md](docs/CONFIG.md)).

## Supported operators

A broad ONNX op set: convolution/pooling, the elementwise unary/binary families, MatMul (batched N-D),
Gemm, LayerNorm, Softmax, Einsum, RoPE, Gather/Scatter, Resize, Pad, GridSample, and the
shape/data-movement ops — enough for CNNs, detection, and transformer/attention models. Per-op
GPU/CPU coverage: [docs/OP_COVERAGE.md](docs/OP_COVERAGE.md). Adding an op is one new file via the
self-registration macros: [docs/ADDING_AN_OPERATOR.md](docs/ADDING_AN_OPERATOR.md).

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — import → IR → passes → segments → backends, and the NC4HW4 compute path.
- [docs/CONFIG.md](docs/CONFIG.md) — every `vknn::Config` field, the `setHint` API, and the JSON form.
- [docs/OP_COVERAGE.md](docs/OP_COVERAGE.md) — the operator set and its backend coverage.
- [docs/BENCHMARK.md](docs/BENCHMARK.md) — on-device VKNN vs MNN numbers and methodology.
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md) — known gaps and the single-device caveat.
- [docs/ADDING_AN_OPERATOR.md](docs/ADDING_AN_OPERATOR.md) · [docs/ADDING_A_BACKEND.md](docs/ADDING_A_BACKEND.md) — extend the engine (one new file, no core edits).
- [docs/adr/](docs/adr/) — architecture decision records.
- [AGENTS.md](AGENTS.md) + [skills/](skills/) — orientation and focused how-to guides.

Build the static documentation site with `./build.sh --docs` (→ `docs/site/index.html`).

## License

MIT — see [LICENSE](LICENSE).
