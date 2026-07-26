<p align="center">
  <img src="docs/images/vknn_logo.svg" width="440" alt="VKNN — Vulkan Neural Network">
</p>

<p align="center">
  <b>A small, dependency-free C++17 inference engine that runs ONNX models — CNNs, YOLO, LLMs,
  vision-language models, 3D Gaussian Splatting — entirely on the Android GPU.</b>
</p>

<p align="center">
  <img alt="C++17" src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white">
  <img alt="Vulkan compute" src="https://img.shields.io/badge/Vulkan-compute-A41E22?logo=vulkan&logoColor=white">
  <img alt="Android arm64-v8a" src="https://img.shields.io/badge/Android-arm64--v8a-3DDC84?logo=android&logoColor=white">
  <img alt="MIT license" src="https://img.shields.io/badge/license-MIT-blue">
  <img alt="no external runtime deps" src="https://img.shields.io/badge/external%20deps-none-success">
</p>

<p align="center">
  <a href="#why-vknn">Why VKNN</a> ·
  <a href="#what-it-runs">What it runs</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#feature-matrix">Features</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#documentation">Docs</a>
</p>

## Why VKNN

- **The GPU is the engine.** Every executable operator has a Vulkan compute kernel; the shipped
  models run with **0 CPU fallbacks**. The scalar + NEON CPU backend exists as the reference oracle
  and a fallback that announces itself.
- **Compile once, run many times.** `vknn_compile` bakes an ONNX model into an optimized `.vxm`
  plan; loading one skips ONNX parsing and graph passes entirely, and the plan is memory-mapped, not
  copied.
- **Nothing to install.** C++17, the Vulkan loader, and a vendored statically linked MessagePack
  library (warm-start-cache serialization) — no protobuf, no Python, no framework at run time. The
  ONNX importer is a hand-rolled protobuf wire parser.
- **Quantization built in.** `vknn_compile -Os` produces int4 (or int8 / lut4) weights with AWQ
  outlier columns kept fp16 and a per-layer error guard, executed by native quantized GPU MatMul
  kernels.
- **Verified, deterministic.** Every path is checked against an onnxruntime golden; autotuning races
  only bit-neutral launch parameters, so `--tuning none/fast/heavy` produce byte-identical output.
- **Fast.** Faster than [MNN](https://github.com/alibaba/MNN)'s best backend on 8 of 9 benchmark
  models, at parity on ResNet-50 ([benchmarks](#benchmarks)).

## What it runs

All of these run on the Android GPU today, from this repo:

- **Image CNNs** — ResNet-50, MobileNetV2/V3, EfficientNet, Inception, DenseNet, ShuffleNet.
- **Detection** — YOLOv8n.
- **LLMs** — Qwen2.5-Coder-0.5B; int4 Llama-3.2-1B and Llama-3.1-8B ([chat demo](#chat-with-an-llm)).
- **A vision-language model** — SmolVLM2-2.2B, vision tower + decoder prefill/decode shipped as
  **one multi-graph `.vxm`** ([VLM demo](#show-it-a-picture)).
- **3D Gaussian Splatting** — the 965M-parameter YoNoSplat feed-forward encoder plus a from-scratch
  Vulkan 3DGS rasterizer.

<p align="center">
  <img src="docs/images/vknn_gpu_outputs.png" alt="VKNN classifying a real photo on the Vulkan GPU" width="780">
</p>
<p align="center"><sub>The benchmark CNNs classifying a real photo on the Vulkan GPU (fp16), with top-5 ImageNet labels.</sub></p>

## How it works, in one minute

1. **Import** — a hand-rolled protobuf parser reads the ONNX file into a backend-agnostic NCHW IR.
2. **Optimize** — graph passes run: shape inference, BatchNorm folding, activation/residual fusion,
   pointwise-chain fusion into producer epilogues, quantized-node dequantization, constant folding.
3. **Plan** — the graph is partitioned into maximal same-backend segments; each op is assigned to
   the first backend that supports it (Vulkan first, CPU as the loud fallback).
4. **Execute** — the Vulkan backend packs tensors into an NC4HW4 layout (flat row-major for
   transformers), stores fp16 / accumulates fp32, pre-records one command buffer per segment, and
   replays it every run. I/O can bind caller-owned DMA-BUF fds with zero copies.

Steps 1–2 happen once, offline, in `vknn_compile`. The documentation site
(`./build.sh --docs` → `docs/site/index.html`) walks this pipeline interactively: **How VKNN works**
is a clickable tour of the compile → `.vxm` → runtime flow, and **Neural brain** is a drill-down
explorer of the engine's class graph.

## Quickstart

Build the engine and tools:

```sh
./build.sh             # host build: CPU backend + IR + ONNX import + tools + tests (no Vulkan)
./build.sh --android   # full engine incl. the Vulkan backend (NDK r27 arm64-v8a)
./build.sh --convert   # only the model compiler (vknn_compile), for the chosen target
./build.sh --test      # build + run the host unit tests (fast; skips examples/tools)
./build.sh --leakcheck # run the tests under memory-leak detection (Linux: ASan+LeakSanitizer+UBSan; macOS: the `leaks` tool)
./build.sh --docs      # the static documentation site -> docs/site/index.html
```

**Compile once, run many times.** `vknn_compile` turns an ONNX model into an optimized `.vxm` that
skips ONNX parsing and graph passes at load:

```sh
# model.onnx -> model.vxm.  --fp16 halves the file + host upload.
vknn_compile model.onnx model.vxm --fp16
```

No shape flag is needed for a static model, and a dynamic **batch** (leading) axis resolves to 1
automatically — every benchmark CNN above compiles exactly like this. Only a dynamic **non-batch**
axis (height/width/sequence) needs resolving, and the compiler refuses to guess it (substituting a
default into a spatial axis would silently compile a 1×1 plan): it stops with the unbound names so
you can pass either form —

```sh
vknn_compile model.onnx model.vxm --fp16 --shape input=1x3x512x512   # per-tensor declared shape
vknn_compile model.onnx model.vxm --fp16 --dim sequence_length=128   # by symbolic dim name, shared across inputs
```

For a **multi-input** model, repeat `--shape` once per input that needs it (static inputs and
dynamic-batch-only inputs still need nothing):

```sh
vknn_compile two_view.onnx two_view.vxm --fp16 \
    --shape image=1x2x3x448x448 --shape intrinsics=1x2x3x3
```

One `--dim` binding is often the shorter form there — a symbol like `sequence_length` is resolved
in every input that references it at once. At run time, tools take the inputs in the model's
declared order (`vknn_run_io model.vxm out in0.bin in1.bin ...`), and the C++ API matches them by
tensor name.

The default (`-O1`) is the fastest configuration: fused pointwise chains run fp32-chained, which is
at least as accurate as the unfused graph on every chain but not bit-identical to it. When the
outputs must be **byte-identical to a compile with no fusion at all** (`-O0` — regression baselines,
cross-build byte gates), add `--strict-fuse`, which keeps every fused step rounded exactly like the
standalone kernels it replaces at a small speed cost:

```sh
# fastest configuration whose outputs are bit-exact vs the unoptimized (-O0) compile
vknn_compile model.onnx model.vxm --fp16 --strict-fuse
```

Everything the runtime adds on top — zero-copy Concat/Split/Slice views, movement-chain folding,
the layout vote, kernel autotuning — is automatic and byte-identical by construction at every
tuning level, so neither command needs further flags for that.

Push the model and run it on the device:

```sh
adb push build-android/vknn_classify model.vxm input.bin /data/local/tmp/vknn/
adb shell /data/local/tmp/vknn/vknn_classify --model model.vxm --input input.bin \
    --backend vulkan --precision low --bench 20
```

`--precision` is a quality tier: **`low`** (fp16 storage + fp32 accumulation), **`normal`** (fp16 with
a precision-critical geometry tail kept fp32), or **`high`** (full fp32).

**Or from C++.** The model reports its own input/output names and shapes — you supply only the data.
This is [`examples/basics/readme_quickstart.cpp`](examples/basics/readme_quickstart.cpp), compiled by the build:

```cpp
#include "vknn/model.h"

vknn::Config cfg;
cfg.backend   = vknn::BackendKind::Vulkan;  // run on the GPU (CPU is the implicit fallback)
cfg.precision = vknn::Precision::Low;       // fp16 storage, fp32 accumulation

vknn::Model net = vknn::Model::load("model.vxm", cfg);  // auto-detects .vxm vs .onnx
if (!net) {
  return 1;
}

// Names + shapes come from the model; you supply only the data.
auto in = net.inputs();
vknn::Tensor input(pixels, in[0].shape, in[0].name);    // pixels: std::vector<float>, NCHW

std::vector<vknn::Tensor> outputs = net.run({input});   // one input in, every output back

const vknn::Tensor& y = outputs.front();
int cls = (int)y.argmax();
```

Link the static lib **whole-archive** so the self-registering operators/backends survive; dropping a
`.cpp` into `examples/` and adding it to the `_vknn_examples` list in `CMakeLists.txt` already does
this. Everything is configured through `vknn::Config` — the engine reads **no environment variables**
([docs/config.md](docs/config.md)).

## Chat with an LLM

VKNN runs **Qwen2.5-Coder-0.5B** (a `qwen2` autoregressive decoder) end to end with **every compute
op on the GPU — zero CPU fallbacks**, generating text that matches the HuggingFace greedy reference
token-for-token. A small terminal chat app drives it: [`examples/llm/chat.cpp`](examples/llm/chat.cpp) owns
the on-device GPU decode loop (fixed-context KV cache, token streaming) and
[`examples/llm/chat_host.py`](examples/llm/chat_host.py) is the one host dependency (HuggingFace tokenizer +
REPL). Asking it a question:

```text
user>  Write a Python function to check if a number is prime.
model> def is_prime(n):
           if n <= 1:
               return False
           if n <= 3:
               return True
           if n % 2 == 0 or n % 3 == 0:
               return False
           i = 5
           while i * i <= n:
               if n % i == 0 or n % (i + 2) == 0:
                   return False
               i += 6
           return True
```

What makes the decode loop fast — all applied at load, never rewriting a compiled `.vxm`, each gated
by a `Config` hint with a `--no-*` flag:

- **RoPE fusion** — the rotate-half chains collapse into one `Rope` dispatch per q/k site.
- **Fused attention** — the single-query attention core (MatMul → scale/mask → Softmax → MatMul)
  fuses into one `FusedAttention` kernel per layer, reading the GQA KV cache through per-axis
  operand-view strides; `repeat_kv` is never materialized.
- **On-GPU argmax** — greedy decode registers the logits output for an engine-side argmax
  (`Session::setOutputArgMax` / `readOutputArgMax`), so the next-token id comes back as 8 bytes from
  a GPU reduction instead of a download of the 151936-wide logits row (the token stream is
  unchanged — first-occurrence argmax).

Together they cut the engine host loop from ~9 ms to ~0.5 ms per token; the int4 Qwen instruct model
decodes a token in a ~19.7 ms GPU span at a 1024-token context. Its
[`qwen-vknn`](https://huggingface.co/katolikov/qwen-vknn) repo ships a 517 MB int4 build — with a
256-token whole-window prefill bucket — next to the fp16 export.

Full walkthrough (export → compile → run + more examples): [docs/running-an-llm.md](docs/running-an-llm.md)
and the [Running an LLM on VKNN](https://github.com/katolikov/vknn/wiki/Running-an-LLM-on-VKNN) wiki page.

## Show it a picture

**SmolVLM2-2.2B** vision-language chat runs full-GPU from **one multi-graph `.vxm`**: the SigLIP
vision tower, the token embedding, and the text decoder's prefill + decode plans compile into a
single file over a content-deduped weight pool
([hf.co/katolikov/SmolVLM2-2.2B-vknn](https://huggingface.co/katolikov/SmolVLM2-2.2B-vknn)), and the session
dispatches each `run()` to the right graph by its bound input names + shapes.
[`examples/llm/vlm.cpp`](examples/llm/vlm.cpp) drives the device loop (image encode → on-device
embedding splice → prefill → streamed decode) with
[`examples/llm/vlm_host.py`](examples/llm/vlm_host.py) as the host front-end. On a current flagship
phone GPU it answers questions about a photo at 6–7.5 tokens/s with a 0.85 s prefill, matching the
fp32 onnxruntime reference token-for-token. The repo publishes a 1.35 GB int4 build of the model
alongside the 4.5 GB fp16 one. Walkthrough: [docs/running-a-vlm.md](docs/running-a-vlm.md).

The same models power [`app-demo/`](app-demo/) — an Android app (Kotlin/Compose over JNI) with four
tabs: **Chat**, **VLM** camera coach, **3D Splat** capture, and a **Library** that downloads each
`.vxm` from HuggingFace. The Chat and VLM tabs each carry a per-tab model picker (Chat: Qwen fp16 / int4 and int4 Llama 3.2 1B / 3.1 8B; VLM: SmolVLM2 fp16 / int4).

## Feature matrix

| Capability | What VKNN does |
|---|---|
| **Backends** | Vulkan compute GPU (primary) + scalar/NEON CPU (reference & automatic fallback), selected per segment. |
| **Full-GPU op coverage** | Every *executable* operator has a Vulkan kernel; a whole benchmark model runs on the GPU with **0 CPU fallbacks**. Only data-dependent control flow (`Loop` / `If` / `NonMaxSuppression`) and const-folded import ops stay off the GPU. See [docs/op-coverage.md](docs/op-coverage.md). |
| **Precision** | fp16 storage + fp32 accumulation (`low`), selective-fp32 geometry tail (`normal`), or full fp32 (`high`). Stores rounded to nearest even; every path checked against an onnxruntime golden. |
| **Dynamic shapes** | Declared shape **plan buckets**: `vknn_compile --shape NAME=D0xD1x...` / `--dim NAME=VALUE` (binds a symbolic axis; `--list-dims` prints a model's free symbols) / `--bucket "..."` bakes one plan per shape set; at runtime `Session::prepareShapes()` compiles more, and `run()` selects a bucket by the bound input shapes. A fixed-shape model is one bucket (a single map lookup on the hot path). |
| **Multi-graph `.vxm`** | `vknn_compile --graph "FILE[;shape/dim segments]"` (repeatable) compiles **several source graphs** — or one graph at several shapes — into one `.vxm` over a content-deduped weight pool; `run()` dispatches to the bucket matching the bound input names + shapes. Buckets stream at load (host peak = one bucket's weights) and share GPU weight copies by content, so a whole VLM (vision tower + embedding + decoder prefill/decode) is one file and one session. See [docs/running-a-vlm.md](docs/running-a-vlm.md). |
| **Quantized models** | QDQ / QLinear / dynamic-quant checkpoints load and run: quantized nodes are **dequantized to float** at import (saturation clamps preserved), so a quantized export runs without a separate float model. `--no-dequantize` opts out. `vknn_compile -Os` goes the other way and **produces** quantized weights — all fusion plus calibration-free weight quantization (`--quant-bits 4|8|lut4`, default int4; AWQ outlier columns kept fp16, per-layer error guard; `--calib` supplies real calibration samples) over native int4 / int8 / lut4 GPU MatMul kernels; the Qwen instruct weights come out ~2.4× smaller, and every bucket of a multi-graph `.vxm` is requantized. |
| **Autotuned kernels** | Load-time GEMM/conv-kernel autotuning (`--tuning none`/`fast`/`heavy`); the chosen kernels + prepacked/Winograd weights are cached per model, so a warm load skips shader compilation, prepacking, and tuning. Tuning affects **speed only**: the timing races cover just bit-neutral launch parameters (workgroup size, tile width, registers per thread), and every kernel choice that changes fp16 rounding — Winograd vs direct, F(2,3) vs F(4,3), implicit-GEMM vs direct — is a deterministic shape rule that holds at every tuning level, so `none` / `fast` / `heavy` produce byte-identical output. `--tuning none` runs no new sweep (a cached pick, also bit-neutral, is still reused); add `--no-cache` to force a fully cold compile. |
| **Zero-copy I/O** | Caller-owned DMA-BUF fds bind straight to the GPU boundary buffer (no host copy) via `Tensor::fromDmaBuf` / `toDmaBuf`, with a declared layout/dtype the GPU converts on the fly when it differs from device-native. See [`examples/io/dmabuf_fd_io.cpp`](examples/io/dmabuf_fd_io.cpp). |
| **Cooperative-matrix GEMM** | On a device exposing `VK_KHR_cooperative_matrix` (enumerated 16x16x16 subgroup rows, pinnable 32-wide subgroups, Vulkan memory model), eligible fp16 MatMuls route to hardware matrix tiles with fp32 accumulation — after a one-time on-device exact self-check against the CPU oracle, so a driver with a different fragment mapping falls back instead of corrupting. Opt-in fp8 (e4m3) and int8 kinds via `setHint(Hint::CoopmatGemm, ...)`. Devices without the capability run the SSBO kernels unchanged. This path is capability-complete but not yet validated on cooperative-matrix hardware. |
| **Warm-start cache** | A self-validating, multi-variant per-model `.cache` (kernel hash + device + config) auto-heals across driver/model/code changes. |
| **Tools** | `vknn_compile` (ONNX → `.vxm`, with `--support-report <out.json>` for the per-node backend assignment), `vknn_run_io` (any multi-input/multi-output model), plus the example runners below. |

## Benchmarks

VKNN v1.4.3+, whole CNN suite, fp16, `--tuning fast`, 20-iteration medians with a cooldown before
every stage. Every number in this section uses the **default compile/run configuration** — the
first quickstart command, no `--strict-fuse` (that flag exists for byte-comparing a fused compile
against an unfused one; the determinism and accuracy guarantees below hold in the default
configuration). The VKNN figure is the full `run()` wall (it includes the host↔device copies).
"vs v1.4.3" is the cooled interleaved paired A/B delta (GPU span, min-of-5 valid pairs, **three
devices**), measured under exclusive device use with each arm's tune table pinned and any pair
whose arms straddled a DVFS step discarded. Accuracy is against fp32 onnxruntime references, and
every output is **byte-identical to v1.4.3** on all three devices at `none`, `fast` and `heavy`
with 0 CPU fallbacks — the speedups below change scheduling and memory traffic, never a result.

| Model (fp16) | VKNN GPU span | vs v1.4.3 (3 devices) | cosine | PSNR |
|---|---|---|---|---|
| SqueezeNet 1.1 | 1.5 ms | **−9 / −12 / −3%** | 1.000000 | 73.0 dB |
| ShuffleNetV2 x1.0 | 1.6 ms | −3% / noise / noise | 0.999998 | 67.5 dB |
| MnasNet 1.0 | 1.9 ms | **−9 / −5 / −5%** | 0.999989 | 63.6 dB |
| MobileNetV2 | 1.9 ms | **−3 / −4 / −3%** | 0.999989 | 64.4 dB |
| MobileNetV3-Large | 2.5 ms | **−12 / −15 / −8%** | 0.999991 | 63.5 dB |
| EfficientNet-B0 | 4.0 ms | **−32 / −33 / −12%** | 0.999987 | 61.7 dB |
| ResNet-50 | 12.0 ms | **−11 / −12 / −2%** | 1.000000 | 83.0 dB |
| DenseNet-121 | 12.5 ms | **−16 / −5 / −14%** | 0.999997 | 70.8 dB |
| Inception-v3 | 15.3 ms | −1.7% / −1.4% / ±0 | 0.999988 | 62.3 dB |
| YOLOv8n (640×640) | 14.9 ms | **−4 / −6 / −4%** | 1.000000 | 87.4 dB |
| YoNoSplat encoder (965M params, 8 views) | 7.60 s | **−13 / −16 / −16%** | 6 outputs ≥ 0.999993 | 65–81 dB |
| Qwen2.5-0.5B int4 (decode) | 14.6 ms/token | **−16%**, token stream identical | — | — |
| SmolVLM2-2.2B int4 (decode) | 59.0 ms/token | **−5%**, token stream identical | — | — |

Cold load also moves, in both directions: an analytical model now prunes the autotuner's candidate
list before it races, which cuts the 965M-parameter encoder's first load from 134 s to 23 s on the
device where it was measured, while models whose candidates all survive pruning pay a little more
first-load time for the more representative timing. Warm load is unchanged.

One table against [MNN](https://github.com/alibaba/MNN) (Alibaba's production engine), both fp16
on the same device class: the **MNN-Vulkan** column is MNN's Vulkan backend, the **MNN best**
column is its strongest backend per model (OpenCL with HEAVY autotuning, or CPU-4-thread where
that wins). MNN figures are from the recorded head-to-head sweeps
([docs/benchmark.md](docs/benchmark.md)); MNN times inference only with the input set once outside
the loop, so the comparison is generous to MNN. VKNN is v1.4.2 throughout:

| Model (fp16) | VKNN v1.4.2 | MNN-Vulkan | vs Vulkan | MNN best (backend) | vs best |
|---|---|---|---|---|---|
| SqueezeNet 1.1 | 1.7 ms | 10.9 ms | **~6.4×** | 2.59 ms (OpenCL-HEAVY) | **−34%** |
| ShuffleNetV2 x1.0 | 1.8 ms | — | — | — | — |
| MnasNet 1.0 | 2.2 ms | — | — | 3.68 ms (CPU-4t) | **−40%** |
| MobileNetV2 | 2.4 ms | 13.8 ms | **~5.8×** | 3.11 ms (OpenCL-HEAVY) | **−23%** |
| MobileNetV3-Large | 3.1 ms | 17.0 ms | **~5.5×** | 3.78 ms (CPU-4t) | **−18%** |
| EfficientNet-B0 | 6.4 ms | 19.9 ms | **~3.1×** | 9.29 ms (OpenCL-HEAVY) | **−31%** |
| ResNet-50 | 12.8 ms | 18.3 ms | **~1.4×** | 9.00 ms (OpenCL-HEAVY) | MNN ahead on a hot device¹ |
| DenseNet-121 | 15.9 ms | — | — | 15.37 ms (CPU-4t) | ~parity |
| Inception-v3 | 16.8 ms | 25.6 ms | **~1.5×** | 18.38 ms (CPU-4t) | **−9%** |
| YOLOv8n (640×640) | 19.4 ms | ~73 ms | **~3.8×** | 24.71 ms (OpenCL) | **−21%** |

¹ ResNet-50 is the one conv net where MNN's HEAVY-tuned buffer GEMM keeps an edge when the device
is deep-warm (its inference-only loop also rides DVFS clocks that VKNN's per-run wall does not);
from a cool device the sweeps measured parity. A "—" means that cell was not part of a recorded
MNN sweep. YOLOv8n and the YoNoSplat encoder run **100% on the GPU** (no CPU fallback); MNN cannot
convert the encoder at all. Methodology and per-stage timings:
[docs/benchmark.md](docs/benchmark.md).

v1.4.0's speedups come from an autotuned OCB×WTILE register-tile axis, a general split-K conv for
parallelism-starved shapes (`setHint(Hint::SplitKConv, ...)` overrides its shape rule), a
sliding-window 1xK/Kx1 kernel, a raced 16×16 LDS-halo tile, and a small-axis softmax mapping — see
[docs/benchmark.md](docs/benchmark.md) § Conv register tiles.

v1.4.1 adds synchronization2-scoped barriers with write-after-read elision, a
one-dispatch ChannelShuffle operator (ShuffleNetV2 −11/−21% per device), a no-LDS register-tile
Winograd GEMM that extends the Winograd shape rule to large-channel 3×3 convs on big output maps,
and depthwise/output-channel tile candidates in the bit-neutral races — output bytes per model are
unchanged from v1.4.0 at every tuning level; deltas in
[docs/benchmark.md](docs/benchmark.md) § Barrier hygiene.

v1.4.2 makes Concat, Split, and contiguous Slice zero-copy wherever their slices tile
the whole contiguously in the stored layout: each slice becomes a sub-buffer view into the whole's
device memory, so producers write the concatenation in place and the copy dispatches disappear
(ShuffleNetV2 −17% on both devices, SqueezeNet −13/−10%, YOLOv8n −8/−4%, Inception-v3 −4% at
`none`). Automatic and structural — no flags, byte-identical output on every model at every tuning
level; mechanism in
[docs/adr/0018-zero-copy-concat-split-views.md](docs/adr/0018-zero-copy-concat-split-views.md),
numbers in [docs/benchmark.md](docs/benchmark.md) § Zero-copy.

Since v1.4.3 the gains come from how work is scheduled rather than from new arithmetic. The dense
fp16 GEMM loads its operand panels 64 bits at a time and an unaligned activation is virtualized
into a padded physical row so the wide path can take it; a pointwise unit is no longer hosted on a
concat whose parts can alias, which would have cancelled that concat's zero-copy views; and the
autotuner now races the kernel the graph actually dispatches, from a cold cache, timed on the GPU,
after an analytical model has pruned the candidate list. Each is byte-neutral by construction, and
each was gated as such.

The accuracy columns do not depend on the tuning level: kernel choices that change fp16 rounding
are deterministic shape rules (see **Autotuned kernels** above), so `none` / `fast` / `heavy` produce
byte-identical output for a given model and device — verified per model, along with run-to-run
byte-identity, on all three test devices.

## Supported operators

A broad ONNX op set: convolution/pooling, the elementwise unary/binary families, MatMul (batched N-D),
Gemm, LayerNorm, Softmax, Einsum, RoPE, Gather/Scatter, Resize, Pad, GridSample, Range, the
QDQ/QLinear quantization ops (dequantized at import), and the shape/data-movement ops — enough for
CNNs, detection, and transformer/attention models. The load-time decode passes additionally synthesize
a `Rope` and a fused single-query attention op that are created in-engine rather than parsed from ONNX.
Per-op GPU/CPU coverage:
[docs/op-coverage.md](docs/op-coverage.md). Adding an op is one new file via the self-registration
macros: [docs/adding-an-operator.md](docs/adding-an-operator.md).

## Documentation

`./build.sh --docs` builds the full documentation site at `docs/site/index.html` — every page below,
plus two interactive ones: **How VKNN works** (a clickable tour of the compile → `.vxm` → runtime
pipeline) and **Neural brain** (a drill-down explorer of the engine's class graph).

- [docs/architecture.md](docs/architecture.md) — import → IR → passes → segments → backends, and the NC4HW4 compute path.
- [docs/config.md](docs/config.md) — every `vknn::Config` field, the `setHint` API, and the JSON form.
- [docs/running-an-llm.md](docs/running-an-llm.md) · [docs/running-a-vlm.md](docs/running-a-vlm.md) — export, compile, and drive an LLM / VLM on the device.
- [docs/op-coverage.md](docs/op-coverage.md) — the operator set and its backend coverage.
- [docs/benchmark.md](docs/benchmark.md) — on-device VKNN vs MNN numbers and methodology.
- [docs/limitations.md](docs/limitations.md) — known gaps, dynamic-shape buckets, quantization, and the single-device caveat.
- [docs/adding-an-operator.md](docs/adding-an-operator.md) · [docs/adding-a-backend.md](docs/adding-a-backend.md) — extend the engine (one new file, no core edits).
- [docs/adr/](docs/adr/) — architecture decision records.
- [AGENTS.md](AGENTS.md) + [skills/](skills/) — orientation and focused how-to guides.

Runnable examples live in [`examples/`](examples/): `readme_quickstart` (load-set-run-read),
`zerocopy_simple` / `zerocopy_cache` and `dmabuf_fd_io` (caller-owned DMA-BUF I/O), `run_io` (generic
multi-I/O), `classify` / `predict` / `predict_cache` (CNN classifiers), `probe` (Vulkan device/feature report), `backend_switch` (per-backend routing), `op_check` (kernel + pipeline-cache smoke test), `profile` (per-op timings + chrome trace), `chat` / `vlm` (LLM and VLM device loops), and
`yonosplat` (the transformer encoder + rasterizer). [`app-demo/`](app-demo/) wraps the LLM, VLM, and
splatting paths in a four-tab Android app.

## Where VKNN fits

VKNN is an **on-device / edge AI inference engine** for **Android GPU acceleration** via **Vulkan
compute**. If you are searching for a way to run **ONNX models on Android**, do **on-device LLM
inference**, run a **vision-language model on a phone**, apply **int4 weight quantization**, or
render **3D Gaussian Splatting on mobile**, that is exactly this project. Compared with
[MNN](https://github.com/alibaba/MNN), [ncnn](https://github.com/Tencent/ncnn),
[TensorFlow Lite / LiteRT](https://ai.google.dev/edge/litert),
[ONNX Runtime Mobile](https://onnxruntime.ai/), or [llama.cpp](https://github.com/ggml-org/llama.cpp),
VKNN is smaller and GPU-first: one Vulkan backend that runs the *whole* model — CNN, transformer, or
both in one file — with the CPU reserved for verification, and a compiler that bakes optimization
into the model file instead of the app.

## License

MIT — see [LICENSE](LICENSE).
