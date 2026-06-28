# VKNN vs MNN — on-device benchmarks

Head-to-head against [MNN](https://github.com/alibaba/MNN) (Alibaba's production inference
engine) on the same device, same model, both at their fastest config. Every VKNN number comes from a
pipeline verified against an onnxruntime golden — fast and correct.

## Setup

- **Device:** an Android arm64-v8a device with an AMD RDNA-class mobile GPU, Vulkan 1.3+.
- **Precision:** fp16 on both (VKNN `--precision fp16`, MNN `precision=Low`), warm caches/tuning.
- **VKNN runner:** `vknn_classify --backend vulkan --precision fp16 --bench 20` (timed `run()` calls,
  including the host↔device pack/unpack).
- **MNN runner:** `MNNV2Basic.out model 20 0 <fwd> <mode> 2 1x3xHxW`. MNN has three backends here —
  Vulkan (`fwd=7`), CPU-4-thread (`fwd=0`), and OpenCL with HEAVY tuning (`fwd=3 mode=2`) — and they
  differ a lot, so "MNN-best" is the min over all three.
- **Thermal control is mandatory.** The device throttles 3–5× under sustained load, and VKNN (GPU-compute-bound)
  throttles more than MNN-Vulkan (overhead-bound). All numbers below use a 12–14 s cooldown
  **before each run**; absolute numbers and ratios from back-to-back sweeps are not trustworthy.

## VKNN vs MNN-Vulkan (fp16)

VKNN beats MNN's Vulkan backend on every model, by a wide margin on the small/depthwise nets:

| Model (Vulkan fp16) | VKNN median | MNN-Vulkan | speedup | VKNN vs ORT |
|---|---|---|---|---|
| MobileNetV2 | 2.8 ms | 13.8 ms | ~4.9× | cosine 0.99997 |
| MobileNetV3-Large | 2.5 ms | 17.0 ms | ~6.8× | cosine 0.99954 |
| SqueezeNet 1.1 | 2.4 ms | 10.9 ms | ~4.5× | cosine 0.99998 |
| EfficientNet-B0 | 4.2 ms | 19.9 ms | ~4.7× | cosine 0.99983 |
| ResNet-50 | 14.7 ms | 18.3 ms | ~1.25× | cosine 1.000000 |
| Inception-v3 | 18.3 ms | 25.6 ms | ~1.4× | cosine 0.99998 |
| YOLOv8n (640×640) | 17.5 ms | ~73 ms | ~4.2× | cosine 1.000000 |

YOLOv8n runs **100% on the GPU** (1 segment, no CPU fallback); the flat row-major op path keeps the
whole DFL / box-decode head on the GPU.

## End-to-end, per stage

A real inference is more than the GPU run: open the model, build the session, copy the input over,
run, and copy the result back. Each stage below is on the same device, both Vulkan fp16, warm (the
unified per-model `<model>.cache` — pipeline, prepacked weights, and tuning — already built):

| Stage | VKNN ResNet-50 | MNN ResNet-50 | VKNN MobileNetV3 | MNN MobileNetV3 |
|---|---|---|---|---|
| open model | 37 ms | —¹ | 6 ms | —¹ |
| create session | 268 ms | 960 ms | 211 ms | 904 ms |
| copy in (host→device) | 0.10 ms | —² | 0.10 ms | —² |
| run (inference) | 10.5 ms | 24.2 ms | 1.95 ms | 19.5 ms |
| copy out (device→host) | 0.03 ms | —² | 0.01 ms | —² |
| **end-to-end (load + 1 run)** | **~316 ms** | **~985 ms** | **~219 ms** | **~924 ms** |

¹ `MNNV2Basic` prints "Open Model" with no time; MNN's `createFromFile` is a few milliseconds.
² `MNNV2Basic` does not time the host↔device copies (the input is set once, outside the timed loop).
VKNN's are sub-millisecond because the device is UMA — there is no staging copy.

VKNN reaches a first result in roughly **3× less wall time**, almost entirely because MNN-Vulkan spends
~0.9 s compiling its pipelines at session creation, while VKNN builds the session in ~0.2–0.3 s from
its cached pipelines/weights and one pre-recorded command buffer. Steady-state inference is 2–10×
faster too, and the pack/unpack at the I/O boundary costs almost nothing.

Methodology: VKNN stages come from a small timer using the public API (`loadGraphBin` = open model,
`Runtime::load` = open + create session, `Config::timing` = pack / submit+gpu / unpack). MNN stages come
from `MNNV2Basic.out` (the `Resize` cost = create session, `Run Avg` = inference). Both warm, 12+ runs,
GPU cooled between measurements.

## VKNN vs MNN's absolute best (OpenCL, HEAVY-tuned)

MNN's true best is the min over its **OpenCL** (HEAVY-tuned), **CPU-4-thread**, and Vulkan backends.
The comparison is generous to MNN: the VKNN number is the full `run()` wall (it *includes* the
host↔device pack/unpack), while MNN's `Avg` times only `runSession` and sets the input once outside
the timed loop. VKNN is faster on **8 of 9** models:

| Model | VKNN wall (median) | MNN-best (backend) | result |
|---|---|---|---|
| SqueezeNet 1.1 | 1.66 ms | 2.59 ms (OpenCL) | **VKNN −36%** |
| MobileNetV2 | 2.30 ms | 3.11 ms (OpenCL) | **VKNN −26%** |
| MobileNetV3-Large | 2.84 ms | 3.78 ms (CPU-4t) | **VKNN −25%** |
| MnasNet 1.0 | 2.68 ms | 3.68 ms (CPU-4t) | **VKNN −27%** |
| EfficientNet-B0 | 4.34 ms | 9.29 ms (OpenCL) | **VKNN −53%** |
| Inception-v3 | 15.46 ms | 19.35 ms (CPU-4t) | **VKNN −20%** |
| DenseNet-121 | 13.90 ms | 15.37 ms (CPU-4t) | **VKNN −10%** |
| YOLOv8n (640²) | 20.00 ms | 24.71 ms (OpenCL) | **VKNN −19%** |
| ResNet-50 | 10.26 ms (cool) | 10.30 ms (OpenCL) | **parity** |

The conv-heavy nets (Inception, DenseNet, YOLO, ResNet) run a **tiled-GEMM Winograd** kernel — see below.

ResNet-50 sits at parity with MNN's best. From a cool device VKNN-wino runs it in 9.96 / 10.26 ms
(min/median) — faster than MNN's *buffer* OpenCL (10.51 ms) and even with MNN's *image* OpenCL
(10.30 ms). MNN keeps a small edge only when the device is already warm: VKNN slows to ~11.7 ms there
while MNN stays ~10.3. This is **not** an image-vs-buffer effect (MNN's SSBO/buffer path is just as
stable) — it is kernel power: MNN's GEMM draws a little less per inference, so it sits further from the
throttle threshold. Closing that last bit requires cutting VKNN's per-layer V/M traffic (~3 MB) or a
more ALU-efficient GEMM.

### Winograd: a tiled-GEMM kernel, autotuned per shape

MNN's OpenCL backend here is **ANGLE translating OpenCL → Vulkan**, so its winning kernels are Vulkan
compute too — reachable natively. MNN wins the 3×3-conv nets with **F(2,3) Winograd + a CLBlast-style
tiled batched GEMM** (`XgemmBatched`) for the transform-domain multiply. VKNN does the same:
`wino_input` → V, **`wino_gemm`** (an LDS-staged, register-blocked batched GEMM) → M, `wino_out` →
output. A naive matmul (1-thread-per-output, memory-bound) is what makes Winograd lose; the algorithm
is sound, GEMM quality is the determinant.

Winograd helps deep / square 3×3 (ResNet, DenseNet) but loses on small-channel or spatially-large 3×3,
so `tuneWino` measures the tiled-GEMM Winograd against the direct kernel **per shape** on scratch buffers
and caches the winner (like the local-size tune; default `fast` tuning, `Config::winograd` = `On`/`Off`
force it). Effect vs direct-only: DenseNet 15.5→13.9 (flips a tie to a win), Inception 16.0→15.5,
YOLOv8n 25.8→20.0, ResNet-50 12.6→12.1 (and ~10.5 cool). cosine ≥ 0.9995 throughout.

Several alternative GEMM/Winograd variants regress; they are kept as documented negative results
(`Config::setHint(Hint::WinogradVariant, …)`): a 2-pass naive matmul (~15 ms, memory-bound on the
global V round-trip), that split 4 ways (no help → bandwidth- not occupancy-bound), a fully-fused single
kernel with V in LDS (~88 ms, the static LDS array collapses occupancy), and a **subgroup-shuffle GEMM**
that shares operands across the 64-wide wave instead of LDS (~15 ms, +47% — on this driver
`subgroupShuffle` costs more than the LDS reads it replaces, and the GEMM is global-traffic-bound so the
swap doesn't touch the bottleneck). Packed fp16 in the GEMM inner loop is neutral (it is memory/LDS-bound,
not ALU-bound). int8 weight-only on the deep 1×1 has a bandwidth ceiling (~0.2 ms; RDNA-pre-4 gives
int8 == fp16 compute), so it cannot close ResNet alone.

**F(4×4,3×3)** is also implemented (`setHint(Hint::WinogradUnit, 4)`): it cuts the transform-domain V/M
traffic to 0.56× and the multiplies to 4× (vs F(2,3)'s 2.25×), and it is numerically fine at fp16
(ResNet cosine 0.999999 — the larger transform coefficients do *not* break half precision here). It is
**slower** on this GPU (~11.5 vs F(2,3)'s 10.5 ms): the 6×6 transforms hold `d[6][6]`+`t[6][6]` = 72
`vec4` per thread (register pressure) and the GEMM has 4× fewer tiles (less parallelism). The traffic
saving is real but the register-heavy transforms eat it — F(2,3) is the default; F(4,3) is a hint for
research / future transform-LDS work.

## YoNoSplat encoder (965M-param transformer)

The feed-forward 3D-Gaussian-Splatting encoder (DINOv2 ViT-L/14 backbone + RoPE decoders + Gaussian /
camera heads) runs **end-to-end on the GPU**, 1 segment over ~8700 nodes:

| Model | VKNN (Vulkan fp16) | MNN |
|---|---|---|
| YoNoSplat encoder (2 views → 100352 Gaussians) | ~13.5 s | cannot convert |

MNN's converter fails on the encoder's dynamic-shape geometry tail (`Reshape error 301056 → 6`,
"Model larger than 2GB"), so VKNN is the only engine that runs this model correctly on-device. The
GPU time is dominated by the 509 batched matmuls (~1538 GFLOP, ALU/latency-bound at ~142 GFLOP/s on
this driver); the rasterizer that consumes the 6 Gaussian outputs is a separate Vulkan compute pass
(see [../skills/run-yonosplat.md](../skills/run-yonosplat.md)).

## Measurement notes

- Set `Config::timing` (the `--timing` flag) for the real submit+GPU time (pack / submit+gpu / unpack). The per-op profiler
  sum is inflated by forced per-op barriers — relative only.
- Warm timings load the unified per-model cache (`<model>.cache`, the pipeline + prepacked-weight +
  autotune bundle). Delete that file before timing a fresh **cold** build. In `vknn_benchmark` /
  `benchmark/run.py`, `"cache"` (default `<model>.cache`) sets the cache path and `"generate_cache":
  true` populates it in an untimed throwaway load first, so the reported `load` is warm and the
  cache-build cost is excluded from `timing_ms`.
- VKNN's latency is very consistent (the whole static graph is one pre-recorded command buffer);
  MNN-Vulkan has higher cold-loop variance.

## Reproduce

```bash
./scripts/bench_vs_mnn.sh 20        # see the MNN SETUP block at the top of the script
```
