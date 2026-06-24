# vxrt vs MNN — Vulkan, on-device benchmark

## Which models MNN benchmarks
MNN's own benchmark suite (`benchmark/models/` in the MNN repo) ships: **MobileNetV1, MobileNetV2,
SqueezeNet v1.0/v1.1, ResNet-v2-50, Inception-v3, MobileNetV3, NASNet**. We compare on the subset
vxrt currently runs end-to-end:

| Model (Vulkan, fp16, same device) | vxrt | MNN | correctness (vxrt vs ORT) |
|---|---|---|---|
| **MobileNetV2** | min 14.7 / median 17.9 ms | avg 14.0 / min 10.9 ms | cosine 0.999965, top-1 match |
| **ResNet-50** | min 60.4 / median 61.1 ms | avg 45.9 / min 44.0 ms | cosine 1.000000, top-1 match |

**Honest bottom line: MNN is still ~1.3–1.4× faster on both.** On MobileNetV2 *cold/best-case*
vxrt's full run (~10 ms, of which ~7 ms is GPU) is close to MNN's best (~10.9 ms), but under
sustained load vxrt throttles more. On ResNet-50 the gap is larger because ResNet is dominated by
3×3 group=1 convolutions, which vxrt runs with a straightforward (untiled) kernel while MNN uses
Winograd / tiled GEMM. SqueezeNet/Inception/MobileNetV3/NASNet need a few more ops (Concat on the
GPU, hard-swish, branchy graphs) before vxrt can run them.

What's been tried to beat MNN (and the result):
- f16vec4 vectorized kernels — **helped**.
- register-tiled 1×1 conv (reuse weights across output pixels) — **helped**.
- NEON fp16 input pack — **helped** (2.0→0.54 ms).
- shared-memory 1×1 GEMM — **hurt**: MobileNet's deep convs have tiny spatial (7×7), so one
  workgroup per output-block means no cross-workgroup LDS reuse, only barrier overhead.
- register-tiled 3×3 — **hurt**: register pressure dropped occupancy.
- **Winograd F(2×2,3×3)** for the 3×3 group-1 convs (input transform → 16 batched matmuls →
  output transform). **Implemented and numerically correct** (ResNet-50 cosine **0.999999**), but
  the un-fused 3-pass version is **memory-bound and ~15% slower** than the direct kernel on this
  GPU (71 ms vs 61 ms): writing/reading the transformed V (16/9× input) and M (16/4× output)
  buffers to global memory costs more than the 2.25× multiply reduction saves. It's behind
  `VXRT_WINOGRAD=1` (default off). To make Winograd win it must be **fused** — transforms kept in
  registers/LDS so V and M never hit global memory (this is what MNN does). That, plus
  texture-backed activations, is the concrete remaining path; it's a substantial effort.



Honest head-to-head against [MNN](https://github.com/alibaba/MNN) (Alibaba's production inference
engine) on the **same phone, same model, both at their fastest config**. No cherry-picking — the
raw tool output is reproducible with `scripts/bench_vs_mnn.sh`.

## Setup
- **Device:** Galaxy S26 (SM-S942B), Exynos 2600, **Xclipse 960**, Vulkan 1.4.304, Android 16.
- **Model:** MobileNetV2 (ONNX opset 12). For MNN it was converted with
  `MNNConvert -f ONNX --fp16` → `mobilenetv2_fp16.mnn`.
- **Both engines: Vulkan backend, fp16** (MNN `precision=Low`, vxrt `--precision fp16`), warm
  (caches/tuning built on a prior run), single inference thread of control.
- **MNN runner:** `MNNV2Basic.out model 30 0 7 1 2 1x3x224x224` (forwardType **7 = Vulkan**,
  precision **2 = Low/fp16**). Reports avg/min over the timed loops (no separate warmup, so the
  first loops are cold → its *avg* is slightly pessimistic; *min* is steady-state).
- **vxrt runner:** `vx_classify --backend vulkan --precision fp16 --bench 30` (5 warmup runs, then
  30 timed `run()` calls including the host↔device pack/unpack).

## Status after the optimization pass (2026-06-24)
After a round of Vulkan tuning (f16vec4 vectorized kernels, register-tiled 1x1 conv, NEON fp16
input pack, compute-only barriers, profiling-only timestamps), vxrt went from ~22 ms to roughly
**parity** with MNN-Vulkan:

| Metric (MobileNetV2, fp16, Vulkan) | vxrt | MNN |
|---|---|---|
| GPU kernel time (timestamp span, cool, stable) | **~7.0 ms** | n/a (MNN doesn't expose it) |
| Cold single inference (full run incl. I/O copy) | **~10 ms** | ~9.5 ms |
| Sustained / throttled (15-loop bench, hot device) | ~17–20 ms | ~14 ms |

Honest reading: **vxrt's kernels are now competitive** — its 7 ms of pure GPU work is in the same
range as MNN's ~7–9 ms full run, and cold single-shot is near-tied (~10 vs ~9.5 ms). **MNN still
wins under sustained load** (~14 vs ~18 ms): its kernels move less memory per inference, so the
SoC throttles less. So we did **not** definitively beat MNN; we closed a 1.5–3x gap down to ~1.0–1.3x
and matched it on best-case latency. To pull ahead, the remaining work is (a) shared-memory /
subgroup-cooperative GEMM for the large-channel 1x1 convs (the 0.5 ms hotspots), (b) depthwise+pointwise
fusion to cut dispatch count, and (c) image/texture-backed activations (better mobile-GPU cache).

The detailed pre-optimization numbers below are kept for the record.

## A note on thermal throttling (read before the numbers)
This phone throttles fast. A short burst (20 loops, cool) and a longer burst (30–100 loops, hot)
give very different numbers for the *same* binary. MNN is affected a lot; vxrt almost not at all
(its whole graph is one pre-recorded command buffer, so its time is dominated by steady GPU
occupancy). To be fair, the headline table uses **alternating 30-loop runs** so both engines see
the same thermal state; the cool-burst numbers are listed separately so nothing is hidden.

## Results (MobileNetV2, 224×224)

| Engine | Backend | Precision | Latency (equal-thermal, 30 loops) | Best case (cool, 20 loops) |
|---|---|---|---|---|
| **MNN** | Vulkan | fp16 | **~15.3 ms** avg (min ~11 ms) | **~7.7 ms** avg (min 6.8 ms) |
| **vxrt** | Vulkan | fp16 | **~22.2 ms** median (p90 22.9) | ~23 ms median |
| MNN | CPU (4-thread, optimized) | fp16 | ~2.7–3.2 ms | — |
| vxrt | CPU (scalar reference) | fp32 | ~672 ms | — |

Raw, repeated (alternating, 30 loops each):
```
MNN-Vulkan : Avg= 15.20 ms  min= 11.03 ms   vxrt-Vulkan: median=22.10 ms p90=22.52 ms   (round 1)
MNN-Vulkan : Avg= 15.47 ms  min= 10.71 ms   vxrt-Vulkan: median=22.12 ms p90=22.47 ms   (round 2)
MNN-Vulkan : Avg= 15.38 ms  min= 12.72 ms   vxrt-Vulkan: median=22.63 ms p90=22.92 ms   (round 3)
```
Cool 20-loop burst (device idle beforehand): MNN-Vulkan Avg 7.68 / min 6.84 ms; vxrt 23.3 ms median.

## Honest read

- **On Vulkan, MNN beats vxrt — by ~1.5× at equal thermal (≈15 vs ≈22 ms), and by ~2–3× in
  MNN's best case (≈7 vs ≈22 ms).** vxrt is genuinely slower. This is expected:
  it's a from-scratch engine with **straightforward NC4HW4 kernels**, while MNN ships
  shape-specialized, register-blocked, vectorized Vulkan kernels tuned over years.
- vxrt's latency is **very consistent** (median 22.1, p90 22.6) because the whole static graph is
  one pre-recorded command buffer; MNN's per-run variance is larger.
- Two concrete reasons vxrt is behind, both already in [LIMITATIONS.md](LIMITATIONS.md) /
  [SUMMARY.md](../SUMMARY.md) as next steps:
  1. **CPU↔device pack/unpack** at the I/O boundary costs ~11 ms of vxrt's ~22 ms (vxrt profiler
     shows only ~12 ms is actual GPU compute). Doing the NC4HW4 conversion on the GPU (the
     `pack`/`unpack` shaders already exist) or feeding NC4HW4 directly via ION would remove most
     of that.
  2. **Naive conv kernels** — no Winograd, no subgroup-tiled GEMM, scalar-ish fp16 loads.
- **Do not read the CPU rows as "vxrt's CPU vs MNN's CPU".** vxrt's CPU backend is a *scalar
  correctness oracle* (it's what the GPU is validated against), not an optimized backend; MNN's
  CPU backend is a production SIMD/threaded one. Also note that on this SoC MNN's optimized CPU
  (2.7 ms) actually beats its own Vulkan (15 ms) for this small model — GPU dispatch/transfer
  overhead dominates at this size, which is a useful reminder that "GPU" isn't automatically
  faster for MobileNet-scale workloads.

## Correctness (not just speed)
vxrt-Vulkan-fp16 matches the onnxruntime golden: **top-1 = class 258, cosine = 0.999965**. The
benchmark above is on a numerically-verified pipeline, not a fast-but-wrong one.

## Reproduce
```bash
# after the MNN SETUP block in scripts/bench_vs_mnn.sh:
./scripts/bench_vs_mnn.sh 30
```
