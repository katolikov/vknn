# vxrt vs MNN — Vulkan, on-device benchmark

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
