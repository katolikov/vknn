# Release verification — 2026-07-02 (branch `fuse-pointwise-epilogue`)

Device: Samsung Xclipse 940 (`R5CWB2KWVJY`), fp16 (`--precision low`), warm timing = median/min over
10-15 in-process iterations after warmup (`benchmark/run.py`, per-stage results under
`benchmark/result/release-suite-final/`). Every stage runs with `fold_islands=false` — the fallback
count is over ALL ops.

## CNN suite — golden = onnxruntime fp32, fixed random input

| model | run median / min ms | fallbacks | cosine | SNR dB | verdict |
|---|---|---|---|---|---|
| densenet121 | 17.5 / 17.2 | 0 | 0.999994 | 48.9 | PASS |
| efficientnet_b0 | 4.1 / 3.8 | 0 | 0.999980 | 43.7 | PASS |
| inceptionv3 | 27.1 / 26.7 | 0 | 0.999996 | 49.8 | PASS |
| mnasnet1_0 | 3.0 / 2.7 | 0 | 0.999989 | 46.6 | PASS |
| mobilenetv2 | 2.7 / 2.6 | 0 | 0.999993 | 48.1 | PASS |
| mobilenetv3 | 2.9 / 2.7 | 0 | 0.999982 | 44.5 | PASS |
| resnet50 | 17.1 / 16.5 | 0 | 1.000000 | 64.8 | PASS |
| shufflenet_v2 | 3.6 / 3.2 | 0 | 0.999998 | 53.4 | PASS |
| squeezenet | 2.6 / 2.5 | 0 | 0.999999 | 59.4 | PASS |
| yolov8n | ~27 | 0 | 1.000000 | 65.4 | PASS |

All fp16-floor accuracy. resnet50 median improved 18.2 → 17.1 ms across the session (conv1x1 WTILE
autotune); efficientnet_b0 5.0 → 4.1 ms.

## Pointwise-epilogue probe matrix

22 per-family probes (conv 3x3/1x1/1x1-s2/deep-splitk/3x3-winograd/dwconv, convtranspose, softmax
flat+NC4, layernorm(±beta), reduce(axes attr+input), gridsample, resize, avg/max/global pool,
matmul, gemm(±bias), same-shape NC4 constants, dead-lane channel counts) × fp32 + fp16 + a
winograd-forced run: **45/45 byte-identical** fused vs `--no-fuse-pointwise`, 0 fallbacks, on real
(non-zero) inputs.

## YoNoSplat encoder (8-view, 3.1 GB fp16 vxm)

- fused vs nofuse: **6/6 outputs byte-identical**, 0 fallbacks, 1 Vulkan segment.
- vs dl3dv goldens: cosine 0.999988–0.999996, SNR 46.0–55.2 dB (fp16 floor).
- submit+gpu 17.07 s; peak device memory 3392 MB / 3628 buffers (the former matmul-epilogue OOM
  case — root cause was the `supportsNode` input-count gate, fixed).

## Frame-interp artifact (rebuilt from model.json via tools/rebuild_onnx_from_dump.py)

139 nodes post-passes; uint8 y/uv I/O, fp16 compute, 2 runtime-grid GridSamples, 10 fused pointwise
chains; every intermediate fp16-finite.

- 0 fallbacks (fold-islands off); fused == nofuse **byte-identical** (3/3 outputs).
- fp32: GPU vs CPU-backend cosine 1.000000, max|d| ≤ 2 fp16-output ULPs; uv output byte-identical.
- fp16: img2 path 38.9–43.9 dB vs CPU/ORT; img1 path 22.8 dB (Reciprocal/Pow amplification of fp16
  storage rounding — fp32 run proves the engine exact); uv 41/1.4M pixels ±2 steps.
- submit+gpu ≈ 52 ms (2×720p warp + color pipeline).

## Bugs found and fixed by this verification pass

1. `supportsNode(MatMul)` counted epilogue operands → silent CPU fallback → 47-segment
   fragmentation → the "matmul epilogue host OOM" (`pwCoreInputs` now bounds every positional read,
   incl. inferShapes' Resize sizes — which silently pushed fused Resize to the CPU).
2. Gather dropped the axis for rank-1 `[1]` indices (ONNX splices indices shape); the collapsed rank
   mis-broadcast downstream and OOB-read in BinaryCpu (intermittent host SIGBUS).
3. run_io: `--cache-mode`'s value was collected as an input file and unopenable inputs silently fed
   zeros — device runs "passed" on wrong data. Values skipped; missing input = hard error.
4. benchmark cosine: byte-identical all-zero outputs scored 0 and FAILed.
5. ReduceMean→GAP unconditional import corrupted non-spatial reductions (now Reduce(Mean) +
   shape-aware lowerReduceToGap).
6. NC4-world constant chain operands uploaded flat were misordered for C%4≠0 / H·W>1 and OOB-read
   pad lanes (now NC4HW4-packed at upload; scalar splat = broadcast mode 3).

## Re-verification after the Range/import fixes (same day, later session)

New engine work since the table above: ONNX Range op; scalar-Gather `idx_scalar` tagging;
inferShapes never resolves from unresolved operands (the Binary/Add "scalar-or-unresolved"
conflation poisoned transformer ranks); Reshape refuses a -1/0 target while its input is
unresolved; NumPy zero-dim broadcast (a 0 dim propagates, never max'd to 1 — fixes a BinaryCpu
null-deref on empty folded constants inside constFold); fold+infer loop runs to convergence
(cap 256, was 8); Range appended to the END of OpType (a mid-enum insert shifts the raw ints
model_io serializes and corrupts every existing .vxm — caught when the prebuilt 8-view vxm
decoded ScatterND as EyeLike and OOM'd; the enum is append-only now).

Plus `pruneDeadInitializers`: const-folding materializes every intermediate of a folded chain and
the unbounded Cast fold copies whole weight tensors; the orphaned payloads serialized into the
.vxm. On the 8-view export that was 7,045 MB of orphans — a 6.05 GiB vxm for 1.9 GiB of real
fp16 weights (and the root of the old 3.14 GB / v2 5.39 GB sizes). Now 2.34 GB: 1.80 GiB fp16
weights + 0.38 GiB live ScatterND meshgrid index.

Every gate re-run on the final ABI + pruning binary:

- Host tests 34/34.
- Probe matrix **45/45** byte-identical fused vs nofuse, 0 fallbacks.
- CNN suite **10/10 PASS**, 0 fallbacks, timings at baseline (resnet50 16.1 ms, effnet 4.1,
  mnv3 2.9, inception 27.7, densenet 17.3).
- YoNoSplat 8-view (prebuilt 3.1 GB vxm, new runtime): fused==nofuse **6/6 byte-identical**,
  0 fallbacks, 1 segment, peak 3392 MB — pre-session numbers reproduced exactly.
- Frame-interp (artifact regenerated from model.json): fused==nofuse **byte-identical** at fp16
  AND fp32, 0 fallbacks, fp32 GPU==CPU cosine 1.000000 (uv byte-identical). The fp16-vs-ORT
  img1 SNR is low on this regeneration (Reciprocal/Pow amplification of rounding order on the
  synthetic constants); the pre-session engine produces identical numbers to the decimal, so it
  is a property of the artifact, not an engine change.

## YoNoSplat 8-view REGENERATED (real dl3dv weights) + pushed to HF

Export rebuilt from scratch after the reboot wiped /tmp/YoNoSplat:
`Benchmark_GPU/yonosplat_export/export_real.py` = the surviving perf harness + ckpt load
(dl3dv_224x224_ctx2to32.ckpt from HF botaoye/YoNoSplat, 0 missing / 0 unexpected) + a
Newton-Schulz polar orthogonalization for the camera head (the faithful torch.svd substitute;
the harness's Gram-Schmidt is only valid for random weights) + legacy tracer (`dynamo=False`) +
the DOUBLE->FLOAT retype (758 casts, the fp64-Einsum ORT failure). Faithfulness gates:
eager-vs-saved-goldens cos 1.0000000 (all 6), traced-ONNX-in-ORT cos 1.0000000 (all 6).

Device (fp16, 0 fallbacks, 1 segment over 7696 nodes, peak 3394 MB): fused==nofuse
**8/8 byte-identical**. fp16 GPU vs ORT fp32 goldens:

| output | cosine | PSNR dB | SNR dB | relL2 | min\|d\| | max\|d\| | NaN |
|---|---|---|---|---|---|---|---|
| covariances | 0.999989 | 79.4 | 46.0 | 5.0e-3 | 0 | 1.2e-3 | 0 |
| harmonics | 0.999998 | 76.7 | 55.2 | 1.7e-3 | 0 | 9.4e-2 | 0 |
| means | 0.999992 | 67.2 | 47.2 | 4.4e-3 | 0 | 3.6e-2 | 0 |
| opacities | 0.999997 | 68.6 | 51.6 | 2.6e-3 | 0 | 4.4e-3 | 0 |
| rotations | 0.999996 | 62.7 | 50.7 | 2.9e-3 | 0 | 3.8e-2 | 0 |
| scales | 0.999998 | 70.0 | 51.5 | 2.7e-3 | 0 | 3.3e-3 | 0 |

Same fp16-floor SNR band (46.0–55.2 dB) as the pre-session verified build. Uploaded to
huggingface.co/katolikov/yonosplat-vknn: README, yonosplat_encoder.onnx (external-data location
repointed to weights.bin), weights.bin 3.86 GB, encoder8_fp16.vxm 2.34 GB, dl3dv inputs + 6
goldens.

## yonosplat_v2 (2-view): CLOSED

The recompiled vxm is 1.89 GiB (was 5.39 GB - dead-initializer pruning plus the honest-shape
fixes; 55 fold rounds, 0 oversized tensors, 235 chains fused / 122 epilogues). Device gate:
0 fallbacks, 1 Vulkan segment over 7623 nodes (a ConstantOfShape above the fold bound got a
flat-fill GPU kernel), run-to-run deterministic, fused==nofuse 8/8 byte-identical, on both
test devices (2.5 s submit+gpu on the newer GPU).

## Second device R3CY905E04M: RoundingModeRTE miscompilation, root-caused and FIXED

The device (a newer driver generation than R5CWB2KWVJY) executed the 8-view encoder
nondeterministically — different output bytes every run at BOTH precisions, surviving per-op
barriers, single submit, fusion off, tuning off, and a reboot — with occasional kernel-level
crashes/reboots under load. Golden metrics there (SNR 16–23 dB) were corruption noise. The
localization ledger: every kernel class is GPU==CPU exact on this device in isolation (model
shapes, odd-M edge tiles, >65535-workgroup dispatch splits, both precisions, 12 sustained
back-to-back runs); per-op-type CPU disables all still raced; all 10 CNNs bit-exact at the same
SNR floor as the other device. A per-shader-group bisect of the RoundingModeRTE execution mode
landed it: **the driver miscompiles the GEMM kernels when they carry the float-controls
execution mode.** The same driver also rounds `float16_t(x)` AND `packHalf2x16` toward zero, so
the fix is an integer-math round-to-nearest-even store (`TO_STORE`/`vknnRte16`, shaders/
store16.glsl) in the GEMM family, execution mode retained everywhere else. Verified: on the
older driver all outputs byte-match the execution-mode build (the conversion is bit-exact RTE);
on this device the encoder is now deterministic at the full RTE band — SNR 46.0–55.2 dB,
digit-identical to the other device's table. Timing here: 8-view fp16 10.2 s run / 12.2 s load
(vs 17.0 s / 12.2 s).
