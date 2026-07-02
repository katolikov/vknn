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

## Open

- yonosplat_v2 (2-view) recompile is 5.39 GB (initializer growth vs the 3.1 GB 8-view build of a
  different export) — forensics tracked separately; the 8-view vxm remains the device target.
