# ADR-0011: fp32-chained fused units (fusion v3)

## Status
Accepted (2026-07-06). Supersedes the fast-mode design in ADR-0010 §3.

## Context
ADR-0010's fast mode kept the strict per-step `vknnRte16` rounding for every fused unit except
three special-cased patterns (swish → `fusedAct`, 1x1-conv residual → `fusedResidual`, MatMul bias
→ `fusedBias`) that bypassed the VM entirely. Two problems surfaced when the suite was profiled
against per-model accuracy and runtime gates:

- **Per-step rounding dominated the mobile-net class.** On graphs whose chains never matched a
  special case, the integer-math RTE per VM step cost roughly half the model runtime:
  MobileNetV2 ran 4.0→1.8 ms on the Xclipse 960 once the per-step rounding was removed
  (MobileNetV3 -48%, MNASNet -44%).
- **The special cases were the wrong boundary.** They covered three fixed patterns; everything
  else paid full price, and the same fp32-accumulator trick generalizes to every unit.

## Decision
1. **The default fast mode fp32-chains every unit.** A fused unit carries the `pw_relax` attr:
   the entry still rounds to the byte the producer would store (`TO_STORE`), the steps chain
   unrounded in fp32 registers, and the unit rounds ONCE per stored stream (the main store and
   each `pw_outs` export) through the integer-exact `vknnRte16`. Consequences by construction:
   every inter-unit tensor stays on the unfused graph's value trajectory; single-step units and
   chains ending in a monotone activation are byte-identical to the unfused graph; multi-step
   chains round strictly fewer times, so unit accuracy is >= the unfused fp16 build. The three
   ADR-0010 special-case emissions are deleted (one emission path again); `--strict-fuse` keeps
   the per-step discipline and its absolute byte gate.
   - Full fp32 chaining with a RAW entry was measured and rejected: removing the producer-store
     rounding moves every inter-unit tensor off the unfused trajectory, and the end-to-end SNR
     walks both ways across a deep network (ResNet-50 -0.59 dB, MobileNetV3 -0.42 dB) even though
     every unit is locally more accurate. Rounding the entry restored ResNet-50 to exact parity
     and turned every suite cell fused >= unfused.
2. **The discipline is compiled, not branched.** Every epilogue-capable kernel builds two SPIR-V
   variants — `<stem>_epi` (strict) and `<stem>_epi_rx` (`-DPW_RELAX`) — and the standalone VM
   gets `fused_pw_flat_rx`/`fused_pw_nc4_rx` twins; the host picks by the node's attr. A runtime
   plan-flag branch carried both loop bodies in every kernel and cost the geometry-heavy models a
   consistent few percent (and mis-executed on the 960 driver: inceptionv3's fused accuracy jumped
   +2.2 dB when the branch became a compile-time variant).
3. **Standalone units monomorphize through spec constants.** Chains of up to 8 steps pass
   25 spec words ({numSteps, per-step kind<<16|code, p0 bits, p1 bits}) so the driver unrolls the
   step loop and folds the dispatch at pipeline creation; operand refs and broadcast geometry stay
   uniform plan-SSBO reads. Identical units share one pipeline via the spec-keyed session pool;
   longer chains keep the runtime interpreter (spec absent → NS=0 default). The standalone VM
   shaders are `VKNN_NO_RTE` like the elementwise kernels they replace — the float-controls RTE
   execution mode changes the driver's float codegen by 1 fp16 ulp on erf-class chains.
4. **The register-tiled GEMM does not host units.** `matmul_tiled` holds 64 fp32 accumulators per
   thread; an attached VM unit collapses its occupancy (+1.65 s of MatMul time on the 8-view
   3DGS encoder against -0.74 s of fused elementwise savings). `canHostUnit` refuses MatMul
   producers at the tiled sizes (M, N, K >= 32); those units run standalone, and a lone
   initializer bias-Add folds onto the MatMul's native `fusedBias` input — the fp32-accumulator
   add with one store rounding, i.e. the fp32-chained semantics at zero VM cost.
5. **Grouped Conv is gated off the GPU.** `supportsNode` rejects `1 < group < Cin` and depthwise
   with a channel multiplier: the dense GPU weight pack mis-indexes those (with a heap overread)
   and the dense kernel sums every input channel block, so results were silently wrong. They fall
   back to the group-aware CPU op; group == 1 and pure depthwise keep their GPU kernels.

## Gates (all on both devices, Xclipse 960 + 940)
- `--strict-fuse` vs `--no-fuse-pointwise`: byte-identical, 22/22 suite models + stand-ins +
  frame-interp.
- Fast mode vs unfused, SNR/cosine against onnxruntime-fp32 goldens with deterministic kernels
  (`--tuning none`): fused >= unfused on 10/10 CNNs (DenseNet +0.96 dB, MobileNetV3 +0.54 dB,
  YOLOv8n +2 dB, InceptionV3-960 +2.2 dB). Under autotuning the unfused baseline itself wobbles
  with Winograd race outcomes; gate comparisons must pin the kernel choice.
- Interleaved min-of-N runtime vs the pre-v3 main: MobileNetV2 -55/-61%, MobileNetV3 -47/-51%,
  MNASNet -44/-55%, ShuffleNet -5/-17%, ResNet-50 -2%, others within noise. Known cost: the
  frame-interp stand-in +2-3% (geometry-op mix; its real-model proxy, the 8-view encoder, is
  faster). 3DGS 8-view encoder: 9.11 → 8.84 s/iter with all 6 outputs at PSNR 63.4-80.1 dB.

## Consequences
- Fast fusion is the default at every `-O` level; the byte gate compiles with `--strict-fuse`.
- The `.vxm` format is unchanged (`pw_relax` is a generic attr; VXM3 files without it run strict).
- The embedded SPIR-V set roughly doubles for the epilogue stems (one variant per discipline).
- Accuracy gates that compare fused against unfused must run `--tuning none`, or the Winograd
  race noise in the baseline masks the comparison.
