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
  `benchmark/scripts/dev_perfab.sh` scripts this protocol for two-build A/B (see Gates and scripts).

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

A cooled protocol measures the **parked / ramping regime**: after the cool-down the GPU has
power-collapsed, so the timed run pays the clock ramp exactly as an intermittent caller (camera frames
tens to hundreds of ms apart) does. `vknn_run_io --repeat N --gap-ms G` reproduces that intermittent
regime directly — a sleep of `G` ms between iterations, each still printing its own `--timing` line —
and adding `--power high` shows what the idle-time keep-alive (`Config::power`, [config.md](config.md))
recovers in it. A back-to-back loop never idles long enough to park, so `--power high` changes nothing
there, and every cross-engine number in this document is measured with it off.

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
so `tuneWino` picks Winograd vs the direct kernel by a **deterministic shape rule** (`Cin·Cout ≤ 32768`
— Winograd's fp16 transform-domain intermediates grow with `Cin·Cout` and go memory-bound above that).
The rule is applied identically at every `--tuning` level, so the choice (and the output bits) never
depends on thermal state or measurement noise — only the bit-neutral tile (`RM`) is still timing-raced
and cached. `setHint(Hint::Winograd, Mode::On/Off)` — the runner's `--winograd on|off` — forces it.
Effect vs direct-only: DenseNet 15.5→13.9 (flips a tie to a win), Inception 16.0→15.5,
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

### Conv register tiles: an autotuned OCB x WTILE axis (v1.4.0)

The register-tiled conv kernels expose a second tile axis: `conv1x1`/`conv1x1_s2` tile OCB output
channel-blocks per thread (1 or 2) on top of the WTILE pixel axis, and the general `conv_reg`
kernel's WTILE (4 or 8) joins its OCB spec constant. An OCB=2 thread reuses every input vec4
across 8 output channels instead of 4 — up to twice the arithmetic per loaded operand — at higher
register pressure, so which tile wins is shape- and device-dependent and is raced per shape by the
existing bit-neutral autotune (`--tuning fast`/`heavy`). Every candidate computes each output with
the identical fp32 accumulation sequence, so the choice never affects output bits: `none`, `fast`
and `heavy` stay byte-identical, and a v1.4.0 build is output-byte-identical to v1.3.1 on the whole
CNN suite (verified per model at `none` and `fast`).

Two autotuner fixes make the wider candidate set safe. The race now issues a compute barrier
between its timing reps: unbarriered reps overlap on the GPU, which systematically favors
low-workgroup-count tiles that lose in the real (op-barriered) command buffer — with the old race
one suite model regressed 19% from exactly such a mis-pick; barriered, the same model gains 7%.
New-class candidates also carry a 3% anti-noise margin over the classic tile, so a single noisy
race sample cannot displace the proven default.

Beyond the register tiles, the release adds four deterministic-rule/kernel changes (all
run-to-run and cross-tuning byte-identical; accuracy within 0.04 dB of the previous release with
cosine unchanged, and BETTER on DenseNet-121 (+2.4 dB) and YOLOv8n (+0.9 dB)):

- **General split-K conv** (`conv_splitk` partial + `conv1x1_reduce`): a deep-reduction conv into a
  small output map (a stride-2 3x3 into 7x7, an 8x8-map Inception branch) is parallelism-starved on
  the register-tiled kernels; the split-K pair reaches 22-39% faster on those shapes. The routing is
  a calibrated shape rule (`taps >= 320` — or `>= 256` for multi-tap kernels — `OHW <= 64`), with
  Kahan-compensated fp32 partials so the two-pass sum tracks the true value tighter than the
  single-pass chain. `setHint(Hint::SplitKConv, Mode::Auto|On|Off)` overrides the rule (Auto is the
  default and the rule is active out of the box); the hint is a cache-variant key field.
- **Conv input-affine prologue** (`core/input_affine.h`, `shaders/input_affine.glsl`): a pointwise
  unit whose producer cannot host it - a BatchNorm affine plus ReLU after a zero-copy Concat view
  (every DenseNet dense-block conv), a squeeze-excite scale on a graph tensor - is applied by its
  single consumer conv at input load instead of running as a standalone FusedPointwise node that
  writes the transformed tensor for the conv to read back. The pass attaches a per-channel scale,
  a per-channel shift and an activation as `pro_*` attributes; conv1x1, its strided and split-K
  forms, the row-halo, register-tiled and direct kernels fold it inline in a `_pro` variant (one
  fma and the activation per loaded input vec4, unrounded fp32 straight into the multiplies), and
  every other conv family applies it through one `input_affine` dispatch into scratch. It is
  attached only in the pass's relaxed mode, where a fused unit already keeps fewer roundings than
  the unfused graph (a strictly-rounded graph keeps the standalone unit), and only to a conv of at
  least nine taps: the prologue runs once per loaded input vec4, which a 3x3 row kernel reuses
  over 36 x OCB multiplies (DenseNet-121's 3x3 convs 0.131 -> 0.122 ms with their units gone)
  but a 1x1 kernel over only 4 x OCB, where it ran 1.8-2.2x slower than the unit it replaced, so
  pointwise consumers keep the standalone unit - which, when it is such an affine, runs on the same
  vec4-per-thread `input_affine` kernel (one scale / shift pair hoisted per channel-block) instead
  of the per-element VM interpreter. The sync check
  (`tools/check_epi_sync.py`) pins the `_pro` surface like the epilogue's.
- **Lane-split fully-connected heads** (`fc_split`, `core/gemm_dispatch_rules.h`): a Gemm with at
  most 16384 outputs (every classifier head) puts 16 lanes on each output with a fixed-order
  shared-memory reduce instead of one thread per output; a 2048->1000 head that ran 16 waves over
  its 4 MB weight read at 23 GB/s dispatches 250 workgroups. A shape rule, never a race, so the
  summation order is deterministic.
- **Per-op traffic audit** (`benchmark/scripts/traffic_audit.py model.onnx profile.log`): joins a
  `--profile` log with the ONNX graph and sets every op's minimal DRAM traffic (fp16 NC4HW4
  activations, fp16 weights) against its GPU time - achieved GB/s and, for MAC-bearing ops,
  TFLOPS - per op and per op type, with elided zero-copy views excluded. Ops far below both the
  device's stream rate and its arithmetic ceiling are latency-bound and the ones to restructure or
  fuse; ops at the stream rate move only with less traffic.
- **Pointwise split-K rule recalibrated** (`ops/pw_splitk_rule.h`): with the chunk decode sized to
  the map, the 1x1 split pair wins only where the output plane has at most 16384 channel-block
  pixels (`Coutb * OH*OW`) and at most 128 of them per input channel-block: 128->64 @14x14 wins
  17%, 128->128 @14x14 ties, 256->128 @14x14 wins 25%, 256->256 @14x14 ties, 512->256 @14x14 wins
  16%, 480->80 @14x14 wins 52%, and the register-tiled kernel wins everywhere above the cap (every
  14x14 plane of 128+ output blocks, every 7x7 plane of 512). The part count targets 384 waves of
  partial-pass threads with at least 16 (input block, tap) steps per part (four parts at 6272
  outputs, two at 12544, seven for a 120-block reduction on a 3920-output plane; the general KxK
  split-K kernel counts its taps, so DenseNet's 128->32 3x3 at 7x7 keeps its 16 parts). ResNet-50's 2048->512 @7x7
  went 0.165 -> 0.117 ms.
- **Sliding-window 1-D conv** (`conv_1d`): 1xK/Kx1 kernels (Inception's 7x1/1x7, 3x1/1x3) load the
  input window into registers once per channel-block and reuse it across every overlapping tap;
  joins the bit-exact direct race.
- **16x16 LDS-halo tile**: the 3x3 halo kernel's tile edge is now raced (8 or 16); the 16 tile cuts
  halo overhead from 1.56x to 1.27x of the tile's input reads.
- **Winograd needs Cout >= 64**: below two `wino_gemm` N-tiles the transform-domain GEMM starves
  (DenseNet's 58 128->32 growth convs ran at ~365 GF/s on it); those shapes take the direct race.
- **Small-axis softmax**: a softmax row narrower than 32 (a detection head's 16-bin DFL
  distribution) runs one thread per row instead of a 128-lane workgroup per row — YOLOv8n's DFL
  head dropped from 13x its copy floor to near it.

Measured, cooled interleaved A/B vs v1.3.1 (`benchmark/scripts/dev_perfab.sh`, min over 5-8
paired iterations, `--tuning fast`, fp16, per-model `submit+gpu` wall; primary benchmark device /
second smaller-GPU device):

| Model | primary device | second device |
|---|---|---|
| Inception-v3 | **-21%** | **-23%** |
| ResNet-50 | **-5%** | **-16%** |
| DenseNet-121 | **-3%** | **-13%** |
| YOLOv8n | **-9%** | **-6%** |
| MnasNet 1.0 | **-8%** | **-4%** |
| MobileNetV3-Large | **-5%** | **-9%** |
| MobileNetV2 | -3% | -4% |
| ShuffleNetV2 | -4% | -3% |
| SqueezeNet / EfficientNet-B0 | within the 3% gate | within the 3% gate |

No model regresses beyond the 3% gate at none/fast/heavy on either device (the sub-2.5 ms models
are judged by the cached-tune protocol — tune once, time from the cache — because a fresh
`--no-cache` race right before the timed window heats the GPU in proportion to the candidate
count). Run-to-run determinism is gate-verified per model: two fresh `fast` runs, a `none` run and
a `heavy` run produce byte-identical outputs. LLM and VLM token streams and the YoNoSplat encoder
outputs are unchanged.

**F(4×4,3×3)** is also implemented (`setHint(Hint::WinogradUnit, 4)`): it cuts the transform-domain V/M
traffic to 0.56× and the multiplies to 4× (vs F(2,3)'s 2.25×), and it is numerically fine at fp16
(ResNet cosine 0.999999 — the larger transform coefficients do *not* break half precision here). It is
The output tile is picked per shape by a deterministic cost model (F(4,3) wins on deep channels,
F(2,3)'s smaller transform wins on shallow); `setHint(Hint::WinogradUnit, 4)` forces F(4,3) on every
3×3, bypassing even the Winograd-vs-direct shape rule. Its transforms are separable two-stage
kernels like F(6,3)'s: six cooperating lanes per (channel-block, tile) unit, one per transform
column in the first stage (staged in shared memory) and one per row in the second, ten units per
64-lane workgroup (`winoTransformGroups` in `core/wino_f63.h` sizes the dispatch). One thread per
unit held `d[6][6]`+`t[6][6]` = 72 `vec4` and ran a deep small map's 1024 units as 16 waves of a
36-load/36-store job, latency-bound at twice the reference engine's transform time; the six-lane
form runs the 256x256 @14x14 transforms in 12 us each (from 27) and the 128x128 @28x28 pair in
15 + 18 us (from 27 + 29), and ResNet-50's GPU span went 8.45 -> 8.2 ms.

### F(6,3) stays hint-only (negative result)

**F(6×6,3×3)** is implemented too (`setHint(Hint::WinogradUnit, 6)`, separable two-stage LDS
transforms — `shaders/wino_input6_fp16.comp` / `wino_out6_fp16.comp`, points and derivation in
`src/core/wino_f63.h`). It stays **explicit-hint only**: measured on the primary device against the
ORT goldens, it is **refused promotion into the automatic F-unit rule** on three independent
grounds. Protocol: cooled interleaved paired runs, min of 5, per model at `fast` and `none`, plus
per-op GPU profiles and single-shape probes (a 3-conv chain per shape, min of 3 cooled rounds);
DenseNet-121 carries the noise floor because it has *no* Winograd-eligible conv (every 3×3 is a
128→32 growth conv, under the `Cout >= 64` gate), so any delta it shows is drift.

- **Accuracy regresses wherever it runs.** Against the ORT goldens, PSNR / SNR in dB. Both F(6,3)
  configurations are listed because they answer different questions, and each is identical at every
  tuning level (the F-unit fixes the output bits; the GEMM-body race is bit-neutral):

  | model | default rule (F(2,3)/F(4,3)) | F(6,3) on the candidate rule's class | F(6,3) forced on every eligible 3×3 |
  |---|---|---|---|
  | ResNet-50 | 82.92 / 65.42 | 82.35 / 64.85 | 75.59 / 58.08 |
  | Inception-v3 | 64.32 / 47.82 | 61.50 / 45.00 | 61.50 / 45.00 |
  | YOLOv8n | 86.86 / 66.36 | 86.26 / 65.76 | 85.97 / 65.46 |
  | DenseNet-121 | 70.83 / 52.78 | unchanged | unchanged |

  The middle column is what promoting F(6,3) into the automatic rule would have shipped; the right
  column is what `setHint(Hint::WinogradUnit, 6)` costs today, and it is worse because forcing the
  unit also drags shapes the Winograd-vs-direct rule keeps on the direct kernel (ResNet-50's
  512×512 @ 7×7 and 256×256 @ 14×14) into the transform domain. Inception-v3's two F(6,3) columns
  agree to 2 dp without being the same output (cosine 0.999988 vs 0.999986). The loss is
  structural, not shape-dependent: the 8×8 transform carries A^T entries up to 32 against F(4,3)'s
  8, so the fp16-stored V/M intermediates lose relative precision on every shape. Same-or-better
  accuracy is a hard gate and no shape rule buys it back.
- **The speed win is not a function of the shape.** Isolated probes separate cleanly on channel
  work per output pixel — `Cin*Cout/(OH*OW)` ≤ ~10 wins (−19.5% at 64×64 @ 28×28, −12.5% at 64×64 @
  40×40, −12.0% at 96×96 @ 35×35, −10.8% at 32→64 @ 147×147), ≥ ~21 loses (+3.5% at 128×128 @
  28×28, +29.8% at 256×256 @ 28×28, +430% at 512×512 @ 7×7) — but **in-model the same shapes
  reverse**: at `none`, Inception-v3's 64→96 and 96→96 @ 35×35 run **+11%** and **+15%**,
  ResNet-50's 64→64 @ 56×56 **+5…8%**, while YOLOv8n's 64→64 @ 40×40 and 80×80 stay ahead
  (−4.8% GPU total). Two models disagree on near-identical shapes, so the discriminating variable
  is the surrounding graph, which a deterministic shape rule (ADR-0009) may not read.
- **What win remains is conditional on the tuning race.** With the bit-neutral GEMM-body race
  (`fast`/`heavy`) F(6,3) took Inception's 35×35 shapes by ~30%; without it (`none`) it lost the
  same shapes by ~13%. F(6,3) needs the register-tile body to be competitive at all, so a rule
  admitting it would pay off at one tuning level and regress at another.

Two measurement lessons are worth keeping. Independently generated caches make an A/B lie: two
tune passes race the *non*-Winograd kernels differently, and DenseNet-121 — which cannot change —
read −6.7% at `fast` purely from that, while a per-op A/B attributed a 3.6% ResNet-50 "regression"
to three **1×1** convs whose tile choice had flipped. And a forced-unit hint must run the same
bit-neutral race the automatic path runs, or every forced measurement is handicapped against
`auto` by the body it never got to choose (up to ~29% on these shapes).

### Barrier hygiene, ChannelShuffle, and the register-tile Winograd GEMM (v1.4.1)

Four further changes, all output-byte-identical to v1.4.0 per model at every tuning level
(verified per model at `none` and `fast` on both devices; run-to-run determinism gate green):

- **synchronization2 scoped barriers + write-after-read elision**: inter-dispatch barriers narrow
  their access scopes to storage reads/writes (every operand is an SSBO), and a write-after-read
  hazard on a reused liveness-pool slot emits an execution-only barrier instead of a full memory
  barrier. One portability finding is baked into the fallback: the target mobile driver drops a
  zero-memory-barrier sync1 `vkCmdPipelineBarrier` outright, so the elision only activates through
  the honored sync2 form (`VK_ACCESS_2_NONE`), and sync1-only devices keep the full barrier.
- **ChannelShuffle as one dispatch**: the Reshape(rank-5) + Transpose + Reshape group interleave
  folds at import into a dedicated layout-agnostic operator that runs in NC4HW4 or the flat
  layout without forcing converts. ShuffleNetV2's 16 shuffle blocks drop from 3 dispatches + 2
  full-tensor layout round-trips each to 1 dispatch (graph: 173 -> 104 nodes).
- **Register-tile Winograd GEMM** (`wino_gemm_reg`): a no-LDS twin of the tiled GEMM (direct
  global reads through the cache hierarchy, same per-output fp32 chain — bit-exact) joins the
  bit-neutral body race, and wins the ResNet-50 Winograd shapes on the primary device. With the
  stronger GEMM the Winograd-vs-direct rule gains a second branch: a large-`Cin*Cout` 3x3 also
  takes Winograd when the output map keeps the GEMM fed (`OHW >= 49`; probe-calibrated with the
  outer-product GEMM: 512x512 @ 7x7 0.63 -> 0.28 ms (ResNet-50's layer4 blocks), 256x256 @ 14x14
  0.33 -> 0.23, 256x256 @ 20x20 -38%, 192x192 @ 35x35 -42%, 512x512 @ 28x28 -69%). Two things
  make the small maps win: the unit rule is map-aware (`winoAutoUnitForMap`: F(4,3) yields to
  F(2,3) where its four times fewer tiles would no longer fill the GEMM's smallest M tile, so a
  7x7 map runs 16 F(2,3) tiles instead of 4 F(4,3) tiles at 75% padding), and the bit-neutral tile
  race carries RM = 2 beside 4 and 8, the tile a 16-tile map fills exactly.
  A per-thread N micro-tile (2 or 4 output channel-blocks per thread, raced beside the M tiles)
  is a measured negative result on the primary device: it loses on every ResNet-50 shape (512x512
  @ 7x7 0.33-0.41 vs 0.27 ms, 256x256 @ 14x14 0.20-0.28 vs 0.19), fewer waves costing more than
  the U reuse buys, so the body keeps one channel-block per thread. What does move the deep
  shapes is memory-level parallelism: the body's K loop runs two input channel-blocks per trip,
  requesting both blocks' U rows and V tiles before either block's fmas (the per-output fma
  sequence is unchanged, so the race stays bit-neutral). A 16-tile GEMM at 512 channels streams
  an 8 MB U from DRAM and is set by the loads a thread keeps in flight: 512x512 @ 7x7 0.272 ->
  0.248 ms, 256x256 @ 14x14 0.191 -> 0.180, 128x128 @ 28x28 0.113 -> 0.099 (3-pass race times).
- **Outer-product tap contraction.** Every group-1 conv kernel (direct, register-tiled, row-halo,
  compact stem, LDS-halo, 1-D, split-K, the pointwise family and the Winograd GEMM) folds a tap
  as `acc = fma(vec4(in.c), w[c], acc)` over the four input lanes, with the weight pack holding a
  4x4 block per tap as four contiguous vec4 indexed by input channel (`[Coutb][Cinb][KH][KW][4 ic]
  [4 oc]`), instead of four per-output-channel `dot()` reductions. On the primary device's shader
  compiler the horizontal reduction was the GEMM-class bottleneck: pointwise 64->256 @56x56
  0.144 -> 0.077 ms at unchanged fp32 accumulation (packed fp16 accumulation is slower still in
  this form, so `low` keeps fp32 accumulation). ResNet-50's 33 pointwise convs went 6.3 -> 4.3 ms.
- **Block group inside the row.** The tile-per-thread kernels decode the output-channel block
  group inside the row block (row-halo, compact stem) or inside a chunk of pixel tiles (pointwise,
  register-tiled), so a map's block groups run back to back and the input a later group re-reads
  is still in cache instead of streaming from DRAM once per group. The chunk is sized to the map
  (at most a workgroup of tiles, the map's tiles split evenly over its chunks): a 7x7 output at
  two pixels per thread has 25 tiles, and padding every block group to a 64-lane chunk would
  retire 39 of every 64 lanes idle - the 7x7 layers of ResNet-50 ran at that utilisation. The
  row-halo kernel additionally races a 2-D workgroup footprint (4, 2 or 1 output rows per
  workgroup), and on strided shapes a narrow two-pixel tile at two and four blocks per thread:
  a stride-2 row reads (WTILE-1)*2+3 columns per tile, so the 5-column narrow tile keeps twice
  the waves in flight of the 9-column default (16->32 s2 @360x480 0.338 -> 0.274 ms, 8->16 s2
  on the probe's map 0.387 -> 0.329, YOLOv8n's two stride-2 stem convs -0.15 ms; at stride 1 the narrow
  tile ties or loses and is not raced). Stride-2 3x3 convs on a shallow, huge-spatial net went 0.62 / 0.57 -> 0.42 / 0.38 ms,
  and a pointwise 64->64 @160x160 0.39 -> 0.14 ms.
- **Depthwise 2x2 output tile** (`dwconv_t2`) and **output-channel-sliced dispatch** for the
  register-tiled conv join the bit-neutral races (the tile carries a 4096-thread occupancy floor).

Measured, cooled interleaved A/B vs v1.4.0 (same protocol as above; sub-2.5 ms models by the
cached-tune protocol, which now isolates a per-binary cache file and cools before each tune pass —
a hot tune pass flips near-tie race picks and reads as a phantom several-percent delta):

| Model | primary device | second device |
|---|---|---|
| ShuffleNetV2 | **-21%** | **-11%** |
| SqueezeNet | **-5%** (kernel-parity `none` A/B) | **-12%** |
| MnasNet 1.0 | **-9%** | **-8%** |
| YOLOv8n | within the 3% gate | **-8%** |
| ResNet-50 | parity | **-4%** |
| Inception-v3 | within the 3% gate | **-3%** |
| MobileNetV2 / V3, EfficientNet-B0, DenseNet-121 | within the 3% gate | within the 3% gate |

The ShuffleNet win is the dispatch-count story (barrier elision + the shuffle fold on a net that
was 60-80% dispatch-floor-bound); the deltas that read "within the gate" are dominated by
tune-time pick variance on a heat-soaked device (branch-vs-branch control runs read parity), with
no reproducible regression on either device at any tuning level.

### Zero-copy Concat/Split/Slice sub-buffer views (v1.4.2)

Concat, Split, and contiguous unit-step Slice move bytes without computing anything, yet each cost
real GPU passes over the activation (one dispatch per concatenated part, one copy or gather per
split output). Wherever the slices tile the whole contiguously in the stored byte layout, the
planner now binds each slice as a sub-buffer VIEW into the whole's device memory: producers write
their slice of the concatenation in place, split/slice consumers read theirs in place, and the
node records nothing (ADR-0018). Always on, no knobs; a node that does real work at its stores (a
Concat carrying a fused pointwise unit — DenseNet's BN+ReLU-riding concats) or whose slices
interleave (a batch>1 channel concat, YOLO's axis-2 head concats) keeps the dispatching path,
decided per node. Outputs are byte-identical to the copying path on every model at every tuning
level — the transform touches addressing only, never arithmetic or rounding.

Per-op effect at `--tuning none` on the primary device (per-op-type profile, ms): ShuffleNetV2
Concat+Split 0.31 → 0.05, YOLOv8n Concat+Split+Slice 0.96 → 0.28, SqueezeNet Concat 0.13 → 0.01,
Inception-v3 Concat 0.20 → 0.02. Activation-pool peak is unchanged (views allocate no memory of
their own, and arenas reuse pool slots).

End-to-end, cooled interleaved min-of-5 vs main (same protocol as above):

| Model | primary, `none` | primary, `fast` | second, `fast` |
|---|---|---|---|
| ShuffleNetV2 | **-13.5%** | **-17.2%** | **-16.5%** |
| SqueezeNet | **-6.4%** | **-13.3%** | **-9.8%** |
| YOLOv8n | **-5.4%** | **-7.7%** | **-4.3%** |
| Inception-v3 | **-4.4%** | -2.4% | within the 3% gate |
| ResNet-50 / DenseNet-121 | parity (no eligible sites / epilogue-carrying concats) | within the 3% gate | within the 3% gate |

### Zero-copy extensions: movement-chain folding and the Flexible layout vote (v1.4.2)

Three further always-on, byte-exact reductions of data movement (same discipline as the section
above: structural rules, per-node fallback, outputs byte-identical to the copying path on every
model at every tuning level):

- **Identity-Transpose alias, leading-axis constant Pad write-through, and constant-contiguous
  Gather -> Slice folding.** An iota-perm Transpose joins the pure-copy alias set; a single-axis
  constant Pad becomes a producer write-through plus two transfer fills for the pad ranges; a
  Gather whose constant indices form a contiguous run rewrites to Slice at import and rides the
  slice-view machinery.
- **Movement-chain folding**: consecutive Transpose/Slice ops compose into ONE strided gather
  (`view_stride`/`view_base`, consumed by the flat_gather geometry and the CPU kernels alike), so
  a chain's intermediate tensor and its full round-trip disappear. A ViT-style attention block
  folds its K-slice+transpose chain; multi-consumer intermediates refuse.
- **Flexible layout vote**: a standalone fused-pointwise unit whose plan is expressible in both
  layouts (rank-4 run, no general-broadcast operand) is re-placed by element-weighted convert
  cost instead of the fixed classifier. On a movement-heavy warp graph (the lens zoom-morph
  class) this drops layout converts 44 -> 14 and the graph 16.5 -> 15.1 ms (-8.5%) at `none`,
  byte-identical; the CNN suite is unaffected (its layouts were already optimal).

### Depthwise + pointwise LDS fusion: byte-exact but a runtime regression (negative result)

Fusing a depthwise conv and its 1x1 projection into one kernel keeps the expanded intermediate in
LDS, so the block's largest activation never reaches global memory and one dispatch plus one
barrier disappear. On this hardware that trade loses, and by a wide margin.

Measured at `--precision low` against the unfused pair, cooled interleaved min-of-5, two devices,
under both the pinned gate config and the shipping defaults:

| Model | fused sites | device A | device B |
|---|---|---|---|
| MobileNetV2 | 17 | **+90%** | **+30%** |
| MnasNet1.0 | 13 | **+145%** | **+68%** |
| ShuffleNetV2 x1.0 | 19 | **+15%** | **+4%** |
| EfficientNet-B0 (no eligible pairs) | 0 | +3% | -2% (noise floor) |

Accuracy is unaffected: cosine, PSNR, and SNR are identical to the unfused pair on every model,
because the fused projection reproduces the pointwise split-K summation order exactly (see
`ops/pw_splitk_rule.h`) and the output is byte-identical.

The cost is parallelism, not memory traffic. The projection stage gives each work item one
(channel-block, tile pixel) pair, so a 64-thread workgroup runs only `Coutb * 4` items — 24 of 64
for a 24-channel project — and each item re-reads the whole weight row. The standalone
`conv1x1` kernel register-tiles WTILE pixels by OCB channel-blocks, reusing every weight `vec4`
across WTILE pixels, and on exactly these deep small-plane shapes it also takes the split-K path
for extra parallelism the fused kernel forfeits. Per-op, the first MobileNetV2 pair costs 0.477 ms
fused against 0.052 + 0.045 ms unfused. The fusion therefore stays opt-in (`-O2` /
`--fuse-dwpw`); shipping it by default would roughly double these models' runtime. A phase B that
register-tiles like `conv1x1` is the prerequisite for revisiting it.

## Full `run()` wall: the host side of a run (v1.5.x branch)

The GPU span is not what an application sees; `run()` also pays the input copy, the submit, the
fence wait and the output copy. Measured with the engine's own per-run stage lines (`--timing`) on
the reference phone under the cooled protocol, four host-side changes landed:

- **Pre-wake fence wait.** A blocking `vkWaitForFences` cost 0.3-1.2 ms more than polling on the
  mobile SoC (the core drops into a deep idle state and wakes late). The wait now sleeps until
  shortly before the previous submission's wall, polls to completion, and falls back to the
  blocking wait past a fixed budget (`vk_fence_wait_policy.h`).
- **One submit for the descriptor-cap chunks.** Chunks split only for `maxSubmitBindings` go out as
  one `vkQueueSubmit` with a barrier at each chunk tail; only watchdog and iteration splits keep
  their own fence. The 3-chunk DenseNet lost its two GPU-idle gaps (1.9 ms).
- **Pinned host memory.** `PinnedHostMemory` blocks (`IOTensor::pinned`, `Tensor::pinned` /
  `Tensor::toPinned`) bind to the GPU through `VK_EXT_external_memory_host`: the boundary convert
  reads the caller's input pages and writes the caller's output pages, so no copy runs on either
  side. A block the device cannot bind is copied like a payload. `vknn_zerocopy_bench` times host,
  pinned and dma-buf modes per run and checks the pinned outputs byte for byte.
- **Detection head.** The YOLOv8 DFL decode chain folds to one blocked `FusedDfl` kernel, and the
  head's spatial-axis concats stay blocked (head converts 8 -> 2, the decode 0.35 -> 0.04 ms).

Cooled per-run `run()` wall in host mode (pack + submit/GPU + unpack; min / median over 29 runs),
before and after this stretch:

| Model | before (min / med) | after (min / med) | GPU span |
|---|---|---|---|
| mobilenetv2 | 2.31 / 2.39 | 1.79 / 2.26 | 1.58 |
| mobilenetv3 | 2.39 / 2.64 | 2.06 / 2.49 | 1.81 |
| efficientnet_b0 | 3.53 / 3.67 | 3.53 / 3.74 | 2.78 |
| resnet50 | 8.32 / 8.63 | 8.09 / 8.60 | 7.61 |
| yolov8n | 8.84 / 11.40 | 8.70 / 10.60 | 7.03 |
| densenet121 | 12.71 / 13.72 | 11.14 / 11.80 | 10.64 |
| six-conv probe | 4.48 / 5.37 | 4.54 / 5.29 | 3.04 |

With pinned host memory (`vknn_zerocopy_bench`, 20 runs, same device): the six-conv probe
4.38 / 4.45 -> 3.42 / 3.81 ms (the dma-buf loop's 3.46 / 4.25), yolov8n 7.90 / 9.05 -> 7.86 / 7.98
(dma-buf 7.97 / 8.07); the pinned outputs are byte-identical to the host mode's.

Tried and rejected on the same device, with the numbers in the source: a pool-partitioned staged
memcpy (0.49 -> 0.78 ms on 11 MB; the copy is bandwidth-bound and the pool wakes on little cores),
and the squeeze-excite chain fusion (generalized to Sigmoid / SiLU / HardSwish and a pooled kernel,
0.025-0.23 ms per block against ~0.05 unfused: the two 1x1 convs on a [N,C,1,1] tensor already sit
at the dispatch floor; it stays opt-in at `-O2`).

### What remains is the platform's, measured

Two residues were chased to their floor with a standalone Vulkan probe on the reference phone
(the probe sources live outside the tree; the numbers are the point):

- **The per-run gap between GPU busy time and the fence (0.4-0.8 ms warm, up to 1.5 ms after a
  gap of 5 ms or more) is the GPU's clock governor and wake, not the engine.** The same
  128-dispatch command buffer takes 2.2 ms of GPU time when submitted 1 ms after the previous one
  and 8-10 ms after any gap of 5 ms or more; an empty submit-and-fence round trip is 0.15-0.2 ms and
  a single tiny dispatch 0.5 ms. Pre-submitting the command buffer while the caller is idle, gated
  on a timeline semaphore the host signals when the inputs are ready, hides nothing: the driver pays
  the latency when the job starts, not when it is queued (1.43 ms against 1.37 ms with 20 ms gaps;
  the parked job neither tripped the watchdog nor kept the GPU clocked, and its output was exact).
  A heartbeat on the same queue, from one to 256 workgroups every 2 or 5 ms, leaves the main
  buffer at its slow-clock time (10.3 ms). What the engine can do it does: `Config::power = High`
  keeps the GPU from the deeper power collapse across long gaps (mnv2 at 400 ms gaps 3.94 -> 2.81
  ms), the pre-wake fence wait removes the host's own wake-up, and a busy application holds the
  clock itself. Raising the clock across idle gaps is the platform's performance-hint API, outside
  the engine.
- **DenseNet's 58 per-channel affine units cost 0.98 ms of 11.8 and stay.** In the first two
  blocks they are bandwidth-bound passes of 14-60 us (40-50% of the 1x1 conv they feed); in the
  last two they sit at the 8 us dispatch floor. Every fused form loses by measurement: the 1x1
  conv's input-load prologue re-applies the affine once per output-channel group (16x with the
  raced tiles) and measured 1.8-2.2x slower; the split-K kernel re-reads each input element once
  per output block (32x); a per-pixel all-outputs form starves the device of parallelism 32x or
  multiplies the split-K partial traffic. The ceiling of any of them is the 0.98 ms itself, and
  none reaches it.

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
- `scripts/profile_on_device.sh <adb-serial> <model.vxm|.onnx> [inputs...] [-- extra run_io flags]`
  captures a full single-pass profile of one model on one device: GPU identity/capabilities, the
  per-op GPU profile table + GPU total, cold vs warm session and steady-state timing, fp16 vs fp32,
  high queue priority, and per-op-type CPU-fallback isolation. Compiles `.onnx` via the host
  `vknn_compile` first; runs zero-filled inputs when none are supplied; env `OUTDIR` / `REPEAT` /
  `BUILD_DIR`; writes per-run logs plus a `summary.txt`.

## Gates and scripts

The thermal A/B discipline and the byte gates below are committed as scripts, not just prose.

- **`scripts/ci_host.sh`** — the host-only "before you push" gate. Runs the host build,
  `vknn_tests`, the `--android` and `--docs` builds, the op-support self-consistency check
  (`tools/check_support_consistency.py`), the epilogue-sync and shader-contract checks
  (`tools/check_epi_sync.py`, `tools/check_shader_contracts.py`), a clang-format drift report
  (advisory; `--strict-format` to enforce), and the CPU determinism check. No device needed; exits non-zero on any hard failure.
  ```bash
  scripts/ci_host.sh                 # full host gate
  scripts/ci_host.sh --no-android    # skip the NDK build (host-logic-only change)
  ```

- **`scripts/check_determinism.sh`** — runs a fixed-shape model (`assets/mnasnet1_0.onnx`) through
  the CPU backend twice and asserts the two runs are byte-identical. A cheap steady-state / no-random
  check; skips cleanly when the (gitignored) model asset is absent. Folded into `ci_host.sh`.

- **`benchmark/scripts/gate_op.sh`** — the reusable per-op **device byte gate**. Given a probe ONNX
  (or an `op_test.py`-style `--builder`) it compiles fused vs `--no-fuse-pointwise`, runs both on
  device with a pinned config (`--tuning none`, forced `--winograd`), and asserts fused==unfused
  byte-identity with zero CPU fallback. (The autotuner's output-affecting kernel choices are now
  deterministic shape rules at every `--tuning` level, so `none`/`fast`/`heavy` are byte-identical;
  the pin is belt-and-suspenders.) Pass `--ref-binary <fresh-main vknn_run_io>`
  to also cross-compare against main for a no-regression verdict.
  ```bash
  benchmark/scripts/gate_op.sh --onnx probe.onnx --inputs "in0.bin" --device <device-serial>
  ```

- **`benchmark/scripts/gate_pw_probes.sh`** — the same byte gate over the fixed 22-probe
  epilogue-fusion suite (generated by `make_pw_probes.py`). **The pass/fail is branch-vs-ref
  byte-identity, never an absolute score:** the probe generator drifts across revisions, so a freshly
  generated suite can score ~22/45 at a healthy base with no regression. A meaningful run builds main
  fresh into a reference `vknn_run_io` and passes it as `--ref-binary`. `gate_op.sh` and
  `gate_pw_probes.sh` share `benchmark/scripts/gate_lib.sh` so both agree on what "byte gate" means.

- **`benchmark/scripts/dev_perfab.sh`** — the cooled interleaved perf **A/B** (the thermal protocol
  above, scripted). Takes two host `vknn_run_io` binaries, a model list, and a device serial; runs
  A,B,A,B... with a cooldown before each run, keeps the min submit wall over N iterations, and prints
  a per-model A/B delta table, flagging regressions beyond a threshold (default 3%). **A must be a
  FRESH-BUILT main**, never the drifted `a1/ref` on-device install.
  ```bash
  ./build.sh --android                              # build B (this branch)
  cp build-android/vknn_run_io /tmp/b_run_io
  git worktree add /tmp/main-ref main && (cd /tmp/main-ref && ./build.sh --android)
  cp /tmp/main-ref/build-android/vknn_run_io /tmp/a_run_io
  benchmark/scripts/dev_perfab.sh --a /tmp/a_run_io --b /tmp/b_run_io \
    --models models.txt --device <device-serial>
  ```

### Warm-start cache persistence (v1.5.1)

v1.5.1 changes when the warm-start cache reaches disk, not what any kernel computes. The cache was
written only from `~Session()`; a host that never reaches the destructor — an Android app killed
while a multi-GB model is resident — discarded the whole cold-load autotune sweep and repaid it on
the next load. `Session::flushNewCacheWork()` now writes at the end of every creation path, gated on
the weight cache being dirty or the driver's pipeline blob having grown, so a warm load with nothing
new costs one size query and no encode.

No kernel, layout, or scheduling decision changed, and the byte gate confirms it: the 3D-splat
encoder's eight outputs are md5-identical between a v1.5.1 and a v1.5.0 build on one device with the
same seeded inputs. Every measurement in this document therefore stands as recorded.

Measuring load time on a multi-GB model needs the same discipline as the perf A/B above, for a
different reason: page-cache state alone swings a warm encoder load between 2.3 s and 6.2 s on one
device. Arms compared across separate batches produce a confident-looking difference that reverses
when the same arms are interleaved inside one batch.

## Reproduce

```bash
./scripts/bench_vs_mnn.sh 20        # see the MNN SETUP block at the top of the script
```
