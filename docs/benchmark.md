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
**slower** on this GPU (~11.5 vs F(2,3)'s 10.5 ms): the 6×6 transforms hold `d[6][6]`+`t[6][6]` = 72
`vec4` per thread (register pressure) and the GEMM has 4× fewer tiles (less parallelism). The traffic
saving is real but the register-heavy transforms can eat it. The output tile is picked per shape by
a deterministic cost model (F(4,3) wins on deep channels, F(2,3)'s smaller transform wins on
shallow); `setHint(Hint::WinogradUnit, 4)` forces F(4,3) on every 3×3, bypassing even the
Winograd-vs-direct shape rule.

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
  takes Winograd when the output map keeps the GEMM fed (`OHW >= 400`; probe-calibrated:
  256x256 @ 20x20 -38%, 192x192 @ 35x35 -42%, 512x512 @ 28x28 -69%, while the tile-starved
  256x256 @ 14x14 stays direct). The rule is inert on this suite (no model sits in the admitted
  region) and unlocks the class for larger models.
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
