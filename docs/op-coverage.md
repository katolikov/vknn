# VKNN operator coverage

VKNN imports ONNX and lowers it to its IR. Each operator has a **CPU oracle** (the bit-exact
reference and automatic fallback) and, for every op that carries compute, a **Vulkan kernel**.
CNN-shaped tensors default to the NC4HW4
GPU layout (channels packed in vec4 blocks); transformer-shaped tensors (attention, RoPE, geometry)
use a **flat row-major GPU path** (any rank — per-axis kernel geometry rides a content-deduped SSBO, not the push constant), and the layout pass splices NC4HW4&harr;flat
converts in at the boundaries. Every operator is checked against onnxruntime (cosine ≥ 0.999 on the
GPU fp16 path, 1.0 on the CPU fp32 path).

**Every executable operator runs on the GPU.** Across the benchmarked model zoo (CNN, detection, and
the transformer encoder), the nodes that carry compute all land on Vulkan. The only operators the
engine does **not** run are data-dependent control flow — `Loop`, `If`, `NonMaxSuppression` (and
`Scan`) — which are not in the op table and do not plan on any backend, plus the shape-arithmetic
ops that the import passes const-fold away before planning (`Shape`, `Constant`, `EyeLike`), so no
runtime kernel is dispatched at all. A small number of ops have a GPU kernel but fall back to the CPU
oracle on a specific input class the kernel cannot represent (an int64→narrow-integer `Cast`, a
runtime-`k` `TopK`, an unresolved-shape `ConstantOfShape` / `Range`); these are called out per row
below.

To generate the exact per-node backend assignment for a given model — the ground truth this table
summarizes — run `vknn_compile model.onnx out.vxm --support-report report.json` (the report comes
from `vkSupportSurvey`, the same capability gate the device engine runs) and inspect it with
`tools/check_model_support.py`.

Every operator lives in its own file under `src/backend/{cpu,vulkan}/ops/` (one op per file).

## Convolution & pooling

| Operator | GPU | CPU | Notes |
|---|---|---|---|
| Conv (group=1, depthwise, 1×1 pointwise, general grouped via lowering) | ✅ | ✅ | NC4HW4; direct 3×3, split-K deep 1×1, fused activation + residual-Add + Relu in the epilogue; a general grouped Conv (1 < group < Cin) lowers at import (`lowerGroupedConv`) into group-1 Convs over channel slices + Concat and runs on the same kernels |
| ConvTranspose | ✅ | ✅ | `auto_pad` + `output_shape` handled (shared `src/core/conv_geom.h` geometry) |
| GlobalAveragePool | ✅ | ✅ | one workgroup / channel-block, LDS tree-reduce |
| AvgPool / MaxPool | ✅ | ✅ | windowed |
| BatchNorm | ✅ | ✅ | per-channel affine; usually folded into Conv |

## Elementwise

| Operator | GPU | CPU | Notes |
|---|---|---|---|
| Unary family | ✅ | ✅ | Sigmoid, Tanh, HardSwish, HardSigmoid, LeakyRelu, Elu, Abs, Neg, Exp, Log, Sqrt, Floor, Ceil, Relu, SiLU, Erf, Cos, Sin, Reciprocal, Softplus, Round, Sign, and an internal Trunc (created only by the float→int→float Cast fold `foldIntRoundtripCast`, not parsed from ONNX) |
| Det | ✅ | ✅ | Batched square-matrix determinant `[..., n, n] → [...]`; GPU covers n ≤ 4 by fixed-order cofactor expansion (bitwise-equal to the CPU oracle in fp32) and 5 ≤ n ≤ 8 by in-register partial-pivot LU (deterministic fixed order, incl. the permutation sign); only n > 8 — no known real model — takes the CPU double-precision LU via a named gate |
| Binary family | ✅ | ✅ | Mul, Sub, Div, Max, Min, Pow, Add — same-shape, channel-broadcast (SE), and general NumPy broadcast on the flat path |
| Relu / Relu6 / Clip | ✅ | ✅ | standalone, and fused into the producing Conv/Gemm |
| PRelu | ✅ | ✅ | per-channel slope |
| Where / Equal / Greater / GreaterEqual / Less / LessOrEqual | ✅ | ✅ | flat broadcast (fp32 + int64) |
| And / IsNaN | ✅ | ✅ | boolean AND with NumPy broadcast / elementwise NaN test — bool results as 1.0/0.0, own flat kernels (not pointwise-fusion members) |

## Transformer / attention

| Operator | GPU | CPU | Notes |
|---|---|---|---|
| MatMul | ✅ | ✅ | general batched N-D + broadcast (QKᵀ, AV, MLP) |
| Gemm / FC | ✅ | ✅ | M rows (per-row strides for the multi-view camera head) |
| Softmax | ✅ | ✅ | channel-axis (NC4HW4) and arbitrary last-axis (flat) |
| LayerNorm | ✅ | ✅ | reduction over the last axes, affine |
| RMSNorm | ✅ | ✅ | root-mean-square norm `y = x·rsqrt(mean(x², last axis) + ε)·γ`, fp32 sum-of-squares in one fused flat kernel; created by `lowerRMSNorm` from the primitive Pow/ReduceMean/Add/Sqrt/Mul chain or mapped from an ORT `SimplifiedLayerNormalization` |
| Einsum | ✅ | ✅ | outer-product (RoPE) on GPU; batched mat-vec / matmul lowered to MatMul |
| Gather | ✅ | ✅ | axis-aware (attention Q/K/V split on axis 2), const or runtime index |
| Rope | ✅ | ✅ | fused rotate-half rotary embedding: one kernel computing `x1·cos − x2·sin` / `x1·sin + x2·cos` per position, reading the cos/sin table row directly; created only by the load-time `fuseRope` pass (`Hint::RopeFusion`) from the primitive Slice/Gather/Mul/Concat chain — never parsed from ONNX, never serialized to a `.vxm` |
| FusedAttention | ✅ | ✅ | single-query (M=1) decode-attention core `softmax(q·Kᵀ·scale + mask)·V` in one kernel; operands read through per-axis operand-view strides so the GQA KV cache is read in place (no materialized repeat_kv); fp32 scores + softmax (numerically finer than the decomposed fp16 round-trips); created only by the load-time `fuseDecodeAttention` pass (`Hint::FusedAttention`) — never imported, never serialized |
| SimplifiedLayerNormalization / SkipSimplifiedLayerNormalization / SkipLayerNormalization / RotaryEmbedding / MultiHeadAttention / GroupQueryAttention | lowered | lowered | ORT contrib transformer ops (com.microsoft), expanded to primitive ops at import by `lowerOrtContribOps` (SimplifiedLayerNorm → RMSNorm; GroupQueryAttention → the rope/concat/repeat_kv/attention subgraph; MultiHeadAttention only in the pure q/k/v + additive-mask form); no backend kernel — a variant the expansion declines surfaces in the support report under its real name |

## Shape / data movement

| Operator | GPU | CPU | Notes |
|---|---|---|---|
| Reshape / Flatten / Squeeze / Unsqueeze | ✅ | ✅ | metadata + flat copy (rank-5 channel-shuffle handled) |
| Transpose / Slice | ✅ | ✅ | flat gather (generic N-D) |
| Concat | ✅ | ✅ | NC4HW4 channel-axis (4-aligned) and flat scatter |
| Split | ✅ | ✅ | 4-aligned channel split (block copy) + flat non-channel split |
| Expand / Tile | ✅ | ✅ | broadcast / repeat, flat gather |
| DepthToSpace | ✅ | ✅ | DCR / CRD (pixel-shuffle) |
| ScatterND | ✅ | ✅ | copy + scatter (runtime float index) |
| Resize / Upsample | ✅ | ✅ | nearest + bilinear, 4 coord modes |
| GridSample | ✅ | ✅ | bilinear/nearest/cubic; constant or runtime grid (optical-flow warps); under fp16 the grid coordinates are fp16-stored, which bounds sampling accuracy near discontinuities |
| Reduce (Mean/Sum/Max/Min/Prod/L2) | ✅ | ✅ | arbitrary axes |
| Cast | ✅ | ✅ | float ↔ float, int → float, and an int64 input to float/INT32/INT64/INT8/UINT8/BOOL on the GPU (the INT8/UINT8/BOOL narrowing matches the CPU op bit-for-bit); an int64 input to INT16/UINT16 or a 32/64-bit unsigned target keeps the exact CPU op |
| Pad | ✅ | ✅ | constant / edge / reflect; GPU = flat row-major, static pads (a runtime pad *value* runs on the GPU; a runtime pads *geometry* falls back to CPU) |
| Shape / Constant / EyeLike | const-fold / ✅ | ✅ | resolved at compile time (const-folded away on the GPU path) |
| ConstantOfShape | ✅ | ✅ | resolved output size fills on the GPU (int fill carried in compute float, repacked to the declared dtype on readback); an unresolved (data-dependent) output size keeps the CPU op |
| Range | ✅ | ✅ | resolved output size generates on the GPU (start/limit/delta may be runtime scalars; int ramps carried in compute float, repacked on readback); an unresolved-size range keeps the CPU op |
| Identity | — | ✅ | rewired to its producer at import (no runtime kernel needed) |
| TopK | ✅ | ✅ | k largest/smallest along an axis; values + int64 indices, ties break to the lower index; GPU flat path when k is a const int64 input (or the opset-9 `k` attribute) and the input shape resolves; a runtime k keeps the CPU op |
| Dropout | eliminated | eliminated | inference-mode identity (training_mode absent or constant false, mask output absent or unconsumed) removed at import, consumers rewired to the producer; a consumed mask or a non-constant training_mode is unsupported |
| InstanceNormalization | lowered | lowered | decomposed at import into spatial ReduceMean + Sub/Mul/Add/Sqrt/Div and a per-channel scale/bias Mul+Add, so it runs wherever those ops run (no dedicated kernel); needs fp32-initializer scale/B of length C and input rank ≥ 3, else the node stays opaque and unsupported |

## Quantization

A quantized ONNX checkpoint runs **dequantized to float** via the import-time dequantize pass
(`src/import/dequantize_graph.cpp`, default on; `--no-dequantize` disables). For static QDQ / QLinear
the pass drops the 8-bit rounding but **preserves the saturation clamp** each quant hop encodes (so a
ReLU folded into an activation quant range survives) — results match an int-exact runtime closely but
not bit-exactly. For dynamic quantization it matches the canonical
`DynamicQuantizeLinear → MatMulInteger / ConvInteger → Cast → Mul` cluster and folds it to a plain
float `MatMul` / `Conv` (weight to fp32, **no** output clamp — the integer matmul carries no output
quant range). A dynamic-quant cluster that does not match this shape stays intact and fails at
planning.

`vknn_compile -Os` additionally quantizes MatMul/Gemm/Conv weights to **int4** by default
(`--quant-bits 8|lut4` selects the int8 / 16-entry-codebook formats; calibration-free or
calibrated; AWQ-style activation-salient outlier columns kept fp16; a per-layer error guard keeps
hostile layers fp16), stored in a VXM5 container (VXM6 when any packed weight is int8/lut4).
MatMul weights execute on native packed GPU kernels for all three formats (a specialized GEMV for
M=1 decode and a tiled kernel for prefill); quantized Conv/Gemm weights are rebuilt to fp16 at
load (`materializeQuantWeights`). This is separate from the
QDQ / QLinear dequantize-at-import path documented above (which targets pre-quantized ONNX
checkpoints). An ORT-contrib pre-quantized `MatMulNBits` (4-bit) checkpoint imports directly to the
native int4 path.

| Operator | GPU | CPU | Notes |
|---|---|---|---|
| DequantizeLinear | lowered / ✅ | ✅ | over an initializer, folds the weight to fp32 `(W_q − zp)·scale` (per-tensor + per-axis); over an already-float edge inside a collapsed sandwich, drops; a genuine int-graph-boundary DQ runs the flat GPU kernel when scale/zero_point are constant initializers (per-axis decoded via one inner-stride scalar, any rank); a runtime scale/zp falls back to the exact CPU op |
| QuantizeLinear | lowered / ✅ | ✅ | an activation Q→DQ sandwich collapses to a `Clip` over the quant range `[(qmin−zp)·scale, (qmax−zp)·scale]`; a graph-boundary Q (`saturate(round_half_even(x/scale) + zp)`) runs the flat GPU kernel when scale/zero_point are constant initializers; a runtime scale/zp falls back to the exact CPU op |
| QLinearConv | lowered | — | → Conv + `Clip` to the output quant range; weights fold fp32, int32 bias rescales by `x_s·w_s` |
| QLinearMatMul | lowered | — | → MatMul + output-range `Clip` |
| QGemm | lowered | — | → Gemm + output-range `Clip` (com.microsoft) |
| QLinearAdd | lowered | — | → Add + output-range `Clip`; a quantized-initializer operand dequantizes with its own scale/zp (com.microsoft) |
| QLinearGlobalAveragePool | lowered | — | → GlobalAveragePool + output-range `Clip` (com.microsoft) |
| DynamicQuantizeLinear | lowered | — | erased when it feeds a canonical `MatMulInteger`/`ConvInteger` cluster (below); a cluster that does not match the pattern fails at planning |
| MatMulInteger | lowered | — | → float `MatMul` with the weight folded fp32 when in a canonical `DynamicQuantizeLinear → … → Cast → Mul` cluster (no output clamp); an unmatched cluster fails at planning |
| ConvInteger | lowered | — | → float `Conv` with the weight folded fp32 in the same canonical cluster (no output clamp); an unmatched cluster fails at planning |

## Fusions and lowerings

The graph passes (and `vknn_compile`) apply the rewrites below. `vknn_compile` groups them behind
an optimization level (`-O0` = none/reference, `-O1` = the default production set, `-O2`/`-O3` =
+ the experimental SE fusion); the individual `--[no-]fuse-*` / `--[no-]lower-conv`
flags override a single pass on top of the level:

- **General pointwise fusion** — the one fusion pass. It grows each maximal same-shape
  per-element region (Binary/Add/Unary/Clip/Relu/PRelu/Where/Greater/GreaterEqual/Less/LessEqual/Equal, fanout
  included) and emits it as a single fused unit: folded into the producing kernel's store epilogue
  (MatMul, Gemm, Conv family, ConvGemm, Softmax, LayerNorm, RMSNorm, Reduce, GridSample, Resize,
  ConvTranspose, pooling, Transpose/Slice, Concat) or one standalone `FusedPointwise` kernel.
  Internal fanout rides the unit's registers; values consumed outside the region export as extra
  output streams. Residual Adds, swish diamonds (`x · sigmoid(x)`), MatMul bias-Adds, and lone
  activations are all cases of this pass; a lone Relu (or a Clip with fp16-representable bounds)
  after a Conv/Gemm folds onto the kernel's own `fusedAct` instead. By default the swish/residual/
  bias patterns use the kernels' fast fp32-accumulator epilogues (old-main speed; not byte-equal to
  unfused); `--strict-fuse` keeps every step rounded, making fused == unfused byte-identical — the
  byte-verification mode. Enabled at `-O1` (default); opt out with `--no-fuse-pointwise`.
- **GridSample warp fusion** — folds a scaled-flow + base-grid coordinate chain into the
  GridSample itself, which then computes each sample coordinate `base + scale·flow` inside the
  sampler (warp form: 4D NC4HW4 data + an NCHW flow `[N,2,Hout,Wout]` + the constant base grid as
  extra inputs; the base grid uploads fp32, and the fp16 variant reproduces the standalone Mul's
  fp16 store, keeping the fusion bit-exact with the materialized-grid path). Enabled at `-O1`
  (default); opt out with `--no-fuse-gridsample-warp`.
- **BatchNorm lowering** — a BatchNorm the conv fold cannot absorb (pre-activation BN, BN after
  Concat) lowers unconditionally to a per-channel Mul+Add with host-folded scale/shift, which the
  pointwise fusion then merges into the neighboring kernels.
- **InstanceNorm lowering** — InstanceNormalization decomposes unconditionally into spatial
  ReduceMean (rank-4 recovers as GlobalAveragePool), centered Sub, squared-diff Mul, epsilon Add,
  Sqrt, Div and a per-channel scale/bias Mul+Add ([1,C,1,..] initializers, the BatchNorm-lowering
  broadcast class); the pointwise fusion then merges the elementwise tail. scale/B must be fp32
  initializers of length C and the input rank ≥ 3 ([N,C,spatial...]); anything else keeps the
  opaque op, which no backend implements.
- **Conv → ConvGemm lowering** — a non-Winograd K×K Conv (strided, dilated, 5×5/7×7, 1×7/7×1,
  shallow 3×3) lowers to one LDS-tiled implicit-GEMM kernel with weights repacked `[K][Cout]` at
  convert time. Deterministic and fp16-floor equivalent to Conv (the fp32 accumulation order
  shifts, exactly as Winograd's does). Experimental and off by default — the current 64×64×16
  kernel loses to the direct conv on classifier-CNN shapes (small output areas starve its pixel
  tiles); opt in with `--lower-conv` and measure per model.
- **Squeeze-Excite** chain folds to one kernel (`-O2` or `--fuse-se`, experimental).
- **Depthwise + 1×1-project** folds to one kernel; the expanded intermediate stays on-chip,
  fp16-rounded exactly like the unfused store, so fused output is byte-identical to the unfused
  graph (default on at `-O1`; `--no-fuse-dwpw` opts out; pairs wider than the kernels'
  1152-channel LDS budget stay separate convs).
- **Einsum lowering** to MatMul/Squeeze/Unsqueeze; **ConvTranspose → Conv + DepthToSpace**
  (subpixel rewrite).

### Load-time LLM-decode fusions

These passes run at session load, each gated by its hint, and **never change the compiled `.vxm`**
(an old model speeds up on load). Each has a `--no-*` flag on the runners.

- **MatMul operand-view fold** (`Hint::MatMulViewFold`) — folds Expand/Transpose "repeat_kv" and
  attention-transpose chains into per-axis stride attributes on the consuming MatMul, so a GQA decode
  reads its KV cache through strides instead of materializing the broadcast. Bit-identical; honored by
  both backends.
- **RoPE chain fusion** (`Hint::RopeFusion`) — collapses each rotate-half chain (last-axis half
  Slices, cos/sin table Gathers, the rotate products, Concat) into one `Rope` node — ~7 dispatches
  per site → 1.
- **Decode-attention fusion** (`Hint::FusedAttention`) — collapses the M=1
  MatMul→scale/mask→Softmax→MatMul(→Transpose→Reshape) chain into one `FusedAttention` node,
  consuming the operand-view strides the fold above composed. Numerics-changing (fp32 scores /
  softmax), so it has its own cache-variant key.
- **KV-cache Concat fold** (`Hint::KvConcatFold`) — folds the per-token past‖new KV Concat
  into split-source `FusedAttention` reads and rewrites the present outputs to the rows-only
  convention, removing a whole-cache copy per token. Bit-identical. The engine-resident KV link and
  the decode drivers read the fold source from the present output's own shape (`io_link.h`), so the
  rows-only present drives the linked cache exactly like the cache-concat present.

## Adding an operator

An operator requires: an `OpType` value appended at the END of the enum in `include/vknn/op_type.h`
(append-only — `.vxm` files store the raw integer, so a mid-enum insert corrupts existing models),
an ONNX name in `src/core/op.cpp` (`opTypeName` + `opTypeFromOnnx`), a shape rule in
`src/import/infer_shapes.cpp` `inferShapes` (if the op changes shape), and a CPU oracle in
`src/backend/cpu/ops/`. A GPU kernel additionally needs a Vulkan op + GLSL shader in
`src/backend/vulkan/ops/` + `shaders/`, a capability gate row in `src/core/vk_gates.cpp`
(`vkKernelDeclared` + `vkNodeGate` — the shape/attribute gate the device and `--support-report`
share), and a row in the OpDescriptor table (`src/core/op_descriptor.cpp`: `LayoutClass` Flat/Nc4/ShapeDependent plus the `pwMember`/`pwEpilogue` fusion flags) so the layout pass marks it flat; a shape-dependent layout additionally needs a case in `gpuFlatNode` (`src/import/insert_layout_converts.cpp`). See [adding-an-operator.md](adding-an-operator.md) and
[../skills/add-an-operator.md](../skills/add-an-operator.md).
