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
runtime-`k` `TopK`, an unresolved-shape `ConstantOfShape` / `Range`, an int64 `Div` or int64-base
`Pow`, an `ArgMax` / `ArgMin` past the kernel's addressing or exact-index range); these are called out
per row below.

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
| Binary family | ✅ | ✅ | Mul, Sub, Div, Max, Min, Pow, Add — same-shape, channel-broadcast (SE), and general NumPy broadcast on the flat path. On int64 operands the CPU op is exact two's-complement arithmetic: Add/Sub/Mul wrap modulo 2^64, Div truncates toward zero (x / 0 = 0, INT64_MIN / −1 = INT64_MIN), Pow on an int64 base is an integer power (a negative exponent gives 1 for base 1, ±1 for base −1, else 0; a fractional float exponent takes the fp64 power truncated to int64). The GPU kernels divide and raise in float, so a Div with an Int64-typed operand and a Pow with an Int64-typed base keep the CPU op (see [Integer values](#integer-values)); a float base raised to an int64 exponent stays on the GPU |
| Sum / Mean / variadic Max, Min | lowered | lowered | `Sum` imports as Add and `Max` / `Min` as Binary; one of them with an operand count other than 2, and every `Mean`, is rewritten at import by `lowerVariadicElementwise` (before shape inference): 1 operand → Identity, N ≥ 2 → a left fold `op(op(x0, x1), x2)…` of 2-input nodes whose last step writes the original output (pairwise NumPy broadcasting). `Mean` is the Add fold followed by one Mul by a rank-0 fp32 initializer holding `1.0f / N` — ONNX Runtime's CPU Mean, bit-identical to it. 0 operands is an import error. `OpType::Mean` has no kernel; a `.vxm` whose graph still carries an unlowered variadic node fails to load with a recompile message |
| Relu / Relu6 / Clip | ✅ | ✅ | standalone, and fused into the producing Conv/Gemm |
| PRelu | ✅ | ✅ | per-channel slope |
| Where / Equal / Greater / GreaterEqual / Less / LessOrEqual | ✅ | ✅ | flat broadcast (fp32 + int64) |
| And / Or / Xor / Not / IsNaN | ✅ | ✅ | boolean AND / OR / XOR with NumPy broadcast (a 0 extent broadcasts to 0), elementwise NOT and NaN test — an operand is true iff it is nonzero (NaN, ±inf and subnormals are true, +0 and −0 false, int64 read exactly); bool results as 1.0/0.0 at the node's normal precision, own flat kernels (not pointwise-fusion members). The Or/Xor/Not shaders test the IEEE bit pattern rather than a float compare, so NaN reads true on every driver. A nonzero value too small for fp16 storage is already 0 in an fp16 tensor and reads false there; `Precision::High` keeps it |
| Mod | ✅ | ✅ | NumPy broadcast. `fmod` 0 (default) is the floor remainder (the divisor's sign), `fmod` 1 the C `fmod` (the dividend's sign). Float operands: the GPU computes `std::fmod` exactly by binary long division (bit-identical to the CPU, never `a − b·trunc(a/b)`); a NaN operand or an infinite dividend gives the canonical quiet NaN, under `fmod` 1 a finite dividend over an infinite divisor is the dividend, a zero divisor gives NaN under `fmod` 1 and +0 under `fmod` 0. Integer operands — resolved from the graph by `modOperandsAreInteger`: an Int64/Int32/Int8/UInt8-typed operand or result, or an operand written by a Cast to an integer type, Shape, ArgMax/ArgMin, TopK indices, an integer ConstantOfShape or a bitwise op, through value-preserving hops — take the exact int64 remainder on both backends: x mod 0 = 0 in both modes, INT64_MIN mod −1 = 0. `fmod` 0 and integer Mods run on fp32-pinned storage; `fmod` 1 on floats runs at the node's precision |
| BitShift | ✅ | ✅ | NumPy broadcast; `direction` is exactly `LEFT` or `RIGHT` (any other spelling keeps the CPU op, whose run fails with InvalidArgument). At the operand width `int_bits`: a shift count outside [0, `int_bits`) gives 0; the operand's low `int_bits` bits shift logically, LEFT keeping the low `int_bits` bits (uint8 200 << 1 = 144). Integer semantics per [Integer values](#integer-values) |
| BitwiseAnd / BitwiseOr / BitwiseXor | ✅ | ✅ | NumPy broadcast; two's-complement int64 `&` / `\|` / `^` on the CPU. The GPU kernel applies the operator in int32 (operands clamped to the int32 range, NaN reads 0), equal to the CPU result for operands in the int32 range |
| BitwiseNot | ✅ | ✅ | same shape; a signed type (`int_signed` 1) or a 64-bit width is `~v`, an unsigned type narrower than 64 bits keeps the low `int_bits` bits of `~v` (uint8 `~0` = 255) |

### Integer values

The CPU oracle carries INT64 tensors — and UINT32 / UINT64 initializers, a UINT64 at or above 2^63
keeping its bit pattern — as int64 host storage and computes the integer ops above exactly in int64.
Every other integer type rides fp32 lanes on the CPU too: INT32 / INT16 / UINT16 / BOOL initializers
materialize as fp32, INT8 / UINT8 initializers keep 1-byte lanes that the session pool widens to fp32,
and INT32 / INT8 / UINT8 graph inputs bind as fp32. An op with an Int64 operand stores an int64 result;
an fp32 lane read as an integer truncates toward zero, NaN reading 0 and out-of-range values saturating
to the int64 range.

The GPU stores every tensor as fp16 or fp32 float lanes, so an integer value is exact only where the lane
holds it: every integer within **±2^24** at fp32 (±2^11 at fp16, which saturates at 65504). The load-time
pass `pinIntegerResultsFp32` pins integer regions to fp32 storage at every precision tier: the outputs of
ArgMax, ArgMin, BitShift, BitwiseAnd/Or/Xor/Not and a Mod with `fmod` 0 or integer operands, the runtime
operands of those Mod and bitwise nodes, integer ArgMax/ArgMin/TopK data, every integer arithmetic node
with its operands (Add, Binary Add/Sub/Mul/Div/Max/Min, Pow of an integer base with its exponent,
ReduceSum/Max/Min/Prod, Range, Clip, Neg, Abs), the operands and 0/1 result of an Equal / Greater /
GreaterEqual / Less / LessEqual comparing integers, and the value-preserving region around each (layout
converts, Identity, metadata reshapes, Cast to an integer type, Where's values, and the movement and
selection ops Slice, Transpose, Expand, Tile, Split, Gather, Pad, DepthToSpace, ChannelShuffle,
ScatterND, TopK's values and Concat), so an integer graph input packs at fp32 and an integer result
reaches a graph output or the next integer op without an fp16 narrowing. A node reading a constant
operand through the segment's fp16-filled activation buffer (an NC4HW4 channel Concat with a constant
part) stays at the segment's precision. Past ±2^24 a GPU result rounds while the CPU op stays
exact. Two int64 forms keep the CPU op on a GPU plan instead of computing in float: a Binary `Div` with an
Int64-typed operand and a `Pow` with an Int64-typed base (support-report reasons `Binary: integer Div on
an int64 operand` / `Binary: integer Pow on an int64 base`). Both refusals read the tensor's recorded
dtype, so an int64 intermediate the importer did not type stays on the float kernel.

BitShift and BitwiseNot results depend on the ONNX element width, which the IR dtype does not record, so
the ONNX importer stamps two VKNN attributes on those nodes: `int_bits` (8 / 16 / 32 / 64) and
`int_signed` (0 / 1). They come from the element type of input 0 (BitShift falls back to input 1): a
declared type (graph input or output, `value_info`, initializer, Constant) wins; otherwise the producer
decides (Cast → `to`; Shape / Size / ArgMax / ArgMin / NonZero / TopK indices → INT64; the comparison and
boolean ops → BOOL, read as 8-bit unsigned; any other op → the type of its type-carrying input). An
unresolved type leaves both absent — 64-bit signed — and logs one warning naming the node. The Vulkan
gate refuses an invalid width by name (`BitwiseNot: int_bits must be 8, 16, 32 or 64`); the CPU op
fails the run with InvalidArgument.

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
| Cast | ✅ | ✅ | float ↔ float, int → float, and an int64 input to float/INT32/INT64/INT8/UINT8/BOOL on the GPU (the INT8/UINT8/BOOL narrowing matches the CPU op bit-for-bit); an int64 input to INT16/UINT16 or a 32/64-bit unsigned target keeps the exact CPU op. Cast to BOOL is a truth test on both backends, not a truncation: 1 for any nonzero value (negative, fractional, ±inf, NaN, subnormal), 0 for +0 and −0 |
| Pad | ✅ | ✅ | constant / edge / reflect; GPU = flat row-major, static pads (a runtime pad *value* runs on the GPU; a runtime pads *geometry* falls back to CPU) |
| Shape / Constant / EyeLike | const-fold / ✅ | ✅ | resolved at compile time (const-folded away on the GPU path) |
| ConstantOfShape | ✅ | ✅ | resolved output size fills on the GPU (int fill carried in compute float, repacked to the declared dtype on readback); an unresolved (data-dependent) output size keeps the CPU op |
| Range | ✅ | ✅ | resolved output size generates on the GPU (start/limit/delta may be runtime scalars; int ramps carried in compute float, repacked on readback); an unresolved-size range keeps the CPU op |
| Identity | — | ✅ | rewired to its producer at import (no runtime kernel needed); an Identity that copies a graph input, an initializer or another graph output onto a graph output stays as that output's copy and runs on the CPU op |
| TopK | ✅ | ✅ | k largest/smallest along an axis; values + int64 indices, ties break to the lower index; GPU flat path when k is a const int64 input (or the opset-9 `k` attribute) and the input shape resolves; a runtime k keeps the CPU op |
| ArgMax / ArgMin | ✅ | ✅ | int64 index of the largest / smallest element along `axis` (default 0; negative counts from the end), `keepdims` (default 1; a rank-1 input with `keepdims` 0 yields `{1}`), `select_last_index` (default 0: a tie keeps the first index; 1: the last). A sequential scan with ONNX Runtime's NaN behaviour: a NaN at index 0 is selected, a NaN anywhere else is never taken (an all-NaN slice yields 0); −0.0 and +0.0 tie; int64 data compares exactly on the CPU. GPU: one flat kernel (`arg_extreme.comp`, op and tie policy as specialization constants) writing fp32-pinned indices and reading the data at its own storage precision — under fp16 storage, values distinct in fp32 can tie and values past 65504 saturate (`Precision::High` keeps the fp32 answer); Int32/Int64-typed data is pinned fp32 and compares exactly within ±2^24. The gate keeps the CPU op for an unresolved or rank-0 input, an out-of-range axis, a zero-extent axis, more than INT32_MAX data elements, or an axis longer than 2^24 + 1 (indices past the exact fp32 range) |
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
+ the experimental SE and dwpw fusions); the individual `--[no-]fuse-*` / `--[no-]lower-conv`
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
  fp16-rounded like the unfused store. Byte-identical to the unfused graph on the CPU oracle and
  on the fp32 GPU path; the fp16 GPU path still diverges, so the pass stays experimental
  (`-O2` or `--fuse-dwpw`; pairs wider than the kernels' 1152-channel LDS budget stay separate
  convs).
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
share), and a row in the OpDescriptor table (`src/core/op_descriptor.cpp`: `LayoutClass` Flat/Nc4/ShapeDependent plus the `pwMember`/`pwEpilogue` fusion flags, with `kMaxOp` raised to the new last enumerator) so the layout pass marks it flat; a shape-dependent layout additionally needs a case in `gpuFlatNode` (`src/import/insert_layout_converts.cpp`). An op whose GPU result holds integers that must stay exact seeds `pinIntegerResultsFp32` (`src/import/mark_fp32.cpp`). See [adding-an-operator.md](adding-an-operator.md) and
[../skills/add-an-operator.md](../skills/add-an-operator.md).
