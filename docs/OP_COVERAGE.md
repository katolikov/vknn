# VKNN operator coverage

VKNN imports ONNX and lowers it to its IR. Each operator has a **CPU oracle** (the bit-exact
reference and automatic fallback) and, for most, a **Vulkan kernel**. CNN-shaped tensors default to
the NC4HW4 GPU layout (channels packed in vec4 blocks); transformer-shaped tensors (attention, RoPE,
geometry) use a **flat row-major GPU path** (rank up to 8), and the layout pass splices NC4HW4&harr;flat
converts in at the boundaries. Every operator is checked against onnxruntime (cosine ≥ 0.999 on the
GPU fp16 path, 1.0 on the CPU fp32 path).

Every operator lives in its own file under `src/backend/{cpu,vulkan}/ops/` (one op per file).

## Convolution & pooling

| Operator | GPU | CPU | Notes |
|---|---|---|---|
| Conv (group=1, depthwise, 1×1 pointwise) | ✅ | ✅ | NC4HW4; direct 3×3, split-K deep 1×1, fused activation + residual-Add + Relu in the epilogue |
| ConvTranspose | ✅ | ✅ | `auto_pad` + `output_shape` handled (shared `src/core/conv_geom.h` geometry) |
| GlobalAveragePool | ✅ | ✅ | one workgroup / channel-block, LDS tree-reduce |
| AvgPool / MaxPool | ✅ | ✅ | windowed |
| BatchNorm | ✅ | ✅ | per-channel affine; usually folded into Conv |

## Elementwise

| Operator | GPU | CPU | Notes |
|---|---|---|---|
| Unary family | ✅ | ✅ | Sigmoid, Tanh, HardSwish, HardSigmoid, LeakyRelu, Elu, Abs, Neg, Exp, Log, Sqrt, Floor, Ceil, Relu, SiLU, Erf, Cos, Sin, Reciprocal, Softplus, Round |
| Binary family | ✅ | ✅ | Mul, Sub, Div, Max, Min, Pow, Add — same-shape, channel-broadcast (SE), and general NumPy broadcast on the flat path |
| Relu / Relu6 / Clip | ✅ | ✅ | standalone, and fused into the producing Conv/Gemm |
| PRelu | ✅ | ✅ | per-channel slope |
| Where / Equal / Greater / GreaterEqual / Less / LessOrEqual | ✅ | ✅ | flat broadcast (fp32 + int64) |

## Transformer / attention

| Operator | GPU | CPU | Notes |
|---|---|---|---|
| MatMul | ✅ | ✅ | general batched N-D + broadcast (QKᵀ, AV, MLP) |
| Gemm / FC | ✅ | ✅ | M rows (per-row strides for the multi-view camera head) |
| Softmax | ✅ | ✅ | channel-axis (NC4HW4) and arbitrary last-axis (flat) |
| LayerNorm | ✅ | ✅ | reduction over the last axes, affine |
| Einsum | ✅ | ✅ | outer-product (RoPE) on GPU; batched mat-vec / matmul lowered to MatMul |
| Gather | ✅ | ✅ | axis-aware (attention Q/K/V split on axis 2), const or runtime index |

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
| Cast | ✅ | ✅ | float ↔ int32/int64 |
| Pad | ✅ | ✅ | constant / edge / reflect; GPU = flat row-major, static pads |
| Shape / Constant / ConstantOfShape / EyeLike | const-fold | ✅ | resolved at compile time |
| Range | ✅ | ✅ | small constant ranges const-fold; a float Range whose size resolves at plan time runs on the GPU (start/delta may be runtime scalars); int64 or unresolved-size ranges use the CPU op, which sizes at run time |
| Identity | — | ✅ | |

## Fusions and lowerings

The graph passes (and `vknn_compile`) apply the rewrites below. `vknn_compile` groups them behind
an optimization level (`-O0` = none/reference, `-O1` = the default production set, `-O2`/`-O3` =
+ the experimental SE and dwpw fusions); the individual `--[no-]fuse-*` / `--[no-]lower-conv`
flags override a single pass on top of the level:

- **General pointwise fusion** — the one fusion pass. It grows each maximal same-shape
  per-element region (Binary/Add/Unary/Clip/Relu/PRelu/Where/Greater/GreaterEqual/Less/LessEqual/Equal, fanout
  included) and emits it as a single fused unit: folded into the producing kernel's store epilogue
  (MatMul, Gemm, Conv family, ConvGemm, Softmax, LayerNorm, Reduce, GridSample, Resize,
  ConvTranspose, pooling, Transpose/Slice, Concat) or one standalone `FusedPointwise` kernel.
  Internal fanout rides the unit's registers; values consumed outside the region export as extra
  output streams. Residual Adds, swish diamonds (`x · sigmoid(x)`), MatMul bias-Adds, and lone
  activations are all cases of this pass; a lone Relu (or a Clip with fp16-representable bounds)
  after a Conv/Gemm folds onto the kernel's own `fusedAct` instead. By default the swish/residual/
  bias patterns use the kernels' fast fp32-accumulator epilogues (old-main speed; not byte-equal to
  unfused); `--strict-fuse` keeps every step rounded, making fused == unfused byte-identical — the
  byte-verification mode. Enabled at `-O1` (default); opt out with `--no-fuse-pointwise`.
- **BatchNorm lowering** — a BatchNorm the conv fold cannot absorb (pre-activation BN, BN after
  Concat) lowers unconditionally to a per-channel Mul+Add with host-folded scale/shift, which the
  pointwise fusion then merges into the neighboring kernels.
- **Conv → ConvGemm lowering** — a non-Winograd K×K Conv (strided, dilated, 5×5/7×7, 1×7/7×1,
  shallow 3×3) lowers to one LDS-tiled implicit-GEMM kernel with weights repacked `[K][Cout]` at
  convert time. Deterministic and fp16-floor equivalent to Conv (the fp32 accumulation order
  shifts, exactly as Winograd's does). Experimental and off by default — the current 64×64×16
  kernel loses to the direct conv on classifier-CNN shapes (small output areas starve its pixel
  tiles); opt in with `--lower-conv` and measure per model.
- **Squeeze-Excite** chain folds to one kernel (`-O2` or `--fuse-se`, experimental).
- **Depthwise + 1×1-project** folds to one kernel; the expanded intermediate stays on-chip (`-O2`
  or `--fuse-dwpw`, experimental; pairs wider than the kernel's 1024-channel LDS budget stay
  separate convs).
- **Einsum lowering** to MatMul/Squeeze/Unsqueeze; **ConvTranspose → Conv + DepthToSpace**
  (subpixel rewrite).

## Adding an operator

An operator requires: an `OpType` value appended at the END of the enum in `include/vknn/op_type.h`
(append-only — `.vxm` files store the raw integer, so a mid-enum insert corrupts existing models),
an ONNX name in `src/core/op.cpp`, a shape rule
in `src/import/infer_shapes.cpp` `inferShapes`, a CPU oracle in `src/backend/cpu/ops/`, and (when the layout
allows) a Vulkan op + GLSL shader gated by `Backend::supportsNode()`. See
[ADDING_AN_OPERATOR.md](ADDING_AN_OPERATOR.md) and [../skills/add-an-operator.md](../skills/add-an-operator.md).
