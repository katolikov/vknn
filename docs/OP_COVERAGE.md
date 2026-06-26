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
| GlobalAveragePool | ✅ | ✅ | one workgroup / channel-block, LDS tree-reduce |
| AvgPool / MaxPool | ✅ | ✅ | windowed |
| BatchNorm | ✅ | ✅ | per-channel affine; usually folded into Conv |

## Elementwise

| Operator | GPU | CPU | Notes |
|---|---|---|---|
| Unary family | ✅ | ✅ | Sigmoid, Tanh, HardSwish, HardSigmoid, LeakyRelu, Elu, Abs, Neg, Exp, Log, Sqrt, Floor, Ceil, Relu, SiLU, Erf, Cos, Sin, Reciprocal, Softplus |
| Binary family | ✅ | ✅ | Mul, Sub, Div, Max, Min, Pow, Add — same-shape, channel-broadcast (SE), and general NumPy broadcast on the flat path |
| Relu / Relu6 / Clip | ✅ | ✅ | standalone, and fused into the producing Conv/Gemm |
| PRelu | ✅ | ✅ | per-channel slope |
| Where / Equal | ✅ | ✅ | flat broadcast (fp32 + int64) |

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
| GridSample | ✅* | ✅ | GPU for a constant grid; runtime grid on CPU |
| Reduce (Mean/Sum/Max/Min/Prod/L2) | ✅ | ✅ | arbitrary axes |
| Cast | ✅ | ✅ | float ↔ int32/int64 |
| Pad | — | ✅ | constant / edge / reflect |
| Shape / Constant / ConstantOfShape / EyeLike | const-fold | ✅ | resolved at compile time |
| Identity | — | ✅ | |

## Fusions

The graph passes (and `vknn_compile`) apply the following fusions:

- **Activation + residual-Add + Relu** folds into the Conv/Gemm epilogue. Enabled by default.
- **Swish / SiLU** (`x · sigmoid(x)`) folds into the producing Conv. Enabled by default; opt out with `--no-fuse-swish`.
- **Squeeze-Excite** chain folds to one kernel (`--fuse-se`, experimental).
- **Depthwise-3×3 + 1×1-project** folds to one kernel; the expanded intermediate stays on-chip (`--fuse-dwpw`, experimental).
- **Einsum lowering** to MatMul/Transpose/Unsqueeze.

## Adding an operator

An operator requires: an enum in `include/vknn/op.h`, an ONNX name in `src/core/op.cpp`, a shape rule
in `src/import/passes.cpp` `inferShapes`, a CPU oracle in `src/backend/cpu/ops/`, and (when the layout
allows) a Vulkan op + GLSL shader gated by `Backend::supportsNode()`. See
[ADDING_AN_OPERATOR.md](ADDING_AN_OPERATOR.md) and [../skills/add-an-operator.md](../skills/add-an-operator.md).
