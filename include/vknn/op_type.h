// Operator types and the ONNX name <-> OpType mapping helpers.
#pragma once
#include <cstdint>
#include <string>

namespace vknn {

    /// Operator types. Add a new value here + a name mapping + register kernels.
    enum class OpType {
        Unknown = 0,
        Conv,          // Conv2D (incl. depthwise via group, pointwise 1x1)
        ConvTranspose, // transposed / fractionally-strided conv (deconv upsample)
        Clip,          // Clip / Relu6
        Relu,
        Add, // elementwise add (residual)
        GlobalAvgPool,
        AvgPool,
        MaxPool,
        Gemm,
        MatMul,
        Einsum, // einsum: outer-product (RoPE) + batched mat-vec/matmul (geometry tail)
        Reshape,
        Expand,  // broadcast X to a target shape (numpy broadcasting), flat gather
        Tile,    // repeat X along each dim by `repeats`, flat gather
        Squeeze, // remove size-1 dims (metadata reshape / flat copy)
        Flatten,
        Softmax,
        LayerNorm, // LayerNormalization: normalize over axes [axis..end], y=(x-mean)/sqrt(var+eps)*g+b
        BatchNorm,
        Concat,
        Pad,
        Identity,
        Constant,
        Shape,
        Gather,
        Unsqueeze,
        Unary,           // elementwise unary family (Sigmoid/Tanh/HardSwish/...), see UnaryType
        Binary,          // elementwise binary family (Mul/Sub/Div/Max/Min/Pow), see BinaryType
        PRelu,           // y = x>0 ? x : slope*x, slope per-channel
        Resize,          // Resize/Upsample (nearest/linear), spatial
        GridSample,      // sample input at grid coords (CPU)
        Transpose,       // permute dims (CPU)
        Slice,           // strided slice (CPU)
        Reduce,          // ReduceMean/Sum/Max/Min/Prod/L2, see ReduceType
        DepthToSpace,    // [N,C,H,W] -> [N,C/b^2,H*b,W*b], DCR|CRD (flat index remap)
        Cast,            // dtype cast (CPU)
        Split,           // split along an axis into N outputs (CPU)
        Where,           // cond ? X : Y, elementwise with full broadcasting (flat path)
        Equal,           // A == B -> 1.0/0.0, elementwise with broadcasting (flat path)
        Greater,         // A >  B -> 1.0/0.0, elementwise with broadcasting (flat path)
        GreaterEqual,    // A >= B -> 1.0/0.0, elementwise with broadcasting (flat path)
        ConstantOfShape, // emit a tensor of the given shape filled with a scalar value
        EyeLike,         // identity-like matrix (ones on a diagonal) matching the input shape
        ScatterND,       // copy data, then scatter update slices at N-D index rows
        FusedSE,         // fused Squeeze-Excite scale: GAP->FC->relu->FC->hardsigmoid (one kernel)
        FusedDwPw,       // fused depthwise-3x3 + 1x1-project (expanded intermediate stays on-chip)
        FusedPointwise, // fused per-element chain (standalone); also the epilogue carried by producers
        // layout conversion nodes (inserted by the layout pass)
        ConvertLayout,
        // fp16 <-> fp32 storage conversion at a selective-fp32 region frontier (inserted by markFp32)
        ConvertDtype,
    };

    // Fused-pointwise-chain limits (the pass splits chains that would exceed these).
    constexpr int kPwMaxSteps    = 8;  // steps per fused unit
    constexpr int kPwMaxOperands = 6;  // extra tensor operands per unit (primary excluded)
    constexpr int kPwMaxRank     = 4;  // flat broadcast rank stored in the plan (rank>4 => not flat-fused)

    const char *opTypeName(OpType t);
    OpType      opTypeFromOnnx(const std::string &s);

} // namespace vknn
