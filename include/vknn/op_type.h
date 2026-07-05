// Operator types and the ONNX name <-> OpType mapping helpers.
#pragma once
#include <cstdint>
#include <string>

namespace vknn {

    /// Operator types. Add a new value here + a name mapping + register kernels.
    /// APPEND-ONLY: model_io serializes these as raw integers, so a value inserted
    /// mid-enum shifts every later op and silently corrupts existing .vxm files.
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
        Range, // arange(start, limit, delta) -- scalar inputs, 1-D output
    };

    /// Fused-pointwise-chain limits. The fusion pass splits any chain that would exceed one of
    /// these; the shader plan layout (pw_plan.h) is sized from the same constants, so they are a
    /// shared contract between the importer and the kernel and cannot be changed independently.
    constexpr int kPwMaxSteps    = 8;  ///< Elementwise steps per fused unit.
    constexpr int kPwMaxOperands = 6;  ///< Extra tensor operands per unit (the primary input is excluded).
    constexpr int kPwMaxRank     = 4;  ///< Flat broadcast rank stored in the plan; rank>4 is not flat-fused.

    /// Stable ONNX-style spelling of an OpType (e.g. OpType::GlobalAvgPool -> "GlobalAveragePool").
    /// @returns A static, null-terminated string owned by the library; never null. An unrecognized
    ///          value maps to "Unknown".
    const char *opTypeName(OpType t);
    /// Map an ONNX operator type name to its OpType. Several ONNX ops collapse onto one OpType
    /// (e.g. every Reduce* variant -> OpType::Reduce; the Unary/Binary elementwise families -> a
    /// single OpType with the specific op recovered separately).
    /// @param s ONNX op_type string (case-sensitive).
    /// @returns The matching OpType, or OpType::Unknown when `s` names no supported operator.
    OpType      opTypeFromOnnx(const std::string &s);

} // namespace vknn
