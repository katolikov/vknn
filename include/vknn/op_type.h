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
        Range,     // arange(start, limit, delta) -- scalar inputs, 1-D output
        ConvGemm,  // Conv lowered to an implicit-GEMM kernel (lowerConv); weights repacked [K][Cout]
        Less,      // A <  B -> 1.0/0.0, elementwise with broadcasting (flat path)
        LessEqual, // A <= B -> 1.0/0.0, elementwise with broadcasting (flat path)
        Dropout,   // identity in inference mode; eliminated at import (eliminateDropout) -- a kept
                   // training-mode / consumed-mask Dropout has no kernel and is unsupported
        TopK,      // k largest/smallest along an axis -> (values, int64 indices); const k (CPU)
        InstanceNorm, // InstanceNormalization: per-channel normalize over the spatial dims; lowered
                      // at import to Reduce/Sub/Mul/Add/Sqrt/Div (lowerInstanceNorm) -- no kernel
        // ONNX quantized operator family (QDQ, QLinear, dynamic quantization). Recognized at
        // import so quantized checkpoints are named precisely; the com.microsoft members (QGemm,
        // QLinearAdd, QLinearGlobalAveragePool) match by op name because the wire parser drops
        // NodeProto.domain. No shape rule and no kernel in either backend: the import-time
        // dequantize lowering is the only execution path, and a node it leaves behind is
        // unsupported at plan (opTypeIsQuantized() tests family membership).
        QuantizeLinear,           // fp -> int8/uint8: saturate(round(x / scale) + zero_point)
        DequantizeLinear,         // int8/uint8/int32 -> fp: (q - zero_point) * scale, per-tensor or per-axis
        DynamicQuantizeLinear,    // derives scale/zp from the input range -> (x_q u8, x_scale, x_zp)
        QLinearConv,              // Conv on int8 x/w with per-input scale/zp + output requant, int32 bias
        QLinearMatMul,            // MatMul on int8 a/b with per-input scale/zp + output requant
        QLinearAdd,               // com.microsoft: Add on int8 a/b with per-input scale/zp + output requant
        QLinearGlobalAveragePool, // com.microsoft: GlobalAveragePool on int8 x with x/y scale/zp
        MatMulInteger,            // int8 x int8 -> int32 MatMul, optional zero points
        ConvInteger,              // int8 x int8 -> int32 Conv, optional zero points
        QGemm,                    // com.microsoft: Gemm on int8 a/b with scales/zps, int32 bias
        IsNaN,                    // elementwise NaN test: float -> bool (1.0/0.0), same shape (flat path)
        And,                      // elementwise boolean AND with NumPy broadcasting -> 1.0/0.0 (flat path)
    };

    /// Fused-pointwise limits. The fusion pass splits any unit that would exceed one of these;
    /// the shader plan layout (pw_plan.h) is sized from the same constants, so they are a shared
    /// contract between the importer and the kernel and cannot be changed independently.
    constexpr int kPwMaxSteps    = 16; ///< Elementwise steps per fused unit.
    constexpr int kPwMaxOperands = 6;  ///< Extra tensor operands per unit (the primary input is excluded).
    constexpr int kPwMaxRank     = 4;  ///< Flat broadcast rank stored in the plan; rank>4 is not flat-fused.
    constexpr int kPwMaxRegs     = 4;  ///< Named registers for step values reused by later steps.
    constexpr int kPwMaxOuts     = 4;  ///< Extra output streams (fanout values exported from the unit).

    /// pw_steps value references (the srcA/srcB/srcC/dst fields of a step). A source names the
    /// accumulator, the entry value, a register, or a tensor operand; a dst is kPwRefNone or a
    /// register encoded the same way (every step's result always lands in the accumulator, a
    /// register dst additionally keeps a copy for later steps).
    constexpr int kPwRefAcc   = -1; ///< The running value (previous step's result; entry value before step 0).
    constexpr int kPwRefEntry = -2; ///< The unit's entry value (producer result / primary stream element).
    constexpr int kPwRefNone  = -3; ///< Unused source slot / no register destination.
    constexpr int kPwRefReg0  = -4; ///< Register r: encoded as kPwRefReg0 - r (r in [0, kPwMaxRegs)).
    constexpr int kPwRefOp0   = -8; ///< Operand i: encoded as kPwRefOp0 - i. In a node's pw_steps
                                    ///< attr i indexes node.inputs; in the device plan i is the
                                    ///< dense physical operand slot (see buildPwPlan).

    /// pw_steps step kinds (the first field of a step's 8-int record: kind, code, srcA, srcB,
    /// srcC, dst, bcast, bcastSrc).
    constexpr int kPwKindBinary = 0; ///< srcA OP srcB, OP from the BinaryType wire code.
    constexpr int kPwKindUnary  = 1; ///< unary(srcA), code from the UnaryType wire code.
    constexpr int kPwKindAct    = 2; ///< activation(srcA), code from the ActType wire code.
    constexpr int kPwKindSelect = 3; ///< srcA != 0 ? srcB : srcC (the Where/mask-blend form).
    constexpr int kPwKindLoad   = 4; ///< pass srcA through (an operand load into the accumulator).

    /// Binary codes private to pw steps. They extend the BinaryType wire-code space (Mul..Add =
    /// 0..6) inside a kPwKindBinary step only — binaryFromOnnx() never yields them and the Binary
    /// op kernels never see them; the comparison/PRelu ops fuse through these instead of a subOp.
    constexpr int kPwBinGreater      = 7;  ///< srcA >  srcB -> 1.0/0.0
    constexpr int kPwBinGreaterEqual = 8;  ///< srcA >= srcB -> 1.0/0.0
    constexpr int kPwBinEqual        = 9;  ///< srcA == srcB -> 1.0/0.0
    constexpr int kPwBinPRelu        = 10; ///< srcA > 0 ? srcA : srcB * srcA (slope on srcB)
    constexpr int kPwBinLess         = 11; ///< srcA <  srcB -> 1.0/0.0
    constexpr int kPwBinLessEqual    = 12; ///< srcA <= srcB -> 1.0/0.0

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
    /// True for the ONNX quantized operator family (QuantizeLinear..QGemm) — the ops the
    /// import-time dequantize lowering rewrites; none has a backend kernel.
    bool        opTypeIsQuantized(OpType t);

} // namespace vknn
