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
        FusedPointwise,  // fused per-element chain (standalone); also the epilogue carried by producers
        // layout conversion nodes (inserted by the layout pass)
        ConvertLayout,
        // fp16 <-> fp32 storage conversion at a selective-fp32 region frontier (inserted by markFp32)
        ConvertDtype,
        Range,        // arange(start, limit, delta) -- scalar inputs, 1-D output
        ConvGemm,     // Conv lowered to an implicit-GEMM kernel (lowerConv); weights repacked [K][Cout]
        Less,         // A <  B -> 1.0/0.0, elementwise with broadcasting (flat path)
        LessEqual,    // A <= B -> 1.0/0.0, elementwise with broadcasting (flat path)
        Dropout,      // identity in inference mode; eliminated at import (eliminateDropout) -- a kept
                      // training-mode / consumed-mask Dropout has no kernel and is unsupported
        TopK,         // k largest/smallest along an axis -> (values, int64 indices); const k (CPU)
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
        RMSNorm,                  // root-mean-square norm: y = x*rsqrt(mean(x^2,last-axis)+eps)*gamma.
                                  // Created by lowerRMSNorm (a Pow/ReduceMean/Add/Sqrt/rsqrt/Mul chain)
                                  // or mapped from SimplifiedLayerNormalization by lowerOrtContribOps;
                                  // fp32 sum-of-squares in a fused flat kernel.
        // ORT contrib operators (com.microsoft domain; ORT transformer exports). Recognized at
        // import and expanded to primitive ops by lowerOrtContribOps — none has a backend kernel,
        // so a variant the expansion declines (an exotic optional input, a non-last axis) surfaces
        // through the support report under its real name instead of "Unknown".
        SimplifiedLayerNorm,     // RMSNorm spelling: (x, gamma) + epsilon/axis -> OpType::RMSNorm
        SkipSimplifiedLayerNorm, // residual add + RMSNorm: (input, skip, gamma[, bias]); output 3 = the sum
        SkipLayerNorm,           // residual add + LayerNorm: (input, skip, gamma[, beta][, bias])
        RotaryEmbedding,         // RoPE: (x, position_ids, cos_cache, sin_cache) + interleaved attr
        MultiHeadAttention,      // fused attention; expanded only in the pure q/k/v(+additive mask) form
        GroupQueryAttention,     // fused GQA with in-op RoPE + KV cache; expanded to the primitive
                                 // rope/concat/repeat_kv/attention subgraph (causal, seqlens_k-driven)
        MatMulNBits,             // blockwise 4-bit weight MatMul: repacked into the int4 wq format
        Rope,                    // fused rotate-half rotary embedding over (x, position_ids,
                                 // cos_table, sin_table) with a `half` attr:
                                 //   y[..., :half]  = x1*cos[p] - x2*sin[p]
                                 //   y[..., half:] = x1*sin[p] + x2*cos[p]   (x1/x2 = last-axis halves)
                                 // ONE dispatch replacing the Slice/Gather/mul/Concat chain a lowered
                                 // contrib RotaryEmbedding expands to. Created ONLY by the load-time
                                 // fuseRope pass (Hint::RopeFusion) — never parsed from ONNX, never
                                 // serialized to a .vxm.
        FusedAttention,          // single-query decode attention core: softmax(q.K^T * scale + mask).V
                                 // in one kernel, operands read through per-axis strides
                                 // (core/fused_attention.h). Created only by the load-time
                                 // fuseDecodeAttention pass — never imported, never serialized.
        ChannelShuffle,          // group-interleave channel permutation (ShuffleNetV2):
                                 //   out[n][c] = in[n][(c % g) * (C/g) + c / g], `groups` attr = g.
                                 // Created by the import-time fuseChannelShuffle pass from the
                                 // Reshape([N,g,C/g,...]) -> Transpose(0,2,1,...) -> Reshape chain;
                                 // never parsed from ONNX (no standard op), but unlike
                                 // Rope/FusedAttention it IS serialized to .vxm (the fold runs in
                                 // runStandardPasses, which vknn_compile applies before saving).
        Det,                     // determinant of (batched) square matrices (ONNX "Det"):
                                 // [..., n, n] -> [...]; a rank-2 input yields a 1-element tensor
                                 // (the engine's IR has no rank-0 activations). The GPU kernel
                                 // covers n <= kDetMaxAnalyticN by fixed-order cofactor expansion;
                                 // larger n runs on the CPU (partial-pivot LU) via the named gate.
    };

    /// Fused-pointwise limits. The fusion pass splits any unit that would exceed one of these;
    /// the shader plan layout (pw_plan.h) is sized from the same constants, so they are a shared
    /// contract between the importer and the kernel and cannot be changed independently.
    constexpr int kPwMaxSteps = 16; ///< Elementwise steps per fused unit.
    /// Largest square-matrix side the GPU Det kernel covers by fixed-order cofactor expansion —
    /// the fast one-thread-per-matrix path every real camera/geometry head hits.
    constexpr int kDetMaxAnalyticN = 4;
    /// Largest side the GPU covers overall: above kDetMaxAnalyticN and up to this bound the kernel
    /// runs an in-register partial-pivot LU per matrix (fp32, deterministic fixed order). Only
    /// n > kDetMaxGpuN — no known real model — takes the CPU's double-precision LU via the named
    /// vkNodeGate refusal.
    constexpr int kDetMaxGpuN    = 8;
    constexpr int kPwMaxOperands = 9; ///< Extra tensor operands per unit (the primary input is excluded).
    constexpr int kPwMaxRank     = 4; ///< Flat broadcast rank stored in the plan; rank>4 is not flat-fused.
    constexpr int kPwMaxRegs     = 4; ///< Named registers for step values reused by later steps.
    constexpr int kPwMaxOuts     = 4; ///< Extra output streams (fanout values exported from the unit).

    /// Broadcast class of a pw operand against the unit's run shape, stored in a step's bcast field.
    /// Every class except kPwBcastGeneral has a closed-form index in BOTH the flat and the NC4HW4
    /// kernel; a general operand is addressable only by the flat kernel's per-axis div/mod walk, so
    /// one of them forces its whole unit onto the flat path (and the layout converts that bracket it).
    constexpr int kPwBcastSame    = 0; ///< Same shape as the run: reads at the output index.
    constexpr int kPwBcastChannel = 1; ///< Per-channel [N,C,1,1]: one value per channel.
    constexpr int kPwBcastGeneral = 2; ///< Anything else: per-axis strided decomposition, flat only.
    constexpr int kPwBcastScalar  = 3; ///< Single element splat.
    constexpr int kPwBcastSpatial = 4; ///< Per-pixel [1,1,H,W]: one value per spatial position.
    /// Row/column masks. The Row/Col forms carry the channel axis (four distinct lane values, a
    /// vec4 load in the NC4HW4 kernel at the operand's own packed index); the *Splat forms hold one
    /// value per row/column at channel lane 0 and splat it across the four lanes, like kPwBcastSpatial.
    /// The class list is append-only: pw_plan.h refuses an unknown class by name at load.
    constexpr int kPwBcastRow      = 5; ///< Per-row [N,C,H,1]: one value per (n, channel, row).
    constexpr int kPwBcastCol      = 6; ///< Per-column [N,C,1,W]: one value per (n, channel, column).
    constexpr int kPwBcastRowSplat = 7; ///< Per-row [1,1,H,1], single batch: one value per row.
    constexpr int kPwBcastColSplat = 8; ///< Per-column [1,1,1,W], single batch: one value per column.
    /// Generic 1-or-full mask: every right-aligned NCHW axis of the operand is 1 or the run's
    /// extent. The NC4HW4 kernel indexes it with per-step packed vec4-space strides (n / cb / h / w,
    /// zero on the broadcast axes) carried in the plan's stride slots — unused by the NC4 world's
    /// other classes — and splats channel lane 0 when the operand's channel axis is 1 (a zero cb
    /// stride). Any batch. The named classes above stay the fast paths; the classifier only lands
    /// here for masks none of them cover, so existing encodings are byte-stable.
    constexpr int kPwBcastPacked = 9;

    /// pw_steps record geometry: ints per step and the field offsets read outside the plan
    /// builder. Mirrored as PW_STEP_FIELDS in shaders/pw_epilogue.glsl.
    constexpr int kPwStepInts       = 8; ///< Ints per step: kind, code, srcA, srcB, srcC, dst, bcast, bcastSrc.
    constexpr int kPwStepSrcA       = 2; ///< Offset of the step's first source field.
    constexpr int kPwStepSrcC       = 4; ///< Offset of the step's last source field (srcA..srcC are contiguous).
    constexpr int kPwStepBcastField = 6; ///< Offset of the step's broadcast-class field.

    /// Stride-slot order of a kPwBcastPacked step's packed vec4-space strides
    /// (plan.stride[s * kPwMaxRank + slot], mirrored as PW_PACKED_STRIDE_* in
    /// shaders/pw_epilogue.glsl). A broadcast axis carries stride 0.
    constexpr int kPwPackedStrideN  = 0; ///< Batch stride: opCb * opH * opW.
    constexpr int kPwPackedStrideCb = 1; ///< Channel-block stride: opH * opW (0 marks a channel-1 operand: splat lane 0).
    constexpr int kPwPackedStrideH  = 2; ///< Row stride: opW.
    constexpr int kPwPackedStrideW  = 3; ///< Column stride: 1.

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
    OpType opTypeFromOnnx(const std::string &s);
    /// True for the ONNX quantized operator family (QuantizeLinear..QGemm) — the ops the
    /// import-time dequantize lowering rewrites; none has a backend kernel.
    bool opTypeIsQuantized(OpType t);

} // namespace vknn
