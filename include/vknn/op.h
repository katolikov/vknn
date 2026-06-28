// Op types, the fused-activation codes, the attribute bag, and the IR node struct.
#pragma once
#include "vknn/common.h"
#include <map>
#include <string>
#include <vector>

namespace vknn {

    using TensorId               = int32_t;
    constexpr TensorId kNoTensor = -1;

    /// Fused activation applied after an op (kept in sync with shaders/common.glsl).
    enum class ActType : int32_t { None = 0, Relu = 1, Relu6 = 2, Clip = 3, HardSwish = 4, SiLU = 5 };

    /// Operator types. Add a new value here + a name mapping + register kernels.
    enum class OpType {
        Unknown = 0,
        Conv, // Conv2D (incl. depthwise via group, pointwise 1x1)
        Clip, // Clip / Relu6
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
        ConstantOfShape, // emit a tensor of the given shape filled with a scalar value
        EyeLike,         // identity-like matrix (ones on a diagonal) matching the input shape
        ScatterND,       // copy data, then scatter update slices at N-D index rows
        FusedSE,         // fused Squeeze-Excite scale: GAP->FC->relu->FC->hardsigmoid (one kernel)
        FusedDwPw,       // fused depthwise-3x3 + 1x1-project (expanded intermediate stays on-chip)
        // layout conversion nodes (inserted by the layout pass)
        ConvertLayout,
    };

    // Sub-codes for the Unary/Binary/Reduce families, stored in Node::subOp. The integer values are
    // kept in sync with the switch in shaders/common.glsl, so they are fixed and explicit. Invalid
    // (-1) marks "this ONNX op is not in the family".
    enum class UnaryType : int32_t {
        Invalid     = -1,
        Sigmoid     = 0,
        Tanh        = 1,
        HardSwish   = 2,
        HardSigmoid = 3,
        LeakyRelu   = 4,
        Elu         = 5,
        Abs         = 6,
        Neg         = 7,
        Exp         = 8,
        Log         = 9,
        Sqrt        = 10,
        Floor       = 11,
        Ceil        = 12,
        Relu        = 13,
        SiLU        = 14,
        Erf         = 15,
        Cos         = 16,
        Sin         = 17,
        Reciprocal  = 18,
        Softplus    = 19 // transformer: GELU, RoPE, ...
    };
    enum class BinaryType : int32_t { Invalid = -1, Mul = 0, Sub = 1, Div = 2, Max = 3, Min = 4, Pow = 5, Add = 6 };
    enum class ReduceType : int32_t { Invalid = -1, Mean = 0, Sum = 1, Max = 2, Min = 3, Prod = 4, L2 = 5 };

    const char *opTypeName(OpType t);
    OpType      opTypeFromOnnx(const std::string &s);
    // The UnaryType/BinaryType/ReduceType for an ONNX op name, or Invalid if not in that family.
    UnaryType  unaryFromOnnx(const std::string &s);
    BinaryType binaryFromOnnx(const std::string &s);
    ReduceType reduceFromOnnx(const std::string &s);

    /// A single attribute value (subset needed for CNNs).
    struct Attr {
        enum Kind { None, Int, Float, Ints, Floats, String } kind = None;
        int64_t              i                                    = 0;
        float                f                                    = 0;
        std::vector<int64_t> ints;
        std::vector<float>   floats;
        std::string          str;
        // For tensor-valued attributes (a Constant node's `value`): the tensor's dims, so the op can
        // emit the right shape instead of a flat 1-D vector (multi-dim constants like anchor grids).
        std::vector<int64_t> shape;
    };

    struct Attributes {
        std::map<std::string, Attr> map;
        bool                        has(const std::string &k) const {
            return map.count(k) > 0;
        }
        int64_t geti(const std::string &k, int64_t d = 0) const {
            auto it = map.find(k);
            return it == map.end() ? d : it->second.i;
        }
        float getf(const std::string &k, float d = 0) const {
            auto it = map.find(k);
            return it == map.end() ? d : it->second.f;
        }
        const std::vector<int64_t> &getints(const std::string &k) const {
            static const std::vector<int64_t> e;
            auto                              it = map.find(k);
            return it == map.end() ? e : it->second.ints;
        }
        std::string gets(const std::string &k, const std::string &d = "") const {
            auto it = map.find(k);
            return it == map.end() ? d : it->second.str;
        }
    };

    /// IR node. References tensors by id into Graph::tensors.
    struct Node {
        OpType                type = OpType::Unknown;
        std::string           name;
        std::vector<TensorId> inputs;
        std::vector<TensorId> outputs;
        Attributes            attr;
        // Fusion metadata filled by graph passes:
        ActType fusedAct = ActType::None;
        float   actLo = 0, actHi = 0;
        // For kUnary/kBinary: the UnaryType/BinaryType code. For unary ops with params (LeakyRelu/Elu
        // alpha, HardSigmoid alpha/beta) the params live in actLo/actHi.
        int32_t subOp = 0;
        // Conv only: a residual tensor fused into the epilogue (out = act(conv + residual)); set by the
        // residual-Add fusion pass. kNoTensor when not fused.
        TensorId fusedResidual = kNoTensor;
    };

} // namespace vknn
