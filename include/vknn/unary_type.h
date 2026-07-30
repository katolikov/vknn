// Sub-codes for the elementwise Unary family (Node::subOp) + the ONNX-name lookup.
#pragma once
#include <cstdint>
#include <string>

namespace vknn {

    /// Which elementwise unary activation a Unary node applies, stored in Node::subOp. The integer
    /// values are part of the serialized model format and are also read by the backend shaders (the
    /// switch in shaders/common.glsl), so they are fixed: never renumber or reorder an enumerator.
    enum class UnaryType : int32_t {
        Invalid     = -1, ///< Not a member of the unary family; sentinel returned by unaryFromOnnx().
        Sigmoid     = 0,  ///< Logistic sigmoid (ONNX "Sigmoid").
        Tanh        = 1,  ///< Hyperbolic tangent (ONNX "Tanh").
        HardSwish   = 2,  ///< Hard swish (ONNX "HardSwish").
        HardSigmoid = 3,  ///< Piecewise-linear sigmoid approximation (ONNX "HardSigmoid").
        LeakyRelu   = 4,  ///< Leaky rectified linear unit (ONNX "LeakyRelu").
        Elu         = 5,  ///< Exponential linear unit (ONNX "Elu").
        Abs         = 6,  ///< Absolute value (ONNX "Abs").
        Neg         = 7,  ///< Arithmetic negation (ONNX "Neg").
        Exp         = 8,  ///< Natural exponential (ONNX "Exp").
        Log         = 9,  ///< Natural logarithm (ONNX "Log").
        Sqrt        = 10, ///< Square root (ONNX "Sqrt").
        Floor       = 11, ///< Round toward negative infinity (ONNX "Floor").
        Ceil        = 12, ///< Round toward positive infinity (ONNX "Ceil").
        Relu        = 13, ///< Rectified linear unit. Imported as its own OpType, so unaryFromOnnx() never yields this.
        SiLU        = 14, ///< Sigmoid-weighted linear unit (swish). Fused from a subgraph, not mapped by unaryFromOnnx().
        Erf         = 15, ///< Gauss error function (ONNX "Erf").
        Cos         = 16, ///< Cosine (ONNX "Cos").
        Sin         = 17, ///< Sine (ONNX "Sin").
        Reciprocal  = 18, ///< Multiplicative inverse (ONNX "Reciprocal").
        Softplus    = 19, ///< Smooth ReLU log(1 + exp(x)) (ONNX "Softplus"); building block for GELU, RoPE, and similar transformer activations.
        Round       = 20, ///< Nearest integer, ties to even (ONNX "Round"); matches GLSL roundEven bitwise, including the sign of a zero result.
        Trunc       = 21, ///< Round toward zero (drop the fraction). Not an ONNX operator: produced only by
                          ///< foldIntRoundtripCast, which collapses a float->wide-int->float Cast pair into
                          ///< this single step so the truncation joins the surrounding pointwise unit.
        Sign = 22,        ///< Elementwise sign: 1 for x>0, -1 for x<0, x itself for +-0 and NaN (ONNX
                          ///< "Sign"). The 0/NaN pass-through keeps the CPU and GPU evaluators
                          ///< bitwise-identical without relying on either language's sign() builtin.
    };

    /// Map an ONNX op-type name to its UnaryType.
    /// @param s ONNX op-type string (e.g. "Sigmoid", "Tanh").
    /// @returns The matching UnaryType, or UnaryType::Invalid when the name is not in the unary
    ///          family. Relu and SiLU are produced by other import paths and are never returned here.
    UnaryType unaryFromOnnx(const std::string &s);

} // namespace vknn
