// Sub-codes for the elementwise Unary family (Node::subOp) + the ONNX-name lookup.
#pragma once
#include <cstdint>
#include <string>

namespace vknn {

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

    // The UnaryType for an ONNX op name, or Invalid if not in that family.
    UnaryType unaryFromOnnx(const std::string &s);

} // namespace vknn
