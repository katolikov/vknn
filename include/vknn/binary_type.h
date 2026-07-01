// Sub-codes for the elementwise Binary family (Node::subOp) + the ONNX-name lookup.
#pragma once
#include <cstdint>
#include <string>

namespace vknn {

    enum class BinaryType : int32_t { Invalid = -1, Mul = 0, Sub = 1, Div = 2, Max = 3, Min = 4, Pow = 5, Add = 6 };

    // The BinaryType for an ONNX op name, or Invalid if not in that family.
    BinaryType binaryFromOnnx(const std::string &s);

} // namespace vknn
