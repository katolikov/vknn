// Sub-codes for the Reduce family (Node::subOp) + the ONNX-name lookup.
#pragma once
#include <cstdint>
#include <string>

namespace vknn {

    enum class ReduceType : int32_t { Invalid = -1, Mean = 0, Sum = 1, Max = 2, Min = 3, Prod = 4, L2 = 5 };

    // The ReduceType for an ONNX op name, or Invalid if not in that family.
    ReduceType reduceFromOnnx(const std::string &s);

} // namespace vknn
