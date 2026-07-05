// Sub-codes for the elementwise Binary family (Node::subOp) + the ONNX-name lookup.
#pragma once
#include <cstdint>
#include <string>

namespace vknn {

    /// Which elementwise binary operation a Binary node applies, stored in Node::subOp. The integer
    /// values are part of the serialized model format and are also read by the backend shaders, so they
    /// are fixed: never renumber or reorder an enumerator.
    enum class BinaryType : int32_t {
        Invalid = -1, ///< Not a member of the binary family; sentinel returned by binaryFromOnnx().
        Mul     = 0,  ///< Elementwise product (ONNX "Mul").
        Sub     = 1,  ///< Elementwise difference (ONNX "Sub").
        Div     = 2,  ///< Elementwise quotient (ONNX "Div").
        Max     = 3,  ///< Elementwise maximum (ONNX "Max").
        Min     = 4,  ///< Elementwise minimum (ONNX "Min").
        Pow     = 5,  ///< Elementwise power (ONNX "Pow").
        Add     = 6,  ///< Elementwise sum. Imported as its own OpType::Add, so binaryFromOnnx() never yields this.
    };

    /// Map an ONNX op-type name to its BinaryType.
    /// @param s ONNX op-type string (e.g. "Mul", "Sub").
    /// @returns The matching BinaryType, or BinaryType::Invalid when the name is not in the binary
    ///          family. "Add" is imported separately and is therefore never returned here.
    BinaryType binaryFromOnnx(const std::string &s);

} // namespace vknn
