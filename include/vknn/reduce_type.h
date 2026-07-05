/// Sub-codes for the Reduce family (stored in Node::subOp) plus the ONNX-name lookup.
#pragma once
#include <cstdint>
#include <string>

namespace vknn {

    /// Reduction kind carried by a Reduce node. The value is the on-wire sub-op code
    /// stored in Node::subOp, so the enumerators are ABI-stable and must not be renumbered.
    enum class ReduceType : int32_t {
        Invalid = -1, ///< Not a member of the Reduce family (returned by reduceFromOnnx on no match).
        Mean    = 0,  ///< Arithmetic mean of the reduced elements (ReduceMean).
        Sum     = 1,  ///< Sum of the reduced elements (ReduceSum).
        Max     = 2,  ///< Maximum of the reduced elements (ReduceMax).
        Min     = 3,  ///< Minimum of the reduced elements (ReduceMin).
        Prod    = 4,  ///< Product of the reduced elements (ReduceProd).
        L2      = 5,  ///< L2 norm (square root of the sum of squares) of the reduced elements (ReduceL2).
    };

    /// Map an ONNX op name to its ReduceType.
    /// @param s ONNX op type (e.g. "ReduceMean", "ReduceSum").
    /// @returns The matching ReduceType, or ReduceType::Invalid if `s` names no Reduce-family op.
    ReduceType reduceFromOnnx(const std::string &s);

} // namespace vknn
