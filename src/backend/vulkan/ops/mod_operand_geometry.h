// Broadcast geometry of the flat Vulkan Mod kernel's operands (backend/vulkan/ops/mod.cpp), kept free
// of Vulkan types so tests/test_mod_ops.cpp checks it on the host.
//
// shaders/mod.comp reads an operand element at the sum over the output axes of coordinate x stride,
// with stride 0 on an operand axis of extent 1. That index stays inside the operand's buffer only when
// every right-aligned operand extent is 1 or the output's extent, so an operand that does not broadcast
// to the output is rejected before any stride is built; the CPU oracle rejects the same node.
#pragma once
#include "vknn/error.h"
#include "vknn/shape.h"
#include <string>
#include <vector>

namespace vknn {

    /// `operandShape` right-aligned into the rank of `outputShape` (leading axes padded with extent 1).
    /// @param nodeName     Node name for the error message.
    /// @param operandShape Shape of one Mod operand.
    /// @param outputShape  Shape of the Mod result.
    /// @throws Error(Status::InvalidArgument) when the operand's rank exceeds the output's, or when an
    ///         aligned extent is neither 1 nor the output's extent on that axis.
    inline std::vector<int64_t> modAlignedOperandExtents(const std::string &nodeName, const Shape &operandShape, const Shape &outputShape) {
        const size_t outputRank = outputShape.size();
        if (operandShape.size() > outputRank)
        {
            throw Error(Status::InvalidArgument, "Mod '" + nodeName + "': operand shape " + shapeStr(operandShape) + " has a higher rank than the output " + shapeStr(outputShape));
        }
        const size_t         paddingAxes = outputRank - operandShape.size();
        std::vector<int64_t> alignedExtents(outputRank, 1);
        for (size_t axis = 0; axis < operandShape.size(); ++axis)
        {
            alignedExtents[paddingAxes + axis] = operandShape[axis];
        }
        for (size_t axis = 0; axis < outputRank; ++axis)
        {
            if (alignedExtents[axis] != 1 && alignedExtents[axis] != outputShape[axis])
            {
                throw Error(Status::InvalidArgument, "Mod '" + nodeName + "': operand shape " + shapeStr(operandShape) + " is not broadcast-compatible with output " + shapeStr(outputShape) + " (axis " + std::to_string(axis) + ")");
            }
        }
        return alignedExtents;
    }

} // namespace vknn
