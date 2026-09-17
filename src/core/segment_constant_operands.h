// Which constant operands the Vulkan segment fills into shared activation buffers.
//
// A GPU kernel reads an activation operand through env.devBuf, which resolves only tensors the segment
// holds a buffer for; an initializer has none unless the segment fills one. Most kernels upload a constant
// operand themselves (operandBuf, or a constant buffer of their own) at the node's precision. The kernels
// that read an operand through env.devBuf instead get it filled by the segment, packed in the
// initializer's layout at the segment's storage precision: operand 0 of every node (a constant left as a
// spatial op's activation input by const folding), every concatenated part of an NC4HW4 Concat (its kernel
// reads each part as an activation), and the fused residual. The segment (VulkanSegment's constructor)
// fills exactly these slots, and the integer fp32 pin (pinIntegerResultsFp32) keeps such a reader at the
// segment's precision, so a pinned kernel never reads an fp16-filled buffer as fp32.
//
// tests/test_integer_region_arithmetic_pins.cpp pins this.
#pragma once
#include "vknn/graph.h"
#include <algorithm>
#include <cstddef>

namespace vknn {

    /// Operand slot range [0, end) of every node the segment fills from a constant: operand 0.
    inline constexpr size_t kSegmentFilledLeadingOperandEnd = 1;

    /// Whether `node` is a Concat running the NC4HW4 kernel (its output is not flat).
    inline bool nc4ConcatNode(const Graph &g, const Node &node) {
        return node.type == OpType::Concat && !node.outputs.empty() && node.outputs[0] != kNoTensor && !g.desc(node.outputs[0]).gpuFlat;
    }

    /// End of the operand slot range [0, end) of `node` the segment fills from constants: every
    /// concatenated part (pwCoreInputs) of an NC4HW4 Concat, operand 0 of every other node, clamped to the
    /// node's operand count. The fused residual (Node::fusedResidual) is filled in addition.
    inline size_t segmentFilledConstantOperandEnd(const Graph &g, const Node &node) {
        if (nc4ConcatNode(g, node))
        {
            return std::min(node.inputs.size(), (size_t) pwCoreInputs(node));
        }
        return std::min(node.inputs.size(), kSegmentFilledLeadingOperandEnd);
    }

} // namespace vknn
