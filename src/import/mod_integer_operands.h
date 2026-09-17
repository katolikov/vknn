// Whether an ONNX Mod computes on an integer element type, resolved from the graph IR.
//
// ONNX gives Mod one element type T for both operands and the result, and an integer T changes the
// answer: the operands follow the exact int64 rule, and a zero divisor yields 0 (ONNX leaves integer
// division by zero undefined; the engine defines it as 0) where a float T gives NaN for fmod 1. The
// engine carries INT32, INT8 and UINT8 values in fp32 lanes on the CPU and in float lanes on the GPU,
// so a runtime dtype does not say which T the model declared. This resolver reads it from the graph
// instead (import/integer_elements.h, ElementFact::IntegerValues), and the CPU kernel
// (backend/cpu/ops/mod.cpp), the Vulkan kernel (backend/vulkan/ops/mod.cpp) and the fp32 pin of integer
// results (pinIntegerResultsFp32) share it, so all three agree on every node.
#pragma once
#include "vknn/graph.h"
#include <vector>

namespace vknn {

    /// Index of the node writing each tensor (the last writer), -1 for a tensor no node writes (graph
    /// inputs, initializers), the convention of pinIntegerResultsFp32's producer index. Sized to
    /// `g.tensors` (tensorProducers).
    std::vector<int> modTensorProducers(const Graph &g);

    /// Whether Mod node `mod` computes on an integer element type: its result's declared dtype is Int64,
    /// Int32, Int8 or UInt8, or its dividend or divisor holds integer element values
    /// (ElementFact::IntegerValues, whose rules integer_elements.h lists). A float-declared graph input or
    /// initializer (an INT32, INT16 or UINT16 initializer imports as Float32 and is not recovered), a Cast
    /// to a float type and a producer hosting a fused pointwise chain do not.
    /// @param producers modTensorProducers(g), or an equivalent last-writer index.
    bool modOperandsAreInteger(const Graph &g, const Node &mod, const std::vector<int> &producers);

    /// modOperandsAreInteger over a producer index built for this call.
    bool modOperandsAreInteger(const Graph &g, const Node &mod);

} // namespace vknn
