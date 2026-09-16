// When the session keeps the Vulkan flat-layout pass on although Config::flatLayout (Hint::FlatLayout,
// CLI --no-flat) asks to skip it.
//
// The flat-layout pass is what makes a graph runnable on the GPU, not an optimization on top of it: an op
// whose kernel reads flat row-major has no plan without the layout assignment and the converts the pass
// splices. The rule reads each op's descriptor layout class, not the node's shapes, so it keeps the pass on
// for every Flat op (Transpose, Slice, MatMul, Clip, Reduce, Where, the comparisons, ArgMax, Mod, ...) and
// every ShapeDependent op (Add, Binary, Concat, Split, Softmax, Pad, Gather, ConvTranspose, ...), including
// the ShapeDependent ops whose node runs an NC4HW4 kernel. A request to skip the pass therefore takes effect
// only on a graph of Nc4-class ops alone (Conv, Relu, the pools, Resize, ...).
//
// tests/test_arg_extreme_ops.cpp pins this.
#pragma once
#include "vknn/graph.h"
#include "vknn/op_descriptor.h"
#include <algorithm>

namespace vknn {

    /// Whether `g` holds an op whose descriptor layout class is Flat or ShapeDependent, so the session keeps
    /// the flat-layout pass on whatever Config::flatLayout asks.
    inline bool graphKeepsFlatLayoutPass(const Graph &g) {
        return std::any_of(g.nodes.begin(), g.nodes.end(), [](const Node &node) {
            const LayoutClass layout = opDescriptor(node.type).layout;
            return layout == LayoutClass::Flat || layout == LayoutClass::ShapeDependent;
        });
    }

} // namespace vknn
