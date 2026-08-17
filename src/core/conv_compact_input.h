// Compact (unpadded) conv input for shallow-channel activations.
//
// NC4HW4 rounds the channel count up to a multiple of 4, so a 1-channel plane carries four times the
// bytes of its data and every kernel load fetches 8 bytes to use 2 -- on a large input plane that is
// the dominant cost of the first conv. A conv whose input has fewer than 4 channels can instead read
// it straight from the flat row-major (dense) buffer the host already knows how to pack.
//
// The layout assignment (insert_layout_converts.cpp, which stamps the tensor flat) and the kernel
// router (ops/conv.cpp, which picks the compact kernel) must agree exactly, or the graph gets a
// convert spliced back in and the saving disappears. Both call the predicate here.
#pragma once
#include "vknn/graph.h"

namespace vknn {

    /// True when a conv node can read input slot 0 from a compact flat plane: 4-D input with 1..3
    /// channels, and the 3x3 dilation-1 group-1 geometry the compact kernel is written for.
    bool convCompactInputEligible(const Graph &g, const Node &n);

    /// True when `tid` is a graph input (no producer) whose every consumer is a
    /// convCompactInputEligible conv, so stamping it flat removes padding without spawning a convert
    /// for some other reader.
    bool tensorWantsCompactConvInput(const Graph &g, TensorId tid);

} // namespace vknn
