// Shared includes + internal helper declarations for the split graph-pass translation units. Each
// pass lives in its own src/import/<pass>.cpp and includes this header.
#pragma once
#include "passes.h"
#include "backend/cpu/cpu_backend.h"
#include "core/conv_geom.h"
#include "vknn/logging.h"
#include "vknn/precision.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>

namespace vknn {
    // Layout classifier shared by insertLayoutConverts and fusePointwiseChains: true iff the node runs
    // on the flat row-major GPU path (defined in insert_layout_converts.cpp).
    bool gpuFlatNode(const Graph &g, const Node &n);

    // Redirect every reference to tensor `from` so it points at `to`: node inputs, the fused-residual
    // edge (which is not in the inputs list on every op), and graph outputs. Fusion passes that delete a
    // node and fold its output into a producer must use this; rewiring only node.inputs leaves a stale
    // fusedResidual edge dangling at a dead tensor, which crashes a conv residual read. Shared by
    // fuseDwPw and fuseSwish (defined in fuse_dwpw.cpp).
    void rewireTensor(Graph &g, TensorId from, TensorId to);

    // Passes used internally by runStandardPasses but not part of the public passes.h umbrella.
    void lowerEinsum(Graph &g);
    void eliminateFloatCast(Graph &g);
    void fuseMatMulBias(Graph &g);
}
