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

    // Lower the two batched-matmul Einsum equations ("...ab,...b->...a", "bij,bnjk->bnik") to
    // Unsqueeze + MatMul (+ Squeeze) so they run on the flat MatMul GPU kernel; other equations are
    // left as Einsum. Needs resolved operand shapes (defined in lower_einsum.cpp).
    void lowerEinsum(Graph &g);
    // Drop Cast nodes converting float->float (a same-size buffer copy), rewiring consumers to the
    // cast input; a forward dtype pass gates removal to a float source so int<->float casts survive.
    // Graph outputs are never renamed (defined in eliminate_float_cast.cpp).
    void eliminateFloatCast(Graph &g);
    // Fuse Add(MatMul(A,W), bias) into the MatMul epilogue when the MatMul output feeds only this Add
    // and the other operand is a rank-1 [N] initializer matching the output's last dim (defined in
    // fuse_matmul_bias.cpp).
    void fuseMatMulBias(Graph &g);
}
