// Shared includes + internal helper declarations for the split graph-pass translation units. Each
// pass lives in its own src/import/<pass>.cpp and includes this header.
#pragma once
#include "passes.h"
#include "backend/cpu/cpu_backend.h"
#include "core/conv_geom.h"
#include "vknn/logging.h"
#include "vknn/op_descriptor.h"
#include "vknn/precision.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>

namespace vknn {
    // Layout classifier shared by insertLayoutConverts and fusePointwiseChains: true iff the node runs
    // on the flat row-major GPU path (defined in insert_layout_converts.cpp).
    bool gpuFlatNode(const Graph &g, const Node &n);

    // The Config::fp32Tensors include/exclude substring matcher, shared by markFp32 (load) and the
    // fusion pass's compile-time fp32 prediction (defined in mark_fp32.cpp).
    bool fp32NameMatch(const std::string &name, const std::string &substrs);

    // Redirect every reference to tensor `from` so it points at `to`: node inputs, the fused-residual
    // edge (which is not in the inputs list on every op), and graph outputs. Fusion passes that delete a
    // node and fold its output into a producer must use this; rewiring only node.inputs leaves a stale
    // fusedResidual edge dangling at a dead tensor, which crashes a conv residual read (defined in
    // fuse_dwpw.cpp).
    void rewireTensor(Graph &g, TensorId from, TensorId to);

    // Passes used internally by runStandardPasses but not part of the public passes.h umbrella.

    // Lower the two batched-matmul Einsum equations ("...ab,...b->...a", "bij,bnjk->bnik") to
    // Unsqueeze + MatMul (+ Squeeze) so they run on the flat MatMul GPU kernel; other equations are
    // left as Einsum. Needs resolved operand shapes (defined in lower_einsum.cpp).
    void lowerEinsum(Graph &g);
    // Lower every inference-mode BatchNorm that foldBatchNorm left standing (pre-activation BN, BN
    // after Concat, BN on a shared Conv output) to a per-channel Mul+Add pair with host-folded
    // [1,C,1..] fp32 scale/shift initializers, so the pointwise fusion pass can fold it. Runs
    // unconditionally after foldBatchNorm (defined in lower_batchnorm.cpp).
    void lowerBatchNorm(Graph &g);
    // Lower every InstanceNormalization with fp32-initializer scale/B and a resolved rank>=3 input
    // to spatial ReduceMean + Sub/Mul/Add/Sqrt/Div ops (no InstanceNorm kernel exists); an
    // ineligible node keeps its opaque op with a WARN. Needs resolved input shapes, so it runs
    // after the const-fold/infer fixpoint (defined in lower_instancenorm.cpp).
    void lowerInstanceNorm(Graph &g);
    // Drop Cast nodes converting float->float (a same-size buffer copy), rewiring consumers to the
    // cast input; a forward dtype pass gates removal to a float source so int<->float casts survive.
    // Graph outputs are never renamed (defined in eliminate_float_cast.cpp).
    void eliminateFloatCast(Graph &g);
    // Lower a general grouped Conv (1 < group < Cin, incl. the channel-multiplier depthwise
    // group == Cin/Cout != Cin) into `group` independent group-1 Convs over per-group channel slices
    // joined by a Concat, so each part runs on the proven dense Conv GPU kernel. Needs a resolved
    // rank-4 input, a constant rank-4 weight, and channels that partition evenly by group; anything
    // else is left as a grouped Conv for the group-aware CPU op (defined in lower_grouped_conv.cpp).
    void lowerGroupedConv(Graph &g);
}
