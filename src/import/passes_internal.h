// Shared includes + internal helper declarations for the split graph-pass translation units. Each
// pass lives in its own src/import/<pass>.cpp and includes this header.
#pragma once
#include "backend/cpu/cpu_backend.h"
#include "core/conv_geom.h"
#include "passes.h"
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
    /// Is this Slice a block-aligned channel slice, i.e. a contiguous NC4HW4 block-range copy?
    bool sliceIsNc4(const Graph &g, const Node &n);
    /// Is this Reduce a spatial reduction, i.e. one reduction per channel over the blocked plane?
    bool reduceIsNc4(const Graph &g, const Node &n);

    // The Config::fp32Tensors include/exclude substring matcher, shared by markFp32 (load) and the
    // fusion pass's compile-time fp32 prediction (defined in mark_fp32.cpp).
    bool fp32NameMatch(const std::string &name, const std::string &substrs);

    // The tensor named `name` when it is a GRAPH INPUT of `g`, else kNoTensor. A bucket plan reads
    // only declared inputs — an internal tensor that happens to share the name carries no boundary
    // shape (defined in plan_widened_bucket.cpp).
    TensorId graphInputByName(const Graph &g, const std::string &name);

    // Redirect every reference to tensor `from` so it points at `to`: node inputs, the fused-residual
    // and fused-bias edges (which are not in the inputs list on every op), and graph outputs. Fusion
    // passes that delete a node and fold its output into a producer must use this; rewiring only
    // node.inputs leaves a stale fused edge dangling at a dead tensor, which crashes a conv residual
    // (or matmul bias) read (defined in fuse_dwpw.cpp).
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
    // Fold a float -> wide-integer -> float Cast pair (INT32/INT64/UINT32/UINT64 intermediate, read by
    // that second cast alone) into one Unary(Trunc), so the truncation becomes a float pointwise
    // member the fuser can span instead of a fusion-splitting integer barrier. Byte-identical to the
    // cast pair (both truncate toward zero) for every finite value; the same forward dtype lattice as
    // eliminateFloatCast gates it to a proven float source. Runs after eliminateFloatCast, before the
    // pointwise fusion (defined in fold_int_roundtrip_cast.cpp).
    void foldIntRoundtripCast(Graph &g);
    // Fold a Gather whose constant rank-1 indices form a contiguous ascending unit run into the
    // equivalent Slice (same elements, same order, byte-identical), so the runtime's slice
    // optimizations apply: identity slices alias their input, leading-axis slices become zero-copy
    // sub-buffer views. Scalar and rank>1 indices are untouched. Runs after the const-fold fixpoint
    // (indices are resolved initializers), before layout/fusion (defined in fold_gather_slice.cpp).
    void foldGatherToSlice(Graph &g);
    // Fuse the decomposed RMSNormalization chain (Pow(x,2) -> ReduceMean(last-axis) -> Add(eps) ->
    // Sqrt -> Reciprocal|Div(1,.) -> Mul(x,.) -> Mul(gamma,.)) into one OpType::RMSNorm node, so the
    // wide sum of squares accumulates in fp32 in a single kernel instead of losing precision across
    // the fp16-stored decomposition. Runs after eliminateFloatCast (the chain is Cast-free) and the
    // const-fold/shape fixpoint (the eps/exponent constants are resolved initializers), before the
    // pointwise fusion (defined in lower_rmsnorm.cpp).
    void lowerRMSNorm(Graph &g);
    // Expand the ORT contrib operators (com.microsoft: Skip/SimplifiedLayerNormalization,
    // RotaryEmbedding, the pure MultiHeadAttention form, GroupQueryAttention, MatMulNBits) into
    // primitive ops in a fixpoint with inferShapes — the attention/rotary expansions emit concrete
    // shape constants, so each round unlocks the next layer's chain. A variant outside the expanded
    // forms is left in place under its real name (unsupported at plan, never silently miscomputed);
    // a graph without contrib ops is a scan-only no-op. Runs right after the first inferShapes
    // (defined in lower_ort_contrib.cpp).
    void lowerOrtContribOps(Graph &g);
    // Fuse the scaled-flow warp coordinate idiom
    // Mul(flow,scale)->Transpose(0,2,3,1)->Add(base_grid,.)->GridSample into one GridSample that
    // computes coord = base + scale*flow in its own coordinate lookup, so the full-resolution grid
    // and its NHWC Transpose are never materialized. Bit-exact with the split chain; opportunistic
    // (only the exact single-consumer pattern). Needs resolved shapes; runs before fusePointwiseChains
    // so the coordinate chain is claimed before the pointwise fusion would host the Add
    // (defined in fuse_gridsample_warp.cpp).
    void fuseGridSampleWarp(Graph &g);
    // Lower a general grouped Conv (1 < group < Cin, incl. the channel-multiplier depthwise
    // group == Cin/Cout != Cin) into `group` independent group-1 Convs over per-group channel slices
    // joined by a Concat, so each part runs on the proven dense Conv GPU kernel. Needs a resolved
    // rank-4 input, a constant rank-4 weight, and channels that partition evenly by group; anything
    // else is left as a grouped Conv for the group-aware CPU op (defined in lower_grouped_conv.cpp).
    void lowerGroupedConv(Graph &g);
    // Fold the group-interleave channel-shuffle idiom (ShuffleNetV2)
    // Reshape([N,C,...] -> [N,g,C/g,...]) -> Transpose(swap axes 1,2) -> Reshape(back to [N,C,...])
    // into one OpType::ChannelShuffle node with a `groups` attr, so the permutation runs as ONE
    // dispatch in the surrounding layout instead of two Reshapes plus a flat gather with layout
    // round-trips. Generic pattern match on resolved shapes (never model-specific); bit-identical
    // (pure data movement). Runs after the const-fold/shape fixpoint (the Reshape targets are
    // resolved), before fusePointwiseChains (defined in fuse_channel_shuffle.cpp).
    void fuseChannelShuffle(Graph &g);
} // namespace vknn
