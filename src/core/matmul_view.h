// Operand-view metadata for MatMul: the attribute contract between the load-time
// foldMatMulViews pass (src/import/fold_matmul_views.cpp) and the two MatMul backends.
//
// A view lets a MatMul read an operand directly from a producer buffer through per-axis element
// strides instead of consuming a materialized Transpose/Expand/Reshape chain. The pass rewires the
// operand input to the chain's source tensor and stores the composed addressing here; a backend
// that sees kMmView uses these values verbatim in place of the dense row-major derivation it would
// otherwise run over the operand shapes.
//
// Addressing model (mirrors the flat matmul geometry SSBO): the OUTPUT is dense row-major over
// kMmViewDims — the output shape with batch axes possibly split so that a head-group broadcast is
// expressible as a plain zero stride (a GQA merge of [kvHeads, group] into one head axis becomes
// two entries, the group entry carrying bStride 0). The last two entries are always the unsplit
// M and N axes. For an output coordinate c over those dims:
//   A element(k) = a[ sum_d c[d]*kMmViewAStride[d] + k*kMmViewAK ]
//   B element(k) = b[ sum_d c[d]*kMmViewBStride[d] + k*kMmViewBK ]
// with k walked 0..K-1 in ascending order (the accumulation-order contract is unchanged, so a
// folded MatMul is bit-identical to the materialized chain).
//
// Set ONLY at session load; never serialized (a .vxm on disk never carries these attrs, so every
// compiled artifact is byte-identical with the fold on or off). Both backends honor the same
// attrs, so CPU-oracle parity holds on folded graphs.
#pragma once

namespace vknn {

    // kMmView is the scheme version (1); its presence marks the node's operands as view-addressed
    // and makes every other kMmView* attribute authoritative.
    inline constexpr const char *kMmView        = "mmview";         // int: scheme version, 1
    inline constexpr const char *kMmViewDims    = "mmview_dims";    // ints: refined output dims
    inline constexpr const char *kMmViewAStride = "mmview_astride"; // ints: A stride per refined dim
    inline constexpr const char *kMmViewBStride = "mmview_bstride"; // ints: B stride per refined dim
    inline constexpr const char *kMmViewAK      = "mmview_ak";      // int: A element stride per k step
    inline constexpr const char *kMmViewBK      = "mmview_bk";      // int: B element stride per k step
    inline constexpr const char *kMmViewM       = "mmview_m";       // int: M (rows of A / output)
    inline constexpr const char *kMmViewN       = "mmview_n";       // int: N (cols of B / output)
    inline constexpr const char *kMmViewK       = "mmview_k";       // int: K, the reduction extent

    inline constexpr int kMmViewVersion = 1;

} // namespace vknn
