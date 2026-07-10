// FusedAttention attribute contract: the interface between the load-time fuseDecodeAttention pass
// (src/import/fuse_decode_attention.cpp) and the two FusedAttention backends.
//
// One FusedAttention node computes the single-query (M == 1) decode-attention core for every
// attention row in one kernel:
//
//   scores[row][s] = (sum_k q[qBase(row) + k*kFaQK] * kv[kBase(row) + s*kFaKN + k*kFaKK])
//                    * kFaScale + mask[mBase(row) + s*kFaMN] * kFaMaskScale
//   p[row][.]      = softmax over s of scores[row][.]          (never stored to memory)
//   out[row*hd + n] = sum_s p[row][s] * v[vBase(row) + s*kFaVK + n*kFaVN]
//
// A "row" is one output attention row: the flattened coordinate over kFaDims, the refined
// batch/head dims taken verbatim from the two folded MatMul views (core/matmul_view.h) — a GQA
// head-group broadcast is a zero stride on the split head axis, so K/V read the KV cache in place.
// qBase/kBase/vBase/mBase are the per-row dot products of the row coordinate with kFaQStride /
// kFaKStride / kFaVStride / kFaMStride. The output is dense row-major over (row, n), which the
// pass verified equals the layout of the tensor it rewired the node's output to (the decomposed
// chain's Transpose/Reshape tail composes to a flat identity there).
//
// The mask operand is node.inputs[3] when present (inputs are {q, k, v[, mask]}); kFaScale and
// kFaMaskScale carry the baked scalar scale in the two placements it occurs (x*s + m for a
// scale-then-mask chain, x*s + m*s for mask-then-scale). Dot products, the softmax running
// max/sum, and the p.V accumulation are fp32 in both backends; the softmax probabilities never
// round through fp16 storage, so a fused graph is numerically finer than — not byte-identical
// to — the decomposed chain.
//
// Set ONLY at session load (gated by Hint::FusedAttention); never serialized — a .vxm on disk
// never contains a FusedAttention node, so every compiled artifact is byte-identical with the
// fusion on or off.
#pragma once

namespace vknn {

    // kFa is the scheme version (1); its presence marks a FusedAttention node's geometry attrs
    // as authoritative.
    inline constexpr const char *kFa          = "fattn";            // int: scheme version, 1
    inline constexpr const char *kFaDims      = "fattn_dims";       // ints: refined row dims (product = row count)
    inline constexpr const char *kFaQStride   = "fattn_qstride";    // ints: q stride per row dim
    inline constexpr const char *kFaKStride   = "fattn_kstride";    // ints: k stride per row dim
    inline constexpr const char *kFaVStride   = "fattn_vstride";    // ints: v stride per row dim
    inline constexpr const char *kFaMStride   = "fattn_mstride";    // ints: mask stride per row dim (mask input only)
    inline constexpr const char *kFaQK        = "fattn_qk";         // int: q element stride per k step
    inline constexpr const char *kFaKN        = "fattn_kn";         // int: k element stride per token step
    inline constexpr const char *kFaKK        = "fattn_kk";         // int: k element stride per k step
    inline constexpr const char *kFaVN        = "fattn_vn";         // int: v element stride per n step
    inline constexpr const char *kFaVK        = "fattn_vk";         // int: v element stride per token step
    inline constexpr const char *kFaMN        = "fattn_mn";         // int: mask element stride per token step (mask input only)
    inline constexpr const char *kFaC         = "fattn_c";          // int: token (score column) count
    inline constexpr const char *kFaHd        = "fattn_hd";         // int: head dim (outputs per row)
    inline constexpr const char *kFaScale     = "fattn_scale";      // float: score scale factor
    inline constexpr const char *kFaMaskScale = "fattn_mask_scale"; // float: mask scale factor (mask input only)
    inline constexpr const char *kFaOut       = "fattn_out";        // ints: output shape (shape-rule backup)

    inline constexpr int kFaVersion = 1;

    // The FusedAttention GPU kernel keeps q and the merged output row in fixed-size shared arrays;
    // head dims above this cap are refused by the pass and the support gate.
    inline constexpr int kFaMaxHeadDim = 256;

} // namespace vknn
