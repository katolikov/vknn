// The distribution-focal-loss decode (OpType::FusedDfl) of a YOLOv8-family detection head. The
// exported head reshapes the box map [N, S*B, L] (S sides, B bins per side, L anchors) to
// [N, S, B, L], transposes to [N, L, S, B], takes a softmax over the B bins, transposes to
// [N, B, S, L], applies a 1x1 Conv whose B weights are the bin values, and reshapes to [N, S, L]:
//     y[n][s][l] = sum_b softmax_b(x[n][s*B + b][l]) * w[b]
// The fused node carries x and the weight, with `bins` = B as an attribute; one kernel does the
// whole chain and keeps the softmax and its expectation in fp32.
#pragma once
#include <cstdint>
#include <vector>

namespace vknn {
    /// The chain's two transposes, as the exporter writes them for the rank-4 [N, S, B, L] map.
    inline const std::vector<int64_t> &dflBinsLastPerm() {
        static const std::vector<int64_t> perm {0, 3, 1, 2}; // [N,S,B,L] -> [N,L,S,B]
        return perm;
    }
    inline const std::vector<int64_t> &dflBinsFirstPerm() {
        static const std::vector<int64_t> perm {0, 3, 2, 1}; // [N,L,S,B] -> [N,B,S,L]
        return perm;
    }
    /// Fewest bins a distribution can have (one bin is a constant, not a distribution).
    constexpr int64_t kDflMinBins = 2;
} // namespace vknn
