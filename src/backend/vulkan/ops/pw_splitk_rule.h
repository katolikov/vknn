// Pointwise (1x1, stride 1, pad 0, group 1) split-K shape rule, shared by the standalone Conv op
// (src/backend/vulkan/ops/conv.cpp) and the fused depthwise+project op
// (src/backend/vulkan/ops/fused_dwpw.cpp).
//
// A deep 1x1 conv on a small output plane has too few standard threads to fill the GPU, so the Conv
// op splits its channel reduction across pwSplitKParts() thread groups: conv1x1_splitk_fp16.comp
// sums each channel-block chunk from zero into an fp32 partial, and conv1x1_reduce_fp16.comp adds
// the partials onto the bias in chunk order. That partition changes the fp32 summation order
// relative to the single running sum conv1x1_fp16.comp keeps, so it is a shape rule (never a timing
// race) and the fused op reads the same rule to reproduce the same summation order — the fused
// output stays byte-identical to the unfused depthwise+pointwise pair it replaces.
#pragma once
#include "vknn/hint.h"
#include <algorithm>
#include <cstdint>

namespace vknn {

    // Output pixels a standard 1x1 thread covers (the conv1x1 kernels' WTILE default), used only to
    // size the "is the standard dispatch starved" test below.
    inline constexpr int64_t kPwSplitKTileWidth = 4;
    // Standard-dispatch thread count under which the reduction is split. Above it the register-tiled
    // conv1x1 kernel already has the parallelism and split-K's partial-buffer round-trip only costs.
    inline constexpr int64_t kPwSplitKMaxThreads = 2048;
    // Channel floor: a shallow reduction has nothing to split.
    inline constexpr int64_t kPwSplitKMinCin = 32;
    // Thread target the partial pass aims for, and the cap on how many ways the reduction splits.
    inline constexpr int64_t kPwSplitKTargetThreads = 8192;
    inline constexpr int64_t kPwSplitKMaxParts      = 16;
    inline constexpr int64_t kPwSplitKMinParts      = 2;

    // Chunks the channel-block reduction is split into. Targets kPwSplitKTargetThreads partial-pass
    // threads, capped by the block count itself and kPwSplitKMaxParts.
    inline int64_t pwSplitKParts(int64_t Cinb, int64_t Coutb, int64_t OHW) {
        int64_t parts = (kPwSplitKTargetThreads + Coutb * OHW - 1) / (Coutb * OHW);
        return std::max<int64_t>(kPwSplitKMinParts, std::min<int64_t>({parts, Cinb, kPwSplitKMaxParts}));
    }

    // True when a 1x1 conv of this shape runs the split-K partial+reduce pair instead of the
    // register-tiled conv1x1 kernel. `splitKHint` is Config's Hint::SplitKConv (0 = Auto, the
    // calibrated rule; Mode::Off disables the path entirely).
    inline bool pwSplitKActive(bool useFp16, int64_t batch, int64_t Cin, int64_t Coutb, int64_t OHW, int splitKHint) {
        if (!useFp16 || batch != 1 || Cin < kPwSplitKMinCin || splitKHint == (int) Mode::Off)
        {
            return false;
        }
        const int64_t stdThreads = Coutb * ((OHW + kPwSplitKTileWidth - 1) / kPwSplitKTileWidth);
        return stdThreads < kPwSplitKMaxThreads;
    }

} // namespace vknn
