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

    // Output channel-block pixels (Coutb * OH*OW - the partial pass's threads at one part) above
    // which the reduction is never split: from 18432 up the register-tiled conv1x1 kernel has the
    // waves to hide its own latency and the pair's partial round trip only costs (1024->512
    // @12x12, every 14x14 plane of 128+ output blocks, every 7x7 plane of 512), measured on the
    // primary device with the map-sized chunk decode.
    inline constexpr int64_t kPwSplitKMaxOutputs = 16384;
    // Under that cap the pair wins when the reduction is deep relative to the plane: at most
    // kPwSplitKOutputsPerInputBlock outputs per input channel-block (Coutb*OHW <= Cinb * 128).
    // Measured boundary: 128->64 @14x14 wins 17%, 128->128 @14x14 ties, 256->128 @14x14 wins
    // 25%, 256->256 @14x14 ties, 512->256 @14x14 wins 16%, 480->80 @14x14 wins 52%.
    inline constexpr int64_t kPwSplitKOutputsPerInputBlock = 128;
    // Channel floor: 64->64 @14x14 loses to the single pass even inside the ratio.
    inline constexpr int64_t kPwSplitKMinCin = 64;
    // Thread target of the partial pass (384 waves of 64 lanes), which sets how many ways the
    // reduction splits: measured best at 4 parts for 6272-8192 outputs and 2 at 12544-12800.
    inline constexpr int64_t kPwSplitKTargetThreads = 24576;
    // Each part keeps at least this many (input channel-block, tap) steps, so a shallow reduction
    // on a tiny plane is not cut into slivers whose reduce pass outweighs the parallelism (240->80
    // @14x14 is best at 3-4 parts, 256->256 @7x7 at 4, not the 7-8 the thread target alone would
    // give). A KxK reduction counts its taps: a 128->32 3x3 at 7x7 has 32 blocks of 9 taps and is
    // best at 16 parts (0.029 vs 0.081 ms at 2).
    inline constexpr int64_t kPwSplitKMinStepsPerPart = 16;
    inline constexpr int64_t kPwSplitKMaxParts         = 16;
    inline constexpr int64_t kPwSplitKMinParts         = 2;

    // Chunks the channel-block reduction is split into: the thread target, bounded by the depth
    // floor per part (`tapsPerBlock` steps per channel-block: 1 for a pointwise conv, KH*KW for
    // the general split-K kernel), the block count itself and kPwSplitKMaxParts.
    inline int64_t pwSplitKParts(int64_t Cinb, int64_t Coutb, int64_t OHW, int64_t tapsPerBlock = 1) {
        const int64_t outputs   = Coutb * OHW;
        const int64_t byThreads = (kPwSplitKTargetThreads + outputs - 1) / outputs;
        const int64_t byDepth   = std::max<int64_t>(kPwSplitKMinParts, Cinb * tapsPerBlock / kPwSplitKMinStepsPerPart);
        return std::max<int64_t>(kPwSplitKMinParts, std::min<int64_t>({byThreads, byDepth, Cinb, kPwSplitKMaxParts}));
    }

    // True when a 1x1 conv of this shape runs the split-K partial+reduce pair instead of the
    // register-tiled conv1x1 kernel. `splitKHint` is Config's Hint::SplitKConv (0 = Auto, the
    // calibrated rule; Mode::Off disables the path entirely).
    inline bool pwSplitKActive(bool useFp16, int64_t batch, int64_t Cin, int64_t Coutb, int64_t OHW, int splitKHint) {
        if (!useFp16 || batch != 1 || Cin < kPwSplitKMinCin || splitKHint == (int) Mode::Off)
        {
            return false;
        }
        const int64_t outputs = Coutb * OHW;
        const int64_t Cinb    = (Cin + 3) / 4;
        return outputs <= kPwSplitKMaxOutputs && outputs <= Cinb * kPwSplitKOutputsPerInputBlock;
    }

} // namespace vknn
