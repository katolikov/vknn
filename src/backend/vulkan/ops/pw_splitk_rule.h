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

    // Output channel-block pixels (Coutb * OH*OW - the partial pass's threads at one part) at or
    // under which the reduction is split. Measured on the primary device with the map-sized chunk
    // decode: the register-tiled conv1x1 kernel wins from 18432 up (1024->512 @12x12, every 14x14
    // plane of 128+ output blocks, every 7x7 plane of 512 blocks), the split pair wins by 16-44% at
    // 12800 and under (1024->256 @14x14, 2048->1024 @7x7, 1024->512 @7x7..10x10).
    inline constexpr int64_t kPwSplitKMaxOutputs = 16384;
    // Channel floor: at 256 input channels the pair only ties the register-tiled kernel (256->256
    // @14x14), at 512 it wins (512->256 @14x14, -16%).
    inline constexpr int64_t kPwSplitKMinCin = 512;
    // Thread target of the partial pass (384 waves of 64 lanes), which sets how many ways the
    // reduction splits: measured best at 4 parts for 6272-8192 outputs and 2 parts at 12544-12800.
    inline constexpr int64_t kPwSplitKTargetThreads = 24576;
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
        return Coutb * OHW <= kPwSplitKMaxOutputs;
    }

} // namespace vknn
