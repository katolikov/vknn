// Workgroup-width contracts of the Vulkan conv op (src/backend/vulkan/ops/conv.cpp).
//
// Every conv pipeline is COMPILED at a workgroup width - the workgroup-size specialization constant
// it is created with - and every dispatch of that pipeline DIVIDES its thread count by a width to
// get a group count. Those two widths are one value: a pipeline compiled 64 lanes wide whose
// dispatch divides by 128 launches half the lanes the kernel's bound expects and leaves half the
// outputs at whatever the destination buffer already held. The rules that derive spec vectors,
// group counts, OC-split slice sizes and race wave counts from that one width live here as pure
// functions, so they are exercised at every width laneWidthFor (src/backend/vulkan/ops/flat_ops.h)
// can return for a device subgroup, not only at the 64 of a wave-64 GPU.
#pragma once
#include "vknn/hint.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vknn {

    /// Lanes in the wave vk::KernelCost::waves counts ("workgroups * localSize / 64", see
    /// src/backend/vulkan/vk_tune_model.h). The analytical race model's saturation constant is
    /// calibrated in these units, so every entrant of one race reports its wave count in them
    /// whatever workgroup width it dispatches at.
    constexpr int64_t kConvCostWaveLanes = 64;

    /// Specialization-constant slots of the register-tiled conv kernel (shaders/conv_reg.comp and
    /// shaders/conv_reg_fp16.comp): 0 = OCB_BLK, 1 = WTILE, 2 = local_size_x_id, the workgroup
    /// width. kConvRegSpecSlots is the count every conv_reg spec vector carries - a vector short of
    /// the trailing slot leaves the shader's DECLARED default width in force while the dispatch
    /// divides by the device width.
    constexpr size_t kConvRegOcbSpecIndex       = 0;
    constexpr size_t kConvRegPixelTileSpecIndex = 1;
    constexpr size_t kConvRegLaneWidthSpecIndex = 2;
    constexpr size_t kConvRegSpecSlots          = 3;

    /// OCB_BLK / WTILE defaults declared by conv_reg.comp: one output channel-block and four output
    /// pixels per thread, the tile a caller that states no other computes.
    constexpr uint32_t kConvRegDefaultOcbBlocks = 1;
    constexpr uint32_t kConvRegDefaultPixelTile = 4;

    /// Workgroups a 1-D dispatch of `threads` lanes takes at workgroup width `laneWidth`.
    constexpr int64_t convDispatchGroups(int64_t threads, int64_t laneWidth) {
        return laneWidth > 0 ? (threads + laneWidth - 1) / laneWidth : 0;
    }

    /// Lanes that dispatch actually launches: whole workgroups of `laneWidth`, the padding lanes
    /// past `threads` included (the kernel's own range bound retires them).
    constexpr int64_t convDispatchLanes(int64_t threads, int64_t laneWidth) {
        return convDispatchGroups(threads, laneWidth) * laneWidth;
    }

    /// The same dispatch in kConvCostWaveLanes-wide waves - the unit every conv race entrant's
    /// KernelCost::waves is expressed in.
    inline double convDispatchWaves(int64_t threads, int64_t laneWidth) {
        return (double) convDispatchLanes(threads, laneWidth) / (double) kConvCostWaveLanes;
    }

    /// conv_reg's specialization constants, always all kConvRegSpecSlots of them.
    inline std::vector<uint32_t> convRegSpecConstants(uint32_t ocbBlocks, uint32_t pixelTile, uint32_t laneWidth) {
        std::vector<uint32_t> spec(kConvRegSpecSlots);
        spec[kConvRegOcbSpecIndex]       = ocbBlocks;
        spec[kConvRegPixelTileSpecIndex] = pixelTile;
        spec[kConvRegLaneWidthSpecIndex] = laneWidth;
        return spec;
    }

    /// Specialization-constant slots of the 3x3 row-halo conv kernel (shaders/conv3x3_row*.comp):
    /// 0 = OCB_BLK, 1 = WTILE, 2 = SW_SPEC (the horizontal stride, which sizes the register row
    /// segment), 3 = local_size_x_id, the workgroup width. Same contract as conv_reg: the host states
    /// every slot, so the kernel's declared width never stands in for the device-resolved one.
    constexpr size_t kConvRowOcbSpecIndex       = 0;
    constexpr size_t kConvRowPixelTileSpecIndex = 1;
    constexpr size_t kConvRowStrideSpecIndex    = 2;
    constexpr size_t kConvRowLaneWidthSpecIndex = 3;
    constexpr size_t kConvRowSpecSlots          = 4;

    inline std::vector<uint32_t> convRowSpecConstants(uint32_t ocbBlocks, uint32_t pixelTile, uint32_t strideW, uint32_t laneWidth) {
        std::vector<uint32_t> spec(kConvRowSpecSlots);
        spec[kConvRowOcbSpecIndex]       = ocbBlocks;
        spec[kConvRowPixelTileSpecIndex] = pixelTile;
        spec[kConvRowStrideSpecIndex]    = strideW;
        spec[kConvRowLaneWidthSpecIndex] = laneWidth;
        return spec;
    }

    /// Specialization-constant slots of the compact-input 3x3 conv kernel
    /// (shaders/conv3x3_cin_lt4*.comp): the row-halo slots 0..2, then 3 = CIN (the 1..3 input
    /// channels it gathers from the dense plane) and 4 = local_size_x_id, the workgroup width.
    constexpr size_t kConvCompactOcbSpecIndex       = kConvRowOcbSpecIndex;
    constexpr size_t kConvCompactPixelTileSpecIndex = kConvRowPixelTileSpecIndex;
    constexpr size_t kConvCompactStrideSpecIndex    = kConvRowStrideSpecIndex;
    constexpr size_t kConvCompactCinSpecIndex       = 3;
    constexpr size_t kConvCompactLaneWidthSpecIndex = 4;
    constexpr size_t kConvCompactSpecSlots          = 5;

    inline std::vector<uint32_t> convCompactSpecConstants(uint32_t ocbBlocks, uint32_t pixelTile, uint32_t strideW, uint32_t inputChannels, uint32_t laneWidth) {
        std::vector<uint32_t> spec(kConvCompactSpecSlots);
        spec[kConvCompactOcbSpecIndex]       = ocbBlocks;
        spec[kConvCompactPixelTileSpecIndex] = pixelTile;
        spec[kConvCompactStrideSpecIndex]    = strideW;
        spec[kConvCompactCinSpecIndex]       = inputChannels;
        spec[kConvCompactLaneWidthSpecIndex] = laneWidth;
        return spec;
    }

    /// Specialization-constant slots of the depthwise 3x3 row-halo kernel
    /// (shaders/dwconv3x3_row*.comp): 0 = WTILE, 1 = SW_SPEC, 2 = local_size_x_id.
    constexpr size_t kDwRowPixelTileSpecIndex = 0;
    constexpr size_t kDwRowStrideSpecIndex    = 1;
    constexpr size_t kDwRowLaneWidthSpecIndex = 2;
    constexpr size_t kDwRowSpecSlots          = 3;

    inline std::vector<uint32_t> dwRowSpecConstants(uint32_t pixelTile, uint32_t strideW, uint32_t laneWidth) {
        std::vector<uint32_t> spec(kDwRowSpecSlots);
        spec[kDwRowPixelTileSpecIndex] = pixelTile;
        spec[kDwRowStrideSpecIndex]    = strideW;
        spec[kDwRowLaneWidthSpecIndex] = laneWidth;
        return spec;
    }

    /// Threads per OC-split slice: the flat gid range divided over `parts`, rounded up to whole
    /// workgroups of the DISPATCH width. Whole-workgroup slices are what makes the slices disjoint -
    /// a non-final slice then dispatches exactly its own range, so none of its padding lanes reach
    /// the next slice's outputs and the replay needs no barrier between dispatches. The final
    /// slice's padding is cut by the kernel's total bound.
    constexpr int64_t convOcSplitSliceThreads(int64_t total, int64_t parts, int64_t laneWidth) {
        return parts > 0 ? convDispatchLanes((total + parts - 1) / parts, laneWidth) : total;
    }

    /// Workgroups one OC-split slice dispatches - the count each slice must keep under the device's
    /// X group limit, since a 2-D-spilled slice would run into its neighbour's range.
    constexpr int64_t convOcSplitSliceGroups(int64_t total, int64_t parts, int64_t laneWidth) {
        return convDispatchGroups(convOcSplitSliceThreads(total, parts, laneWidth), laneWidth);
    }

    /// The Winograd kernel variant a shape runs, from the Hint::WinogradVariant value (Mode
    /// TiledGemm / Fused / FusedSplit / FullyFused / SubgroupGemm) and the input depth. The
    /// fully-fused kernel stages every input channel-block's transformed input in shared memory, so
    /// its LDS array bounds the depth it can serve; a deeper shape resolves to the default
    /// tiled-GEMM 3-pass, the variant every other refusal in the op falls back to.
    constexpr int resolveWinogradVariant(int hintVariant, int64_t cinBlocks, int64_t fullVariantMaxCinBlocks) {
        return (hintVariant == (int) Mode::FullyFused && cinBlocks > fullVariantMaxCinBlocks) ? (int) Mode::TiledGemm : hintVariant;
    }

} // namespace vknn
