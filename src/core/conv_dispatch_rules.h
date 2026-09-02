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
#include "vknn/nchw.h"
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
    /// segment), 3 = TILES_PER_WG (column-tiles per workgroup, the wave's 2-D footprint), 4 =
    /// local_size_x_id, the workgroup width. Same contract as conv_reg: the host states every slot,
    /// so the kernel's declared width never stands in for the device-resolved one.
    constexpr size_t kConvRowOcbSpecIndex        = 0;
    constexpr size_t kConvRowPixelTileSpecIndex  = 1;
    constexpr size_t kConvRowStrideSpecIndex     = 2;
    constexpr size_t kConvRowTilesPerWgSpecIndex = 3;
    constexpr size_t kConvRowLaneWidthSpecIndex  = 4;
    constexpr size_t kConvRowSpecSlots           = 5;

    /// The row kernel's wave footprint is stated as output ROWS per workgroup; the kernel takes the
    /// complementary column-tile count, laneWidth / rows, so a footprint is valid on every device
    /// whose width the row count divides. kConvRowFootprintRows lists the raced footprints in the
    /// order their choice codes (kChoiceFootprintShift in the conv op) enumerate them: 4 rows (a
    /// 16-tile-wide patch at 64 lanes; measured best on strided 3x3), 2 rows (32 tiles; best on
    /// stride 1), 1 row (the plain one-row strip).
    constexpr uint32_t kConvRowFootprintRows[]      = {4, 2, 1};
    constexpr size_t   kConvRowFootprintCount       = sizeof(kConvRowFootprintRows) / sizeof(kConvRowFootprintRows[0]);
    constexpr uint32_t kConvRowFootprintStridedCode = 0; // 4 rows: the compact stem's footprint at stride >= 2
    constexpr uint32_t kConvRowFootprintUnitCode    = 1; // 2 rows: the compact stem's footprint at stride 1

    /// Column-tiles per workgroup for a footprint code at a lane width; 0 when the row count does
    /// not divide the width (the caller must not dispatch such a pipeline).
    constexpr uint32_t convRowTilesPerWorkgroup(uint32_t footprintCode, uint32_t laneWidth) {
        return (footprintCode < kConvRowFootprintCount && laneWidth % kConvRowFootprintRows[footprintCode] == 0) ? laneWidth / kConvRowFootprintRows[footprintCode] : 0;
    }

    /// Workgroups the row kernel dispatches for one output map: every (column-tile block,
    /// output-channel block group, row block, batch) tuple is one workgroup of `laneWidth` lanes.
    constexpr int64_t convRowWorkgroups(int64_t batch, int64_t ocbGroups, int64_t outH, int64_t outW, uint32_t pixelTile, uint32_t tilesPerWg, uint32_t laneWidth) {
        return (tilesPerWg == 0 || laneWidth == 0) ? 0 : batch * ocbGroups * ((outH + (laneWidth / tilesPerWg) - 1) / (laneWidth / tilesPerWg)) * (((outW + pixelTile - 1) / pixelTile + tilesPerWg - 1) / tilesPerWg);
    }

    inline std::vector<uint32_t> convRowSpecConstants(uint32_t ocbBlocks, uint32_t pixelTile, uint32_t strideW, uint32_t tilesPerWg, uint32_t laneWidth) {
        std::vector<uint32_t> spec(kConvRowSpecSlots);
        spec[kConvRowOcbSpecIndex]        = ocbBlocks;
        spec[kConvRowPixelTileSpecIndex]  = pixelTile;
        spec[kConvRowStrideSpecIndex]     = strideW;
        spec[kConvRowTilesPerWgSpecIndex] = tilesPerWg;
        spec[kConvRowLaneWidthSpecIndex]  = laneWidth;
        return spec;
    }

    /// Specialization-constant slots of the compact-input 3x3 conv kernel
    /// (shaders/conv3x3_cin_lt4*.comp): the row-halo slots 0..2, then 3 = CIN (the 1..3 input
    /// channels it gathers from the dense plane), 4 = TILES_PER_WG and 5 = local_size_x_id.
    constexpr size_t kConvCompactOcbSpecIndex        = kConvRowOcbSpecIndex;
    constexpr size_t kConvCompactPixelTileSpecIndex  = kConvRowPixelTileSpecIndex;
    constexpr size_t kConvCompactStrideSpecIndex     = kConvRowStrideSpecIndex;
    constexpr size_t kConvCompactCinSpecIndex        = 3;
    constexpr size_t kConvCompactTilesPerWgSpecIndex = 4;
    constexpr size_t kConvCompactLaneWidthSpecIndex  = 5;
    constexpr size_t kConvCompactSpecSlots           = 6;

    inline std::vector<uint32_t> convCompactSpecConstants(uint32_t ocbBlocks, uint32_t pixelTile, uint32_t strideW, uint32_t inputChannels, uint32_t tilesPerWg, uint32_t laneWidth) {
        std::vector<uint32_t> spec(kConvCompactSpecSlots);
        spec[kConvCompactOcbSpecIndex]        = ocbBlocks;
        spec[kConvCompactPixelTileSpecIndex]  = pixelTile;
        spec[kConvCompactStrideSpecIndex]     = strideW;
        spec[kConvCompactCinSpecIndex]        = inputChannels;
        spec[kConvCompactTilesPerWgSpecIndex] = tilesPerWg;
        spec[kConvCompactLaneWidthSpecIndex]  = laneWidth;
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

    /// Cache key and block size of the group-1 conv weight pack (see the pack in ops/conv.cpp): a
    /// [Coutb][Cinb][KH][KW] array of 4x4 blocks, each block four vec4 indexed by input channel over
    /// the block's four output channels - kConvWeightBlockFloats floats per (block pair, tap). The key
    /// names the layout, so a cache written for the earlier output-channel-major pack never aliases.
    constexpr const char *kConvWeightPackKey     = "#wT";
    constexpr int64_t     kConvWeightBlockFloats = 16;

    /// Lanes a tile-per-thread kernel (conv1x1, conv_reg) dispatches when it orders the
    /// output-channel block group INSIDE a workgroup-sized chunk of pixel tiles: whole workgroups
    /// per (chunk, block group, batch), so the last chunk's padding lanes are part of the count and
    /// retire on the kernel's own tile bound.
    constexpr int64_t convChunkedTileLanes(int64_t batch, int64_t ocbGroups, int64_t tiles, int64_t laneWidth) {
        return laneWidth > 0 ? batch * ocbGroups * convDispatchLanes(tiles, laneWidth) : 0;
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

    /// Operand layout of the tiled Winograd GEMM's transformed weights U (shaders/wino_gemm_fp16.comp
    /// and its register twin wino_gemm_reg_fp16.comp; the conv op's "#winoT" pack): per transform
    /// position, output channel-block and input channel-block, kNC4Block consecutive vec4s indexed by
    /// the INPUT channel lane, each one vec4 over the block's kNC4Block OUTPUT channels. The GEMM
    /// contracts one input channel per fma - acc = fma(vec4(v.lane), U[lane], acc) - so the operand it
    /// needs per input channel is that output-channel vector, and this pack hands it over as one
    /// contiguous vec4; a block's whole K row is contiguous ((k, lane) row-major). Output channels
    /// past Cout inside the last block stay zero, so a partial block's stored M lanes are zero.
    constexpr int64_t winoGemmUVec4Index(int64_t position, int64_t outBlock, int64_t inBlock, int64_t inLane, int64_t coutBlocks, int64_t cinBlocks) {
        return ((position * coutBlocks + outBlock) * cinBlocks + inBlock) * kNC4Block + inLane;
    }

    /// vec4 count of that pack: one per (position, output channel-block, input channel-block, input
    /// lane) - the size the host allocates and the bound the shaders' addressing stays under.
    constexpr int64_t winoGemmUVec4Count(int64_t positions, int64_t coutBlocks, int64_t cinBlocks) {
        return positions * coutBlocks * cinBlocks * kNC4Block;
    }

} // namespace vknn
