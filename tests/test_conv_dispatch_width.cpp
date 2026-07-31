// Workgroup-width contracts of the Vulkan conv op (src/core/conv_dispatch_rules.h, consumed by
// src/backend/vulkan/ops/conv.cpp). The op resolves the per-thread conv family's width from device
// caps at load (flat::laneWidthFor over flat::kConvFamilyLaneWidth), so every rule that derives a
// spec vector, a group count, an OC-split slice or a race wave count from that width has to hold at
// every width a subgroup can make it return - not only at the 64 of a wave-64 GPU.
//
// What these tests can and cannot prove: they are host tests, so they prove the HOST rules. That a
// pipeline compiled at width W and dispatched at width W writes every output is a GPU statement and
// needs a device gate on a device whose subgroupSize is not 64.
#include "core/conv_dispatch_rules.h"
#include "vknn/hint.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

using namespace vknn;

namespace {

    // Every workgroup width the conv family can resolve to: laneWidthFor rounds the family ceiling
    // down to whole subgroups and floors at one whole subgroup, so a wave-32 device gives 64 (two
    // subgroups under the 64 ceiling), a wave-64 device 64, and a wave-128 device the full 128 -
    // plus the narrow widths a caps-constrained device or the Heavy race's half-subgroup candidate
    // can produce.
    const std::vector<int64_t> kLaneWidths = {32, 64, 128, 256};

    // Independent reference walk of an OC-split replay: the lanes each slice dispatch actually
    // launches, derived from the dispatch loop in ConvOp::record (slice bases stepping by the slice
    // size, each dispatch covering whole workgroups of the dispatch width) rather than from the
    // rule under test.
    struct SliceLaunch {
        int64_t base, launchedLanes;
    };
    std::vector<SliceLaunch> replaySlices(int64_t total, int64_t parts, int64_t laneWidth) {
        std::vector<SliceLaunch> launched;
        const int64_t            sliceThreads = convOcSplitSliceThreads(total, parts, laneWidth);
        for (int64_t base = 0; base < total; base += sliceThreads)
        {
            const int64_t want = std::min(sliceThreads, total - base);
            launched.push_back({base, convDispatchGroups(want, laneWidth) * laneWidth});
        }
        return launched;
    }

} // namespace

// --- OC-split slice alignment -------------------------------------------------------------------

// The disjointness the barrier-free back-to-back replay rests on: a non-final slice must dispatch
// EXACTLY its own range, so no padding lane of one slice computes an output that belongs to the
// next slice (which is running concurrently, with no barrier between them).
TEST(ConvDispatchWidth, OcSplitSlicesNeverOverlapAtAnyLaneWidth) {
    for (int64_t laneWidth: kLaneWidths)
    {
        for (int parts: {2, 4})
        {
            for (int64_t total = 1; total <= 4096; ++total)
            {
                const auto slices = replaySlices(total, parts, laneWidth);
                ASSERT_FALSE(slices.empty()) << "total=" << total;
                for (size_t s = 0; s + 1 < slices.size(); ++s)
                {
                    const int64_t launchedEnd = slices[s].base + slices[s].launchedLanes;
                    EXPECT_LE(launchedEnd, slices[s + 1].base) << "slice " << s << " of " << slices.size() << " overruns its successor: laneWidth=" << laneWidth << " parts=" << parts << " total=" << total;
                }
                // The last slice may pad past the range end - the kernel's total bound retires
                // those lanes - but the replay must still cover every thread of the range.
                const int64_t coveredEnd = slices.back().base + slices.back().launchedLanes;
                EXPECT_GE(coveredEnd, total) << "laneWidth=" << laneWidth << " parts=" << parts << " total=" << total;
            }
        }
    }
}

// The same property stated as the invariant it comes from: the slice size is a whole number of
// dispatch workgroups. Rounding to a fixed 64 while the dispatch divides by the device width is
// exactly what breaks it.
TEST(ConvDispatchWidth, OcSplitSliceIsWholeWorkgroupsOfTheDispatchWidth) {
    for (int64_t laneWidth: kLaneWidths)
    {
        for (int parts: {2, 4})
        {
            for (int64_t total: {1, 63, 64, 65, 192, 384, 1000, 4097, 100000})
            {
                const int64_t sliceThreads = convOcSplitSliceThreads(total, parts, laneWidth);
                EXPECT_EQ(sliceThreads % laneWidth, 0) << "laneWidth=" << laneWidth << " parts=" << parts << " total=" << total;
                EXPECT_GE(sliceThreads * parts, total) << "laneWidth=" << laneWidth << " parts=" << parts << " total=" << total;
            }
        }
    }
}

// The reported case: a 192-thread slice on a wave-128 device. Aligning to 64 yields 192, which
// dispatches ceil(192/128) = 2 workgroups = 256 lanes and reaches 64 threads into the next slice.
TEST(ConvDispatchWidth, OcSplitSliceOnWave128DeviceDispatchesExactlyItsRange) {
    constexpr int64_t kWave128Width = 128;
    constexpr int64_t kTotalThreads = 384;
    constexpr int     kParts        = 2;
    const int64_t     sliceThreads  = convOcSplitSliceThreads(kTotalThreads, kParts, kWave128Width);
    EXPECT_EQ(sliceThreads, 256);
    EXPECT_EQ(convDispatchLanes(sliceThreads, kWave128Width), sliceThreads);
    const auto slices = replaySlices(kTotalThreads, kParts, kWave128Width);
    ASSERT_EQ(slices.size(), 2u);
    EXPECT_EQ(slices[0].base + slices[0].launchedLanes, slices[1].base);
}

// Every output of the range is computed by exactly one slice's lanes: the union of the launched
// lane ranges, intersected with the range, is a partition.
TEST(ConvDispatchWidth, OcSplitReplayComputesEveryOutputExactlyOnce) {
    for (int64_t laneWidth: kLaneWidths)
    {
        for (int parts: {2, 4})
        {
            for (int64_t total: {1, 64, 65, 192, 384, 385, 1023, 1024, 4097})
            {
                std::vector<int> computedBy((size_t) total, 0);
                for (const SliceLaunch &slice: replaySlices(total, parts, laneWidth))
                {
                    for (int64_t lane = slice.base; lane < slice.base + slice.launchedLanes && lane < total; ++lane)
                    {
                        ++computedBy[(size_t) lane];
                    }
                }
                for (int64_t lane = 0; lane < total; ++lane)
                {
                    EXPECT_EQ(computedBy[(size_t) lane], 1) << "flat gid " << lane << " computed " << computedBy[(size_t) lane] << " times: laneWidth=" << laneWidth << " parts=" << parts << " total=" << total;
                }
            }
        }
    }
}

// The eligibility gate counts the workgroups one slice dispatches, so it must count them at the
// dispatch width too.
TEST(ConvDispatchWidth, OcSplitSliceGroupsMatchTheSliceDispatch) {
    for (int64_t laneWidth: kLaneWidths)
    {
        for (int parts: {2, 4})
        {
            for (int64_t total: {64, 192, 384, 5000, 1 << 20})
            {
                const int64_t sliceThreads = convOcSplitSliceThreads(total, parts, laneWidth);
                EXPECT_EQ(convOcSplitSliceGroups(total, parts, laneWidth), convDispatchGroups(sliceThreads, laneWidth));
                EXPECT_EQ(convOcSplitSliceGroups(total, parts, laneWidth), sliceThreads / laneWidth);
            }
        }
    }
}

// --- conv_reg specialization constants -----------------------------------------------------------

// conv_reg declares its workgroup width as a specialization constant with a 64 default, so a spec
// vector that omits the slot compiles the pipeline 64 lanes wide whatever the device resolved. The
// builder always fills every slot, so an omitted width cannot occur.
TEST(ConvDispatchWidth, ConvRegSpecAlwaysCarriesTheLaneWidth) {
    for (int64_t laneWidth: kLaneWidths)
    {
        for (uint32_t ocbBlocks: {1u, 2u, 3u})
        {
            for (uint32_t pixelTile: {4u, 8u})
            {
                const std::vector<uint32_t> spec = convRegSpecConstants(ocbBlocks, pixelTile, (uint32_t) laneWidth);
                ASSERT_EQ(spec.size(), kConvRegSpecSlots);
                EXPECT_EQ(spec[kConvRegOcbSpecIndex], ocbBlocks);
                EXPECT_EQ(spec[kConvRegPixelTileSpecIndex], pixelTile);
                EXPECT_EQ(spec[kConvRegLaneWidthSpecIndex], (uint32_t) laneWidth);
            }
        }
    }
}

// The forced register-tiled path (Hint::DirectConv3x3 = Mode::RegisterTiled) computes its thread
// count at the kernel's declared tile, so it must still state that tile AND the width explicitly:
// the pipeline's width is the one its dispatch divides by.
TEST(ConvDispatchWidth, ForcedConvRegPipelineWidthMatchesItsDispatchWidth) {
    for (int64_t laneWidth: kLaneWidths)
    {
        const std::vector<uint32_t> spec    = convRegSpecConstants(kConvRegDefaultOcbBlocks, kConvRegDefaultPixelTile, (uint32_t) laneWidth);
        const int64_t               threads = 1 * 16 * ((56 * 56 + kConvRegDefaultPixelTile - 1) / kConvRegDefaultPixelTile);
        EXPECT_EQ(spec[kConvRegLaneWidthSpecIndex], (uint32_t) laneWidth);
        // Lanes launched at the compiled width cover every thread the tile needs.
        EXPECT_GE(convDispatchGroups(threads, spec[kConvRegLaneWidthSpecIndex]) * (int64_t) spec[kConvRegLaneWidthSpecIndex], threads);
        EXPECT_EQ(convDispatchGroups(threads, laneWidth), convDispatchGroups(threads, spec[kConvRegLaneWidthSpecIndex]));
    }
}

// --- race wave counts ----------------------------------------------------------------------------

// KernelCost::waves is "workgroups * localSize / 64". Entrants of one race dispatch at one width,
// so their wave counts have to be derived from that width; deriving one entrant's from 64 while it
// dispatches at 128 double-counts nothing on a wave-64 device and misreports on every other.
TEST(ConvDispatchWidth, DispatchWavesCountLaunchedLanes) {
    for (int64_t laneWidth: kLaneWidths)
    {
        for (int64_t threads: {1, 63, 64, 65, 16384, 100003})
        {
            const double waves = convDispatchWaves(threads, laneWidth);
            EXPECT_DOUBLE_EQ(waves, (double) (convDispatchGroups(threads, laneWidth) * laneWidth) / (double) kConvCostWaveLanes);
            EXPECT_GE(waves * (double) kConvCostWaveLanes, (double) threads);
        }
    }
}

// A dispatch at a wider workgroup launches at least as many lanes as the thread count asks for, and
// the wave figure is monotone in the thread count at a fixed width - the two properties the
// analytical prefilter's saturation term relies on to rank entrants.
TEST(ConvDispatchWidth, DispatchWavesAreMonotoneInThreadCount) {
    for (int64_t laneWidth: kLaneWidths)
    {
        double previous = 0.0;
        for (int64_t threads = 1; threads <= 8192; threads += 7)
        {
            const double waves = convDispatchWaves(threads, laneWidth);
            EXPECT_GE(waves, previous) << "laneWidth=" << laneWidth << " threads=" << threads;
            previous = waves;
        }
    }
}

// The concrete mixed-basis case: 16384 threads on a wave-128 device. Counting waves at a fixed 64
// reports 256 for a dispatch that launches 128 waves' worth of lanes at the width it actually runs.
TEST(ConvDispatchWidth, DispatchWavesOnWave128DeviceUseTheDispatchWidth) {
    constexpr int64_t kWave128Width = 128;
    constexpr int64_t kTinyThreads  = 16384;
    EXPECT_DOUBLE_EQ(convDispatchWaves(kTinyThreads, kWave128Width), 256.0);
    EXPECT_EQ(convDispatchGroups(kTinyThreads, kWave128Width), 128);
    // The same thread count at the family's 64-wide width: the same launched lanes, so the two
    // widths only disagree once a dispatch pads (below), never on a fully packed range.
    EXPECT_DOUBLE_EQ(convDispatchWaves(kTinyThreads, kConvCostWaveLanes), 256.0);
    // A padding case where the bases genuinely differ: 65 threads launches one 128-lane workgroup
    // (2 waves) at width 128 and two 64-lane workgroups (2 waves) at width 64 - but 129 threads
    // launches 4 waves at width 128 and 3 at width 64.
    EXPECT_DOUBLE_EQ(convDispatchWaves(129, kWave128Width), 4.0);
    EXPECT_DOUBLE_EQ(convDispatchWaves(129, kConvCostWaveLanes), 3.0);
}

// --- Winograd variant resolution -----------------------------------------------------------------

// The fully-fused variant's LDS array bounds the input depth it serves. A refused shape resolves to
// the DEFAULT tiled-GEMM 3-pass; resolving it to the fused variant would put the shape on a kernel
// the op documents as a hint-gated regression that nothing selects automatically.
TEST(ConvDispatchWidth, RefusedFullyFusedWinogradResolvesToTheTiledGemm) {
    constexpr int64_t kFullVariantMaxCinBlocks = 128;
    EXPECT_EQ(resolveWinogradVariant((int) Mode::FullyFused, kFullVariantMaxCinBlocks + 1, kFullVariantMaxCinBlocks), (int) Mode::TiledGemm);
    EXPECT_EQ(resolveWinogradVariant((int) Mode::FullyFused, 130, kFullVariantMaxCinBlocks), (int) Mode::TiledGemm);
}

// An eligible shape keeps the variant it asked for, and no other variant is touched by the depth
// bound (only the fully-fused kernel stages the transformed input in shared memory).
TEST(ConvDispatchWidth, WinogradVariantResolutionLeavesEveryOtherVariantAlone) {
    constexpr int64_t kFullVariantMaxCinBlocks = 128;
    EXPECT_EQ(resolveWinogradVariant((int) Mode::FullyFused, kFullVariantMaxCinBlocks, kFullVariantMaxCinBlocks), (int) Mode::FullyFused);
    EXPECT_EQ(resolveWinogradVariant((int) Mode::FullyFused, 1, kFullVariantMaxCinBlocks), (int) Mode::FullyFused);
    for (Mode variant: {Mode::TiledGemm, Mode::Fused, Mode::FusedSplit, Mode::SubgroupGemm})
    {
        for (int64_t cinBlocks: {1, 128, 129, 4096})
        {
            EXPECT_EQ(resolveWinogradVariant((int) variant, cinBlocks, kFullVariantMaxCinBlocks), (int) variant) << "cinBlocks=" << cinBlocks;
        }
    }
}
