// Workgroup width of the LDS tree-fold kernels (src/backend/vulkan/ops/reduction_tree_width.h).
//
// The channel softmax, the reduce partial/combine passes and the norm kernels all finish their
// per-workgroup reduction with the same halving fold:
//
//     for (int s = int(gl_WorkGroupSize.x) / 2; s > 0; s >>= 1) { if (tid < s) red[tid] op= red[tid + s]; barrier(); }
//
// That fold reaches lane 0 from every lane only at a power-of-two width. The family width the
// backend resolves from device caps (VkOpEnv::flatLocalSize) is rounded down to whole SUBGROUPS,
// not to a power of two, so a device whose invocation limit is a non-power-of-two multiple of its
// subgroup size yields a width the fold does not cover -- a partial max and a partial exp-sum, i.e.
// a silently wrong softmax. reductionTreeWidth() is the clamp that closes that gap, and
// halvingFoldCoversEveryLane() is the fold itself, simulated, so the rule is checked rather than
// asserted.
//
// Scope note: the shaders are compiled only when VKNN_ENABLE_VULKAN is on, so these tests prove the
// host WIDTH RULE, not the kernels. That the specialized kernel then reduces correctly at the
// chosen width is a device gate.
#include "backend/vulkan/ops/reduction_tree_width.h"
#include <gtest/gtest.h>

using namespace vknn;

namespace {
    // The family width laneWidthFor() resolves: a ceiling clamped by the device invocation limit and
    // rounded DOWN to whole subgroups (flat_ops.h). Mirrored here so a caps combination can be named
    // without a Vulkan header.
    uint32_t familyWidthFor(uint32_t ceiling, uint32_t maxWorkGroupInvocations, uint32_t subgroupSize) {
        uint32_t width = ceiling;
        if (maxWorkGroupInvocations != 0u && maxWorkGroupInvocations < width)
        {
            width = maxWorkGroupInvocations;
        }
        width = width / subgroupSize * subgroupSize;
        return width != 0u ? width : subgroupSize;
    }
} // namespace

// The defect the clamp exists for: a conformant device reporting a 192-invocation limit with
// 64-wide subgroups resolves a 192-lane family width, and the halving fold leaves the top third of
// the lanes out of lane 0's accumulator.
TEST(ReductionTreeWidth, SubgroupRoundedWidthCanMissLanes) {
    const uint32_t familyWidth = familyWidthFor(/*ceiling=*/256u, /*maxWorkGroupInvocations=*/192u, /*subgroupSize=*/64u);
    EXPECT_EQ(familyWidth, 192u);
    EXPECT_FALSE(flat::halvingFoldCoversEveryLane(familyWidth));
    // Exactly the lanes at and above the first halving's reach are stranded: 192/2 == 96 merges
    // lanes 96..191, leaving 128..191 (a third of the width) never folded in.
    EXPECT_EQ(flat::halvingFoldLanesReachingLaneZero(familyWidth), 128u);
}

// The clamp: the same caps yield a width the fold covers completely.
TEST(ReductionTreeWidth, ClampedWidthCoversEveryLane) {
    const uint32_t familyWidth = familyWidthFor(256u, 192u, 64u);
    const uint32_t treeWidth   = flat::reductionTreeWidth(familyWidth);
    EXPECT_EQ(treeWidth, 128u);
    EXPECT_TRUE(flat::halvingFoldCoversEveryLane(treeWidth));
    EXPECT_EQ(flat::halvingFoldLanesReachingLaneZero(treeWidth), treeWidth);
}

// Close the class, not the reported width: for every family width a device can resolve, the clamped
// width folds every one of its lanes and never exceeds the width it clamps (the shared reduction
// array is sized to the family ceiling, so growing the width would overrun it).
TEST(ReductionTreeWidth, EveryResolvableFamilyWidthClampsToACoveringWidth) {
    for (uint32_t subgroupSize: {1u, 4u, 8u, 16u, 32u, 64u, 128u})
    {
        for (uint32_t limit = 1u; limit <= 1024u; ++limit)
        {
            const uint32_t familyWidth = familyWidthFor(256u, limit, subgroupSize);
            const uint32_t treeWidth   = flat::reductionTreeWidth(familyWidth);
            ASSERT_GE(treeWidth, 1u) << "subgroup " << subgroupSize << " limit " << limit;
            ASSERT_LE(treeWidth, familyWidth) << "subgroup " << subgroupSize << " limit " << limit;
            ASSERT_TRUE(flat::halvingFoldCoversEveryLane(treeWidth)) << "subgroup " << subgroupSize << " limit " << limit << " width " << treeWidth;
        }
    }
}

// The clamp is the identity on a width that is already a power of two, so the devices whose limit is
// 256/128/64 keep the exact width (and the exact fold order) they already run.
TEST(ReductionTreeWidth, PowerOfTwoWidthsAreUnchanged) {
    for (uint32_t width: {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u})
    {
        EXPECT_EQ(flat::reductionTreeWidth(width), width);
        EXPECT_TRUE(flat::halvingFoldCoversEveryLane(width));
    }
}

// The clamp is the largest covering width, never a smaller safe one: halving the workgroup twice
// over would halve the reduction's occupancy for nothing.
TEST(ReductionTreeWidth, ClampIsTheLargestCoveringWidth) {
    for (uint32_t width = 1u; width <= 1024u; ++width)
    {
        const uint32_t treeWidth = flat::reductionTreeWidth(width);
        ASSERT_LE(treeWidth, width) << "width " << width;
        ASSERT_TRUE(flat::halvingFoldCoversEveryLane(treeWidth)) << "width " << width;
        // Nothing between the clamped width and the family width covers its lanes.
        for (uint32_t bigger = treeWidth + 1u; bigger <= width; ++bigger)
        {
            ASSERT_FALSE(flat::halvingFoldCoversEveryLane(bigger)) << "width " << width << " candidate " << bigger;
        }
    }
}
