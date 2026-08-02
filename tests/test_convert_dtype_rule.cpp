// ConvertDtype converts a tensor's STORED footprint, not its logical element count.
//
// The two agree for a flat tensor and diverge for a blocked one, where NC4HW4 pads the channel axis
// to a multiple of kNC4Block. Walking the logical count on a blocked tensor converts a prefix of the
// buffer and leaves the remainder holding source-width bytes; every consumer then reads those at the
// destination width and sees garbage whose magnitude is unrelated to the data -- a wrong-answer class
// that survives at fp32, because it is a footprint bug and not a precision one.
#include "backend/vulkan/ops/convert_dtype_rule.h"
#include <gtest/gtest.h>

using namespace vknn;

TEST(ConvertDtypeRule, FlatTensorConvertsItsLogicalCount) {
    EXPECT_EQ(convertDtypeElemCount({1, 2, 144, 96}, /*gpuFlat=*/true), 1 * 2 * 144 * 96);
    EXPECT_EQ(convertDtypeElemCount({1, 3, 8, 8}, /*gpuFlat=*/true), 1 * 3 * 8 * 8);
    EXPECT_EQ(convertDtypeElemCount({7}, /*gpuFlat=*/true), 7);
}

TEST(ConvertDtypeRule, BlockedTensorConvertsThePaddedFootprint) {
    // A two-channel flow -- the coordinate tensor an optical-flow cone pins to fp32 -- occupies a
    // whole four-lane block, so half its buffer is padding and the logical count covers half of it.
    EXPECT_EQ(convertDtypeElemCount({1, 2, 144, 96}, /*gpuFlat=*/false), 1 * 4 * 144 * 96);
    EXPECT_EQ(convertDtypeElemCount({1, 1, 8, 8}, /*gpuFlat=*/false), 1 * 4 * 8 * 8);
    EXPECT_EQ(convertDtypeElemCount({1, 3, 8, 8}, /*gpuFlat=*/false), 1 * 4 * 8 * 8);
    EXPECT_EQ(convertDtypeElemCount({2, 5, 3, 3}, /*gpuFlat=*/false), 2 * 8 * 3 * 3);
}

TEST(ConvertDtypeRule, BlockedAndFlatAgreeOnlyOnFullBlocks) {
    for (int64_t c = 1; c <= 16; ++c)
    {
        const Shape   s {1, c, 5, 7};
        const int64_t flat = convertDtypeElemCount(s, true), blocked = convertDtypeElemCount(s, false);
        EXPECT_GE(blocked, flat) << "a blocked footprint is never smaller, c=" << c;
        if (c % kNC4Block == 0)
        {
            EXPECT_EQ(blocked, flat) << "a full block needs no padding, c=" << c;
        } else
        {
            EXPECT_GT(blocked, flat) << "a partial block pads, c=" << c;
        }
    }
}

TEST(ConvertDtypeRule, RankBelowFourStillCountsItsBlocks) {
    // NCHW::from right-aligns, so a rank-2 [N,C] tensor is C channels at 1x1 and still blocks.
    EXPECT_EQ(convertDtypeElemCount({4, 6}, /*gpuFlat=*/false), 4 * 8 * 1 * 1);
    EXPECT_EQ(convertDtypeElemCount({4, 6}, /*gpuFlat=*/true), 4 * 6);
}
