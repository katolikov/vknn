// Kernel choice of the GPU DepthToSpace op (src/backend/vulkan/ops/depth_to_space_plan.h).
//
// The op owns two kernels with DIFFERENT push-constant blocks: the packed NC4HW4 kernel (36 B, no
// thread total) and the flat row-major kernel (40 B, thread total first). prepare() creates the
// pipeline -- and therefore the pipeline layout's push-constant RANGE -- for one of them, and
// record() pushes one of the two blocks. Picking the pipeline and picking the block from different
// facts is a Vulkan API violation the moment the two disagree: vkCmdPushConstants with a size past
// the layout's range is out of range, and the packed kernel reads the flat block's fields at the
// wrong offsets.
//
// The fact that separates them is layout eligibility (depthToSpaceIsNc4), a property of the channel
// counts alone. A lane COUNT is not that fact: it collapses to zero on any zero-extent dimension
// while the node stays packed-eligible.
//
// Scope note: the kernels are compiled only when VKNN_ENABLE_VULKAN is on, so these tests prove the
// host dispatch plan -- which pipeline, how many lanes, how many push-constant bytes -- not the
// kernels themselves. Byte-exactness of the packed remap is a device gate.
#include "backend/vulkan/ops/depth_to_space_plan.h"
#include "vknn/nchw.h"
#include <gtest/gtest.h>

using namespace vknn;

namespace {
    // The lane counts prepare() derives: one lane per output NC4HW4 block-pixel for the packed
    // kernel, one lane per output element for the flat kernel.
    int64_t packedLanes(int64_t n, int64_t outChannels, int64_t outHeight, int64_t outWidth) {
        return n * cBlocks(outChannels) * outHeight * outWidth;
    }
    int64_t flatLanes(int64_t n, int64_t outChannels, int64_t outHeight, int64_t outWidth) {
        return n * outChannels * outHeight * outWidth;
    }
} // namespace

// The defect: a packed-eligible node with a zero-extent dimension. Every lane count is zero, but the
// node is still packed-eligible, so the plan must stay on the packed pipeline and push the packed
// block. Selecting on the lane count instead pushes the 40-byte flat block into the 36-byte packed
// range.
TEST(DepthToSpaceDispatch, ZeroExtentPackedNodeKeepsThePackedBlock) {
    // [1,16,0,5] with blocksize 2 -> [1,4,0,10]: 16 % 4 == 0 and 4 % 4 == 0, so the node is packed.
    const int64_t                  lanes = packedLanes(1, 4, 0, 10);
    const DepthToSpaceDispatchPlan plan  = planDepthToSpaceDispatch(/*packedEligible=*/true, lanes, flatLanes(1, 4, 0, 10));
    EXPECT_EQ(lanes, 0);
    EXPECT_EQ(plan.path, DepthToSpacePath::kPackedNc4);
    EXPECT_EQ(plan.laneCount, 0);
    EXPECT_EQ(plan.pushConstantBytes(), (uint32_t) sizeof(DepthToSpacePackedPC));
    // The two blocks really do differ in size, which is what makes the mismatch an out-of-range push.
    EXPECT_LT(sizeof(DepthToSpacePackedPC), sizeof(DepthToSpaceFlatPC));
}

// Close the class, not the reported shape: every zero-extent combination of a packed-eligible node
// keeps the packed pipeline, including a zero CHANNEL count (0 % 4 == 0 is packed-eligible too).
TEST(DepthToSpaceDispatch, EveryZeroExtentCombinationKeepsItsPipeline) {
    const int64_t extents[] = {0, 1, 5};
    for (int64_t n: extents)
    {
        for (int64_t c: {(int64_t) 0, (int64_t) 4, (int64_t) 8})
        {
            for (int64_t h: extents)
            {
                for (int64_t w: extents)
                {
                    const DepthToSpaceDispatchPlan packed = planDepthToSpaceDispatch(true, packedLanes(n, c, h, w), flatLanes(n, c, h, w));
                    ASSERT_EQ(packed.path, DepthToSpacePath::kPackedNc4) << n << "," << c << "," << h << "," << w;
                    ASSERT_EQ(packed.pushConstantBytes(), (uint32_t) sizeof(DepthToSpacePackedPC));
                    ASSERT_EQ(packed.laneCount, packedLanes(n, c, h, w));

                    const DepthToSpaceDispatchPlan flat = planDepthToSpaceDispatch(false, packedLanes(n, c, h, w), flatLanes(n, c, h, w));
                    ASSERT_EQ(flat.path, DepthToSpacePath::kFlatRowMajor) << n << "," << c << "," << h << "," << w;
                    ASSERT_EQ(flat.pushConstantBytes(), (uint32_t) sizeof(DepthToSpaceFlatPC));
                    ASSERT_EQ(flat.laneCount, flatLanes(n, c, h, w));
                }
            }
        }
    }
}

// A non-degenerate packed node still dispatches one lane per output block-pixel, so the plan did not
// buy its safety by changing what the working shapes run.
TEST(DepthToSpaceDispatch, PopulatedShapesKeepTheirLaneCounts) {
    // [1,16,8,8], blocksize 2 -> [1,4,16,16]: one lane per (n, channel block, oh, ow).
    const DepthToSpaceDispatchPlan packed = planDepthToSpaceDispatch(true, packedLanes(1, 4, 16, 16), flatLanes(1, 4, 16, 16));
    EXPECT_EQ(packed.path, DepthToSpacePath::kPackedNc4);
    EXPECT_EQ(packed.laneCount, 1 * 1 * 16 * 16);
    // [1,18,8,8], blocksize 3 -> [1,2,24,24]: 18 % 4 != 0, so the flat kernel runs one lane per element.
    const DepthToSpaceDispatchPlan flat = planDepthToSpaceDispatch(false, packedLanes(1, 2, 24, 24), flatLanes(1, 2, 24, 24));
    EXPECT_EQ(flat.path, DepthToSpacePath::kFlatRowMajor);
    EXPECT_EQ(flat.laneCount, 1 * 2 * 24 * 24);
}
