// The allocation-COUNT high-water rule (src/core/allocation_budget.h).
//
// Every vk::Buffer owns one vkAllocateMemory, so a graph can exhaust the device's allocation count
// while its heap still has gigabytes free -- and vkAllocateMemory reports that the same way it
// reports a byte shortage. The warning that separates the two cases hangs on this predicate, and
// the host build compiles no Vulkan sources, so this is the only place it can be tested.
#include "core/allocation_budget.h"
#include <gtest/gtest.h>

using namespace vknn;

namespace {
    // The Vulkan-guaranteed floor for maxMemoryAllocationCount, which mobile drivers commonly
    // report verbatim.
    constexpr uint32_t kVulkanFloorAllocLimit = 4096;
    // The floor's high-water mark, spelled independently of the predicate's own arithmetic. The
    // exact threshold (4096 * 80 / 100 = 3276.8) is not an integer, and the predicate compares
    // without dividing, so the first count that fires is the CEILING of it -- truncating here would
    // put the expectation one allocation below the boundary.
    constexpr size_t kFloorHighWater = (4096 * kAllocCountHighWaterPercent + 99) / 100;
} // namespace

TEST(AllocationBudget, QuietWellBelowTheLimit) {
    EXPECT_FALSE(allocationCountNearLimit(0, kVulkanFloorAllocLimit));
    EXPECT_FALSE(allocationCountNearLimit(1, kVulkanFloorAllocLimit));
    EXPECT_FALSE(allocationCountNearLimit(kFloorHighWater - 1, kVulkanFloorAllocLimit));
}

TEST(AllocationBudget, FiresAtTheHighWaterMarkAndAbove) {
    // The boundary itself: quiet one allocation below, loud at it.
    EXPECT_FALSE(allocationCountNearLimit(kFloorHighWater - 1, kVulkanFloorAllocLimit));
    EXPECT_TRUE(allocationCountNearLimit(kFloorHighWater, kVulkanFloorAllocLimit));
    EXPECT_TRUE(allocationCountNearLimit(kFloorHighWater + 1, kVulkanFloorAllocLimit));
    EXPECT_TRUE(allocationCountNearLimit(kVulkanFloorAllocLimit, kVulkanFloorAllocLimit));
    // Past the limit the allocation has already failed; the predicate must not go quiet again.
    EXPECT_TRUE(allocationCountNearLimit(kVulkanFloorAllocLimit * 2, kVulkanFloorAllocLimit));
}

// A device that reports no limit is not a device with an infinite one -- there is simply nothing to
// compare against, and a warning naming a limit of zero would be worse than silence.
TEST(AllocationBudget, SaysNothingWhenTheDeviceReportsNoLimit) {
    EXPECT_FALSE(allocationCountNearLimit(0, 0));
    EXPECT_FALSE(allocationCountNearLimit(1u << 20, 0));
}

// The multiplication runs on size_t, so a limit near the 32-bit ceiling must not wrap into a
// spurious warning at a live count of one.
TEST(AllocationBudget, DoesNotWrapOnAHugeLimit) {
    constexpr uint32_t huge = 0xffffffffu;
    EXPECT_FALSE(allocationCountNearLimit(1, huge));
    EXPECT_FALSE(allocationCountNearLimit(huge / 2, huge));
    EXPECT_TRUE(allocationCountNearLimit(huge, huge));
}
