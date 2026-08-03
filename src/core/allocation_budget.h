#pragma once
// When a device's ALLOCATION COUNT, not its memory, is the resource about to run out.
//
// Every vk::Buffer owns one vkAllocateMemory (see vk_buffer.cpp), and a graph spends those on
// weights, activation buffers, and each distinct fused-pointwise plan. VkPhysicalDeviceLimits
// caps how many may be live at once -- the Vulkan floor is 4096, which mobile drivers typically
// report verbatim. Exhausting the count fails inside vkAllocateMemory exactly like running out of
// bytes, so without naming the count the diagnosis reads as an out-of-memory error on a device with
// gigabytes free, and the two need different fixes.
//
// The rule lives here rather than in the Vulkan backend because the host build compiles no Vulkan
// sources; keeping it a pure predicate is what makes it testable at all.
#include <cstddef>
#include <cstdint>

namespace vknn {

    /// Percentage of the device's allocation limit past which the live count is worth reporting.
    /// Below a warning threshold there is nothing to act on; at the limit it is already too late.
    constexpr size_t kAllocCountHighWaterPercent = 80;

    /// Is `liveAllocations` at or past kAllocCountHighWaterPercent of `limit`?
    /// A zero `limit` means the device never reported one, and nothing can be said.
    constexpr bool allocationCountNearLimit(size_t liveAllocations, uint32_t limit) {
        return limit != 0 && liveAllocations * 100 >= (size_t) limit * kAllocCountHighWaterPercent;
    }

} // namespace vknn
