// Element-count domain of the flat GPU kernel family (src/backend/vulkan/ops/dispatch_extent.h).
//
// Two rules live here, both about a lane count that does not describe the buffers it will address:
//
//   * sharedElementCapacity(): an element-wise kernel walks a source AND a destination, so its lane
//     count is bounded by the SMALLER allocation. The two are not necessarily equal -- a pooled
//     activation slot keeps the larger-or-equal capacity of the freed slot it reuses -- so a count
//     taken from the destination alone reads past the end of an exactly-sized source. Nothing in the
//     backend enables robustBufferAccess, so that read is undefined behavior on device.
//
//   * dispatchExtentFits(): every flat kernel's element count is an int32 push-constant field and
//     every in-shader index product is an int, so a count past INT32_MAX wraps negative. The kernels
//     then derive SIGNED quantities from that total -- a lane-quad count `(total + QUAD-1) / QUAD`,
//     a mean divisor tested with `rcount > 0` -- and a negative total turns the quad count into a
//     value whose `uint()` is near 2^32, which disables the lane bounds guard outright, and turns
//     the mean divide into a skipped branch. The count is therefore checked before it is narrowed,
//     and an op that cannot express its extent refuses instead of dispatching.
//
// Scope note: the kernels are compiled only when VKNN_ENABLE_VULKAN is on, so these tests prove the
// host rules. That a kernel dispatched inside the domain reads only in-bounds is a device gate.
#include "backend/vulkan/ops/dispatch_extent.h"
#include <gtest/gtest.h>

using namespace vknn;

// The defect: the destination is a pooled slot that is larger than the tensor, the source is an
// exactly-sized buffer, and a lane count taken from the destination walks off the end of the source.
TEST(DispatchExtent, LaneCountFollowsTheSmallerAllocation) {
    const int    elementBytes = 2; // fp16 compute precision
    const size_t exactSource  = 100u * 1024u;
    const size_t pooledDest   = 1024u * 1024u; // best-fit reuse of a larger freed slot

    // The destination alone would dispatch five times the source's elements.
    EXPECT_GT((int64_t) (pooledDest / elementBytes), (int64_t) (exactSource / elementBytes));
    EXPECT_EQ(flat::sharedElementCapacity(exactSource, pooledDest, elementBytes), (int64_t) (exactSource / elementBytes));
    // Symmetric: an oversized SOURCE must not make the kernel write past an exact destination.
    EXPECT_EQ(flat::sharedElementCapacity(pooledDest, exactSource, elementBytes), (int64_t) (exactSource / elementBytes));
}

// Equal allocations (the ordinary case: both buffers sized for the same tensor) keep the full count,
// so the clamp costs the working shapes nothing.
TEST(DispatchExtent, EqualAllocationsKeepEveryElement) {
    for (int elementBytes: {2, 4})
    {
        for (size_t bytes: {(size_t) 0, (size_t) 16, (size_t) 4096, (size_t) 1u << 20})
        {
            ASSERT_EQ(flat::sharedElementCapacity(bytes, bytes, elementBytes), (int64_t) (bytes / (size_t) elementBytes)) << bytes << " / " << elementBytes;
        }
    }
}

// A partial trailing element (an allocation that is not a whole multiple of the element size) is not
// a lane: rounding up would be the same over-read in miniature.
TEST(DispatchExtent, PartialTrailingElementIsNotALane) {
    EXPECT_EQ(flat::sharedElementCapacity(9u, 16u, 4), 2);
    EXPECT_EQ(flat::sharedElementCapacity(3u, 3u, 4), 0);
}

// The int32 element domain: the ceiling is representable, one past it is not, and a negative count
// (an already-wrapped narrowing) never passes.
TEST(DispatchExtent, Int32ElementDomainIsClosedAtBothEnds) {
    EXPECT_EQ(flat::kMaxDispatchElements, (int64_t) INT32_MAX);
    EXPECT_TRUE(flat::dispatchExtentFits(0));
    EXPECT_TRUE(flat::dispatchExtentFits(1));
    EXPECT_TRUE(flat::dispatchExtentFits(flat::kMaxDispatchElements));
    EXPECT_FALSE(flat::dispatchExtentFits(flat::kMaxDispatchElements + 1));
    EXPECT_FALSE(flat::dispatchExtentFits(-1));
}

// Why the domain is checked rather than clamped: past the ceiling the narrowed total goes negative,
// and the kernels derive their bounds and their mean divisor from it in SIGNED arithmetic.
TEST(DispatchExtent, NarrowingPastTheDomainCorruptsDerivedKernelBounds) {
    const int64_t elements = (int64_t) INT32_MAX + 1;
    ASSERT_FALSE(flat::dispatchExtentFits(elements));
    const int narrowed = (int) elements;
    EXPECT_LT(narrowed, 0);

    // The vectorized kernels dispatch lane-quads: `quadTotal = (total + QUAD-1) / QUAD` stays
    // negative, and the guard `if (q >= uint(quadTotal)) return;` then admits every lane in the
    // dispatch -- roughly 2^32 of them -- instead of bounding them.
    const int      quadLanes = 4;
    const int      quadTotal = (narrowed + quadLanes - 1) / quadLanes;
    const uint32_t asGuard   = (uint32_t) quadTotal;
    EXPECT_LT(quadTotal, 0);
    EXPECT_GT(asGuard, (uint32_t) INT32_MAX);

    // The reduce combine pass finalises a Mean only under `if (pc.rcount > 0)`, so a wrapped
    // reduced-element count makes ReduceMean silently return the raw SUM.
    EXPECT_FALSE(narrowed > 0);
}

// A shape whose reduced extent or element count sits inside the domain is accepted, so the gate does
// not refuse anything a device can actually allocate.
TEST(DispatchExtent, RealisticTensorExtentsAreAccepted) {
    const int64_t extents[] = {1, 1000, 1 * 3 * 1024 * 1024, 64LL * 3 * 512 * 512};
    for (int64_t e: extents)
    {
        ASSERT_TRUE(flat::dispatchExtentFits(e)) << e;
    }
}
