// Element-count domain of the flat GPU kernel family, and the buffer pair a lane count must fit.
//
// Extent domain. Every flat kernel's element count is an `int` push-constant field and every
// in-shader index/stride product is an `int`, so a count past INT32_MAX wraps negative. A negative
// total does not merely under-dispatch: the kernels derive their bounds from it in SIGNED
// arithmetic, and `(total + QUAD-1) / QUAD` stays negative, so the vectorized guard
// `if (q >= uint(quadTotal)) return;` compares against a value near 2^32 and admits every lane in
// the dispatch. The same wrap turns the reduce combine pass's `if (pc.rcount > 0)` mean divide into
// a skipped branch. The count is therefore checked on the host before it is narrowed, and an op
// whose extent does not fit refuses instead of dispatching an out-of-range kernel.
//
// Buffer pair. An element-wise kernel walks a source and a destination, so its lane count is bounded
// by the SMALLER of the two allocations. The two are not necessarily equal: a pooled activation slot
// keeps the larger-or-equal capacity of the freed slot it reuses (vk_segment.cpp's best-fit reuse),
// so a lane count taken from one buffer's capacity alone over-reads or over-writes the other. No
// backend pipeline enables robustBufferAccess, so such an access is undefined behavior on device.
//
// Nothing here includes a Vulkan header, so both rules are host-testable
// (tests/test_dispatch_extent.cpp).
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace vknn { namespace flat {

    /// Largest element count a flat kernel's int32 push-constant field and index math can express.
    constexpr int64_t kMaxDispatchElements = INT32_MAX;

    /// True when `elements` is a non-negative count the flat kernels can address.
    inline bool dispatchExtentFits(int64_t elements) {
        return elements >= 0 && elements <= kMaxDispatchElements;
    }

    /// Refusal text for an op whose extent leaves the domain. `opLabel` names the node, `quantity`
    /// names which count overflowed (an op carries several).
    inline std::string dispatchExtentRefusal(const std::string &opLabel, const char *quantity, int64_t extent) {
        return opLabel + ": " + quantity + " " + std::to_string(extent) + " exceeds the flat kernel family's int32 element domain (" + std::to_string(kMaxDispatchElements) + ")";
    }

    /// Whole elements of `elementBytes` that BOTH allocations can address: the smaller allocation's
    /// element count. A partial trailing element is not a lane.
    inline int64_t sharedElementCapacity(size_t sourceBytes, size_t destBytes, int elementBytes) {
        if (elementBytes <= 0)
        {
            return 0;
        }
        const size_t bytes = sourceBytes < destBytes ? sourceBytes : destBytes;
        return (int64_t) (bytes / (size_t) elementBytes);
    }

}} // namespace vknn::flat
