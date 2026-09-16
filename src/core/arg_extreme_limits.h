// ArgMax / ArgMin GPU kernel limits (shaders/arg_extreme.comp). The Vulkan gate (core/vk_gates.cpp)
// and the kernel's plan (backend/vulkan/ops/arg_extreme_plan.h) both decide through
// argExtremeGpuGeometryRefusal, so every node the gate admits is one the plan accepts, and a geometry
// past a limit stays on the CPU op, which addresses the data and stores the indices in int64. Free of
// Vulkan types so core includes it.
#pragma once
#include "vknn/shape.h"
#include <cstdint>
#include <limits>

namespace vknn {

    /// Largest index the fp32 indices buffer stores exactly: every integer up to 2^24 is exact in fp32,
    /// so an axis of up to 2^24 + 1 elements (indices 0..2^24) is accepted.
    inline constexpr int64_t kArgExtremeMaxExactFp32Index = int64_t(1) << 24;
    /// Most data elements the shader addresses: it computes data and indices offsets in int32.
    inline constexpr int64_t kArgExtremeMaxShaderElements = std::numeric_limits<int32_t>::max();

    /// The [outer, extent, inner] view of a data shape around its selected axis.
    struct ArgExtremeGeometry {
        int64_t outer  = 1; // product of the dims before the axis
        int64_t extent = 0; // axis length
        int64_t inner  = 1; // product of the dims after the axis
    };

    /// Geometry of `shape` around `axis`. Preconditions: `shape` is non-empty and `axis` is normalized
    /// into [0, rank).
    inline ArgExtremeGeometry argExtremeGeometry(const Shape &shape, int64_t axis) {
        ArgExtremeGeometry geometry;
        geometry.extent = shape[(size_t) axis];
        for (int64_t d = 0; d < axis; ++d)
        {
            geometry.outer *= shape[(size_t) d];
        }
        for (int64_t d = axis + 1; d < (int64_t) shape.size(); ++d)
        {
            geometry.inner *= shape[(size_t) d];
        }
        return geometry;
    }

    /// Why the GPU kernel cannot compute `geometry` exactly, or nullptr when it can. The product of the
    /// non-zero factors must fit int32: it bounds every data and indices offset and every push-constant
    /// value (outer, extent, inner, total), including those of an empty geometry. The product is
    /// bounded factor by factor, so its computation never overflows.
    inline const char *argExtremeGpuGeometryRefusal(const ArgExtremeGeometry &geometry) {
        const int64_t factors[] = {geometry.outer, geometry.extent, geometry.inner};
        int64_t       product   = 1;
        for (int64_t factor: factors)
        {
            if (factor < 0)
            {
                return "the data shape has an unresolved dimension";
            }
            if (factor == 0)
            {
                continue;
            }
            if (product > kArgExtremeMaxShaderElements / factor)
            {
                return "data elements exceed the kernel's int32 addressing";
            }
            product *= factor;
        }
        if (geometry.extent - 1 > kArgExtremeMaxExactFp32Index)
        {
            return "axis extent has indices beyond the exact fp32 integer range";
        }
        return nullptr;
    }

} // namespace vknn
