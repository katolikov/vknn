// Kernel choice and push-constant blocks of the GPU DepthToSpace op (ops/depth_to_space.cpp).
//
// The op owns two kernels with DIFFERENT push-constant blocks: the packed NC4HW4 kernel, which
// derives its own block counts and so carries no thread total, and the flat row-major kernel, which
// leads with one. prepare() creates the pipeline for one of them -- fixing the pipeline layout's
// push-constant RANGE -- and record() pushes one of the blocks, so BOTH decisions must come from the
// same fact. A push whose size exceeds the layout's range is out of range for vkCmdPushConstants,
// and the fields the kernel then reads are offset into the wrong ones.
//
// That fact is layout eligibility (depthToSpaceIsNc4), a property of the channel counts alone. A
// lane COUNT is not that fact: it collapses to zero on any zero-extent dimension while the node
// stays packed-eligible, which is exactly the case that splits the two decisions apart.
//
// Nothing here includes a Vulkan header, so the plan is host-testable
// (tests/test_depth_to_space_dispatch.cpp).
#pragma once
#include <cstdint>

namespace vknn {

    /// Push-constant block of shaders/flat_depth_to_space.comp; field order and types mirror it.
    struct DepthToSpaceFlatPC {
        int total, N, C, H, W, C2, OH, OW, b, mode;
    };
    /// Push-constant block of shaders/depth_to_space_nc4.comp; field order and types mirror it. The
    /// packed kernel derives its own block counts, so the block carries no thread total.
    struct DepthToSpacePackedPC {
        int N, C, H, W, C2, OH, OW, b, mode;
    };

    /// Which DepthToSpace kernel a node runs.
    enum class DepthToSpacePath {
        kFlatRowMajor, ///< flat_depth_to_space.comp: one lane per output element.
        kPackedNc4,    ///< depth_to_space_nc4.comp: one lane per output NC4HW4 block-pixel.
    };

    /// The kernel a node runs and the lane count that goes with it. The push-constant size follows
    /// the path, so the pipeline's range and the pushed block are the same choice.
    struct DepthToSpaceDispatchPlan {
        DepthToSpacePath path      = DepthToSpacePath::kFlatRowMajor;
        int64_t          laneCount = 0;

        /// Bytes of the block this path pushes; also the range its pipeline is created with.
        uint32_t pushConstantBytes() const {
            return path == DepthToSpacePath::kPackedNc4 ? (uint32_t) sizeof(DepthToSpacePackedPC) : (uint32_t) sizeof(DepthToSpaceFlatPC);
        }
    };

    /// Plan a node's dispatch. `packedEligible` is depthToSpaceIsNc4's verdict; the two lane counts
    /// are the packed kernel's output block-pixel count and the flat kernel's output element count.
    inline DepthToSpaceDispatchPlan planDepthToSpaceDispatch(bool packedEligible, int64_t packedLaneCount, int64_t flatLaneCount) {
        DepthToSpaceDispatchPlan plan;
        plan.path      = packedEligible ? DepthToSpacePath::kPackedNc4 : DepthToSpacePath::kFlatRowMajor;
        plan.laneCount = packedEligible ? packedLaneCount : flatLaneCount;
        return plan;
    }

} // namespace vknn
