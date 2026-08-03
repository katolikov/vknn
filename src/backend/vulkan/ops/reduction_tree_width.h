// Workgroup width of the LDS tree-fold kernels (channel softmax, reduce partial/combine, the norm
// kernels): the width at which their halving fold covers every lane.
//
// The fold is
//     for (int s = int(gl_WorkGroupSize.x) / 2; s > 0; s >>= 1) { if (tid < s) red[tid] op= red[tid + s]; barrier(); }
// which merges lane tid + s into lane tid. At a power-of-two width every lane's value reaches lane 0.
// At any other width the first halving reaches only lanes [s, 2s), so the lanes at and above 2s are
// never folded in and lane 0 holds a PARTIAL reduction -- a silently wrong result, not a crash.
//
// The family width the backend resolves from device caps (VkOpEnv::flatLocalSize) is rounded down to
// whole SUBGROUPS, not to a power of two, so it is not itself a fold width: a device reporting a
// 192-invocation limit with 64-wide subgroups resolves 192. reductionTreeWidth() is the clamp every
// tree-fold pipeline applies to it. flat::laneWidthPow2For (flat_ops.h) applies the identical clamp
// to a caps-derived ceiling; this form clamps a width that is already resolved.
//
// Nothing here includes a Vulkan header, so the width rule and the fold it encodes are host-testable
// (tests/test_reduction_tree_width.cpp).
#pragma once
#include <cstdint>
#include <vector>

namespace vknn { namespace flat {

    /// Largest power of two that fits in `familyWidth`, and at least one lane.
    inline uint32_t reductionTreeWidth(uint32_t familyWidth) {
        uint32_t pow2 = 1u;
        while (pow2 * 2u <= familyWidth)
        {
            pow2 *= 2u;
        }
        return pow2;
    }

    /// Lanes whose value reaches lane 0 when the halving fold runs over `laneCount` lanes. Simulates
    /// the fold itself (each step moves lane tid + stride's accumulated set into lane tid), so it
    /// reports the shaders' real coverage rather than restating the power-of-two rule.
    inline uint32_t halvingFoldLanesReachingLaneZero(uint32_t laneCount) {
        if (laneCount == 0u)
        {
            return 0u;
        }
        std::vector<uint32_t> lanesHeld(laneCount, 1u); // each lane starts holding its own value
        for (uint32_t stride = laneCount / 2u; stride > 0u; stride >>= 1)
        {
            for (uint32_t tid = 0u; tid < stride; ++tid)
            {
                lanesHeld[tid] += lanesHeld[tid + stride];
                lanesHeld[tid + stride] = 0u;
            }
        }
        return lanesHeld[0];
    }

    /// True when the halving fold over `laneCount` lanes reduces every lane into lane 0.
    inline bool halvingFoldCoversEveryLane(uint32_t laneCount) {
        return laneCount != 0u && halvingFoldLanesReachingLaneZero(laneCount) == laneCount;
    }

}} // namespace vknn::flat
