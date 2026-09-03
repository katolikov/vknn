// Dispatch rules of the fully-connected (Gemm) op, shared by the Vulkan op and the host tests.
#pragma once
#include <cstdint>

namespace vknn {

    /// Output elements (M rows x Cout) at or under which the one-thread-per-output fc kernel is
    /// parallelism-starved and the reduction is split over lanes instead: a 1000-way classifier head
    /// on a 2048-channel pool ran 16 waves over a 4 MB weight read at 23 GB/s. The split kernel puts
    /// kFcSplitLanesK lanes on every output (a fixed-order shared-memory reduce, deterministic), so
    /// the same head dispatches 256 waves. Above the threshold the serial kernel already fills the
    /// device and keeps its single running sum.
    constexpr int64_t kFcSplitMaxOutputs = 16384;
    /// Lanes that split one output's contraction (local_size_x of shaders/fc_split*.comp).
    constexpr uint32_t kFcSplitLanesK = 16;
    /// Outputs per workgroup (local_size_y of shaders/fc_split*.comp); lanes x outputs = 64 threads.
    constexpr uint32_t kFcSplitOutputsPerGroup = 4;

    constexpr bool fcSplitActive(int64_t outputs) {
        return outputs > 0 && outputs <= kFcSplitMaxOutputs;
    }

    /// Workgroups the split kernel dispatches for `outputs` (M x Cout) elements.
    constexpr int64_t fcSplitWorkgroups(int64_t outputs) {
        return (outputs + kFcSplitOutputsPerGroup - 1) / kFcSplitOutputsPerGroup;
    }

} // namespace vknn
