// ONNX Slice bound resolution, shared by shape inference and both backends.
//
// The three sites that need a Slice's per-axis geometry — inferShapes (output shape), the CPU
// kernel (walk bounds), and the Vulkan flat gather (base offset + signed stride) — must agree
// exactly, or a graph compiles to an output shape its kernels do not fill. Keeping the rule in one
// place is what makes them agree.
//
// ONNX clamps differently per step direction (Slice-13, "Notes"): a forward slice clamps both
// bounds to [0, dim], while a reverse slice clamps `start` to [0, dim-1] and `end` to [-1, dim-1],
// so that end = -1 addresses "past index 0" and index 0 stays reachable.
#pragma once
#include <algorithm>
#include <cstdint>

namespace vknn {

    /// One axis of a resolved Slice: where the walk begins and how many elements it yields. `count`
    /// is the output extent for that axis; the walk visits start, start + step, ... count times.
    struct SliceAxisBounds {
        int64_t start = 0;
        int64_t count = 0;
    };

    /// Resolve one axis of an ONNX Slice against the input extent.
    /// @param dim      The input extent of this axis.
    /// @param rawStart The `starts` entry (negative indexes from the end).
    /// @param rawEnd   The `ends` entry (negative indexes from the end; large values saturate).
    /// @param step     The `steps` entry. Negative walks the axis backwards; zero is malformed ONNX
    ///                 and yields an empty axis rather than a division by zero.
    /// @returns The walk start and the number of elements it yields (0 when the range is empty).
    inline SliceAxisBounds resolveSliceAxis(int64_t dim, int64_t rawStart, int64_t rawEnd, int64_t step) {
        if (dim <= 0 || step == 0)
        {
            return {};
        }
        // A negative bound counts from the end. INT64_MIN would overflow the addition, and it only
        // ever means "before the beginning", so it saturates to the same place the clamps below put it.
        const int64_t kFarNegative = -(dim + 1);
        int64_t       start        = rawStart < 0 ? (rawStart < kFarNegative ? kFarNegative : rawStart + dim) : rawStart;
        int64_t       end          = rawEnd < 0 ? (rawEnd < kFarNegative ? kFarNegative : rawEnd + dim) : rawEnd;
        if (step > 0)
        {
            start = std::max<int64_t>(0, std::min(start, dim));
            end   = std::max<int64_t>(0, std::min(end, dim));
            return {start, std::max<int64_t>(0, (end - start + step - 1) / step)};
        }
        // Reverse: the walk runs start, start-|step|, ... down to (but not including) end.
        start                = std::max<int64_t>(0, std::min(start, dim - 1));
        end                  = std::max<int64_t>(-1, std::min(end, dim - 1));
        const int64_t stride = -step;
        return {start, std::max<int64_t>(0, (start - end + stride - 1) / stride)};
    }

} // namespace vknn
