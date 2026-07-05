// Per-op timing record: CPU wall clock plus GPU timestamp, dispatch dims, and IO bytes.
#pragma once
#include "vknn/op.h"
#include <array>
#include <cstdint>
#include <string>

namespace vknn {

    /// One entry in a per-op execution profile: what ran, on which backend, how long it took on the
    /// CPU and GPU, and how much work it moved. The profiler appends one record per executed op.
    struct OpRecord {
        std::string             name;                       ///< Op instance name from the graph (unique per node).
        OpType                  type = OpType::Unknown;     ///< Operator kind; Unknown until the record is filled in.
        std::string             backend;                    ///< Backend that executed the op (e.g. "vulkan", "cpu").
        double                  cpuMs    = 0.0;             ///< CPU wall-clock time for the dispatch, in milliseconds.
        double                  gpuMs    = -1.0;            ///< GPU timestamp-query time in milliseconds; negative means not measured.
        std::array<uint32_t, 3> dispatch = {0, 0, 0};       ///< Compute dispatch dimensions (workgroup counts x/y/z).
        int64_t                 bytesIO  = 0;               ///< Total input + output bytes touched by the op.
        bool                    fellBack = false;           ///< True when the primary backend could not run the op and it fell back to CPU.
    };

} // namespace vknn
