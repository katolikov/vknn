// Per-op timing record: CPU wall clock plus GPU timestamp, recorded dispatch count, and IO bytes.
#pragma once
#include "vknn/op.h"
#include <cstdint>
#include <string>

namespace vknn {

    /// One entry in a per-op execution profile: what ran, on which backend, how long it took on the
    /// CPU and GPU, and how much work it moved. The profiler appends one record per executed op.
    struct OpRecord {
        std::string name;                   ///< Op instance name from the graph (unique per node).
        OpType      type = OpType::Unknown; ///< Operator kind; Unknown until the record is filled in.
        std::string backend;                ///< Backend that executed the op (e.g. "vulkan", "cpu").
        double      cpuMs = 0.0;            ///< CPU wall-clock time for the op, in milliseconds.
        double      gpuMs = -1.0;           ///< GPU timestamp-query time in milliseconds; negative means not measured.
        /// Compute dispatches the op recorded. One op is routinely several dispatches (a split-K
        /// GEMM's partial + reduce, Winograd's transform/GEMM/output passes, fused attention's
        /// partial + combine), so this is what a node count understates. 0 on a backend that
        /// dispatches nothing (the CPU oracle) and on an op the planner elided to a zero-copy view.
        uint32_t dispatches = 0;
        int64_t  bytesIO    = 0;     ///< Total input + output bytes touched by the op.
        bool     fellBack   = false; ///< True when the primary backend could not run the op and it fell back to CPU.
    };

} // namespace vknn
