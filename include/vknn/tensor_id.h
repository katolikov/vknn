// Tensor identifier type and the sentinel value that denotes "no tensor".
#pragma once
#include <cstdint>

namespace vknn {

    /// Dense, zero-based handle for a tensor within a graph. Values index the graph's tensor table,
    /// so they are stable only for the lifetime of the graph that issued them. Signed so that
    /// kNoTensor can share the type as an out-of-band sentinel.
    using TensorId = int32_t;

    /// Sentinel TensorId meaning "absent" — an unwired operand or an unset result. Distinct from every
    /// valid handle, which is always non-negative.
    constexpr TensorId kNoTensor = -1;

} // namespace vknn
