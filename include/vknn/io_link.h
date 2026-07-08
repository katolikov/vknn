// Caller-declared linkage between a graph output and a graph input for engine-resident I/O.
#pragma once
#include <cstdint>

namespace vknn {

    /// One contiguous element range copied from a linked output into a linked input at the start of
    /// a run (see Session::linkOutputToInput). Offsets and count are CANONICAL elements — positions
    /// in the tensor's logical NCHW row-major order — independent of the device layout or storage
    /// precision; the engine maps them to the backing layout. A recurrent-state caller (e.g. an
    /// autoregressive decoder's KV cache) updates these per run to place the newly produced rows.
    struct LinkRange {
        int64_t sourceElem = 0; ///< First element read from the linked output.
        int64_t destElem   = 0; ///< First element written in the linked input.
        int64_t count      = 0; ///< Number of elements copied.
    };

} // namespace vknn
