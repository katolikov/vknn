// Per-run execution context shared with operators.
#pragma once
#include "vknn/config.h"
#include "vknn/graph.h"
#include "vknn/tensor.h"
#include <vector>

namespace vknn {

    class Profiler;

    /// Per-run execution context handed to every operator's run(). Bundles the four pieces of
    /// per-run state an operator needs — the runtime tensor pool, the graph IR, the run config, and
    /// the optional profiler — into one argument.
    ///
    /// All members are non-owning views into Session-owned state. They are valid only for the
    /// duration of a single run() and must not be retained past it; the Session owns the backing
    /// objects and outlives the context.
    struct ExecContext {
        std::vector<RtTensor> *pool     = nullptr; ///< Runtime tensor pool, indexed by TensorId (see t()).
        const Graph           *graph    = nullptr; ///< The graph IR being executed (tensor descs, nodes, initializers).
        const Config          *config   = nullptr; ///< The run configuration (read-only during a run).
        Profiler              *profiler = nullptr; ///< Per-op timing sink, or nullptr when profiling is off.
        /// The runtime tensor for `id`. Precondition: `pool` is bound and `id` is a valid index into it.
        RtTensor &t(TensorId id) {
            return (*pool)[id];
        }
    };

} // namespace vknn
