// Per-run execution context shared with operators.
#pragma once
#include "vknn/config.h"
#include "vknn/graph.h"
#include "vknn/tensor.h"
#include <vector>

namespace vknn {

    class Profiler;

    /// Per-run execution context shared with operators.
    struct ExecContext {
        std::vector<RtTensor> *pool     = nullptr; // indexed by TensorId
        const Graph           *graph    = nullptr;
        const Config          *config   = nullptr;
        Profiler              *profiler = nullptr;
        RtTensor              &t(TensorId id) {
            return (*pool)[id];
        }
    };

} // namespace vknn
