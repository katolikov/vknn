// An executable run of nodes belonging to one backend.
#pragma once
#include "vknn/exec_context.h"
#include "vknn/tensor.h"
#include <vector>

namespace vknn {

    class Backend;

    /// An executable run of nodes belonging to one backend.
    class Segment {
      public:
        virtual ~Segment()                 = default;
        virtual void run(ExecContext &ctx) = 0;
        Backend     *backend               = nullptr;
        bool         isFallback            = false; // this CPU segment exists because the primary backend could not
                                                    // run these ops (drives the fallback warning + profiler tag)
        std::vector<int> nodeIdx;
        // tensor ids this segment consumes from outside / produces for outside (boundary set)
        std::vector<TensorId> boundaryInputs;
        std::vector<TensorId> boundaryOutputs;
    };

} // namespace vknn
