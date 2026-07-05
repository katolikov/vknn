// Segment: one compiled, executable run of graph nodes assigned to a single backend.
#pragma once
#include "vknn/exec_context.h"
#include "vknn/tensor.h"
#include <vector>

namespace vknn {

    class Backend;

    /// One compiled, executable run of graph nodes assigned to a single Backend. A session's graph is
    /// partitioned into segments (one per contiguous run of same-backend nodes); running the graph
    /// runs its segments in order. Produced by Backend::compileSegment() and owned by the Session.
    class Segment {
      public:
        virtual ~Segment()                 = default;
        /// Execute this segment's nodes against the run's tensor pool. `ctx` is valid only for the
        /// duration of the call and must not be retained past it.
        virtual void run(ExecContext &ctx) = 0;

        /// Backend that compiled and owns this segment. Non-owning; the backend outlives the segment.
        Backend *backend = nullptr;
        /// True when this is a CPU fallback segment that exists because the primary backend cannot run
        /// these ops. Drives the fallback warning and the profiler tag.
        bool isFallback = false;
        /// True when the whole graph runs on this (GPU) backend, so 8-bit image graph-inputs are
        /// uploaded raw and converted on the GPU (no host uint8->fp32->fp16 pack). Off whenever any CPU
        /// segment exists, since a CPU consumer needs the fp32 host copy.
        bool ioGpuConvert = false;
        /// Indices into graph.nodes of the nodes this segment executes, in execution order.
        std::vector<int> nodeIdx;
        /// Boundary set: tensor ids this segment consumes from outside the segment.
        std::vector<TensorId> boundaryInputs;
        /// Boundary set: tensor ids this segment produces for consumers outside the segment.
        std::vector<TensorId> boundaryOutputs;
    };

} // namespace vknn
