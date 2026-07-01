// Abstract backend interface.
#pragma once
#include "vknn/config.h"
#include "vknn/exec_context.h"
#include "vknn/graph.h"
#include "vknn/tensor.h"
#include <memory>
#include <vector>

namespace vknn {

    class Segment;

    /// Abstract backend. Subclass + register to add a backend (see docs/ADDING_A_BACKEND.md).
    class Backend {
      public:
        virtual ~Backend()               = default;
        virtual BackendKind kind() const = 0;
        virtual const char *name() const = 0;
        /// Whether the backend is usable on this device (false => skip in selection).
        virtual bool available() const = 0;
        /// Apply session Config to the backend before planning (e.g. the debug op-disable list). Default
        /// no-op. Called once per session create, before supportsNode() is used for assignment.
        virtual void configure(const Config &cfg) {
        }
        /// Capability query used for per-op backend assignment / fallback decisions.
        virtual bool supports(OpType t, DType dt) const = 0;
        /// Shape-aware capability query. Defaults to the type-only check; backends override this when
        /// support depends on the node's attributes/shapes (e.g. Concat axis, broadcast layout).
        virtual bool supportsNode(const Graph &g, const Node &nd, DType dt) const {
            return supports(nd.type, dt);
        }

        /// Ensure tensor `rt` has valid host data (NCHW canonical). Default: assume host already valid.
        virtual void toHost(RtTensor &rt, ExecContext &ctx) {
        }
        /// Ensure tensor `rt` is resident on this backend (e.g. uploaded+packed). Default: no-op.
        virtual void toDevice(RtTensor &rt, ExecContext &ctx) {
        }

        /// Compile a contiguous run of nodes (indices into graph.nodes) into a Segment.
        virtual std::unique_ptr<Segment> compileSegment(const std::vector<int> &nodeIdx, Graph &g, const Config &cfg) = 0;

        /// Called once after all segments are compiled (flush pipeline/weight/tuning caches to disk).
        virtual void finalize() {
        }
    };

} // namespace vknn
