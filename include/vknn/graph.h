// The graph IR. Backend-agnostic; every tensor here is NCHW.
#pragma once
#include "vknn/op.h"
#include "vknn/tensor.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vknn {

    /// A whole model as a backend-agnostic intermediate representation: a flat tensor table, a node
    /// list, and the host data for constant initializers. Tensors are referenced everywhere by
    /// TensorId (an index into `tensors`), never by pointer, so the containers may reallocate freely.
    /// The importer and every graph pass mutate a Graph in place; a backend consumes it read-only.
    class Graph {
      public:
        /// Every tensor's shape/dtype descriptor, indexed by TensorId.
        std::vector<TensorDesc> tensors;
        /// The op instances. Topologically ordered after import (see topoSort()).
        std::vector<Node> nodes;
        /// Name -> id lookup for tensors that carry a name (graph I/O and ONNX-named intermediates).
        std::map<std::string, TensorId> tensorByName;
        /// Ids of the graph's external input tensors, in declaration order.
        std::vector<TensorId> inputs;
        /// Ids of the graph's external output tensors, in declaration order.
        std::vector<TensorId> outputs;
        /// Host-side constant data (weights, biases, shape constants) keyed by the tensor it backs.
        std::map<TensorId, HostBuffer> initializers;

        /// Look up `name`, appending a fresh tensor registered under `name` if none exists yet.
        /// @returns The id of the existing or newly created tensor.
        TensorId findOrAdd(const std::string &name);
        /// Look up an existing tensor by name without creating one.
        /// @returns Its id, or kNoTensor if no tensor is registered under `name`.
        TensorId find(const std::string &name) const;
        /// Append `d` to the tensor table.
        /// @returns The id (table index) of the newly added tensor.
        TensorId addTensor(TensorDesc d);
        /// True when `id` has constant host data in `initializers` (i.e. it is a weight/constant, not
        /// a runtime activation).
        bool isInitializer(TensorId id) const {
            return initializers.count(id) > 0;
        }

        /// The descriptor for `id`. Precondition: `id` is a valid index into `tensors`.
        const TensorDesc &desc(TensorId id) const {
            return tensors[id];
        }
        /// Mutable overload of desc(); same precondition.
        TensorDesc &desc(TensorId id) {
            return tensors[id];
        }

        /// Reorder `nodes` so every node follows its tensor producers (stable: nodes already in
        /// dependency order keep their relative positions). @throws Error if the graph contains a cycle.
        void topoSort();
        /// A readable, multi-line listing of the tensors and nodes, for debugging/logging.
        std::string dump() const;
    };

    /// Import an ONNX model file into the backend-agnostic IR (canonical NCHW).
    Graph importOnnx(const std::string &path);

    /// Save the optimized graph (post-passes) as a compact self-contained ".vxm" binary, so a reload
    /// skips ONNX parsing + all graph passes.
    /// @returns True on success; false if the file cannot be written.
    bool saveGraphBin(const Graph &g, const std::string &path);
    /// Load a ".vxm" binary written by saveGraphBin() into `g`, replacing its contents.
    /// @returns True on success; false if the file is missing or not a valid ".vxm".
    bool loadGraphBin(Graph &g, const std::string &path);

} // namespace vknn
