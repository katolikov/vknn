// The graph IR. Backend-agnostic; every tensor here is NCHW.
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vknn/op.h"
#include "vknn/tensor.h"

namespace vknn {

class Graph {
public:
  std::vector<TensorDesc> tensors;  // indexed by TensorId
  std::vector<Node> nodes;          // topologically ordered after import
  std::map<std::string, TensorId> tensorByName;
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;
  // Initializer host data, keyed by tensor id.
  std::map<TensorId, HostBuffer> initializers;

  TensorId findOrAdd(const std::string& name);
  TensorId find(const std::string& name) const;
  TensorId addTensor(TensorDesc d);
  bool isInitializer(TensorId id) const { return initializers.count(id) > 0; }

  const TensorDesc& desc(TensorId id) const { return tensors[id]; }
  TensorDesc& desc(TensorId id) { return tensors[id]; }

  // Topological sort of nodes by tensor dependencies (stable). Throws on cycle.
  void topoSort();
  std::string dump() const;
};

/// Import an ONNX model file into the backend-agnostic IR (canonical NCHW).
Graph importOnnx(const std::string& path);

/// Save/load the optimized graph (post-passes) as a compact self-contained ".vxm" binary, so a
/// reload skips ONNX parsing + all graph passes.
bool saveGraphBin(const Graph& g, const std::string& path);
bool loadGraphBin(Graph& g, const std::string& path);

}  // namespace vknn
