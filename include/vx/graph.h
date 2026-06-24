// vxrt — backend-agnostic graph IR. Canonical tensor layout = NCHW.
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "vx/op.h"
#include "vx/tensor.h"

namespace vx {

class Graph {
 public:
  std::vector<TensorDesc> tensors;            // indexed by TensorId
  std::vector<Node> nodes;                    // topologically ordered after import
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

}  // namespace vx
