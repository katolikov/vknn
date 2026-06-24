#include "vx/graph.h"
#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace vx {

TensorId Graph::find(const std::string& name) const {
  auto it = tensorByName.find(name);
  return it == tensorByName.end() ? kNoTensor : it->second;
}

TensorId Graph::findOrAdd(const std::string& name) {
  auto it = tensorByName.find(name);
  if (it != tensorByName.end()) return it->second;
  TensorDesc d;
  d.name = name;
  return addTensor(std::move(d));
}

TensorId Graph::addTensor(TensorDesc d) {
  TensorId id = (TensorId)tensors.size();
  if (!d.name.empty()) tensorByName[d.name] = id;
  tensors.push_back(std::move(d));
  return id;
}

void Graph::topoSort() {
  // Kahn's algorithm over tensor producers/consumers.
  const size_t n = nodes.size();
  std::vector<int> indeg(n, 0);
  // producer map: tensor id -> node index
  std::unordered_map<int, int> producer;
  for (size_t i = 0; i < n; ++i)
    for (TensorId o : nodes[i].outputs)
      if (o != kNoTensor) producer[o] = (int)i;

  std::vector<std::vector<int>> succ(n);
  for (size_t i = 0; i < n; ++i) {
    std::unordered_set<int> preds;
    for (TensorId in : nodes[i].inputs) {
      auto it = producer.find(in);
      if (it != producer.end() && it->second != (int)i) preds.insert(it->second);
    }
    for (int p : preds) { succ[p].push_back((int)i); indeg[i]++; }
  }
  std::vector<int> q;
  for (size_t i = 0; i < n; ++i)
    if (indeg[i] == 0) q.push_back((int)i);
  std::vector<Node> ordered;
  ordered.reserve(n);
  size_t head = 0;
  while (head < q.size()) {
    int u = q[head++];
    ordered.push_back(nodes[u]);
    for (int v : succ[u])
      if (--indeg[v] == 0) q.push_back(v);
  }
  if (ordered.size() != n)
    throw Error(Status::kInvalidArgument, "graph has a cycle");
  nodes = std::move(ordered);
}

std::string Graph::dump() const {
  std::ostringstream os;
  os << "Graph: " << nodes.size() << " nodes, " << tensors.size() << " tensors, "
     << initializers.size() << " initializers\n";
  for (TensorId i : inputs) os << "  input  " << tensors[i].name << " " << shapeStr(tensors[i].shape) << "\n";
  for (TensorId o : outputs) os << "  output " << tensors[o].name << " " << shapeStr(tensors[o].shape) << "\n";
  for (const auto& nd : nodes) {
    os << "  " << opTypeName(nd.type) << " '" << nd.name << "' (";
    for (size_t k = 0; k < nd.inputs.size(); ++k)
      os << (nd.inputs[k] == kNoTensor ? "<none>" : tensors[nd.inputs[k]].name)
         << (k + 1 < nd.inputs.size() ? "," : "");
    os << ") -> (";
    for (size_t k = 0; k < nd.outputs.size(); ++k)
      os << tensors[nd.outputs[k]].name << (k + 1 < nd.outputs.size() ? "," : "");
    os << ")";
    if (nd.fusedAct != ActType::kNone) os << " +act" << (int)nd.fusedAct;
    os << "\n";
  }
  return os.str();
}

}  // namespace vx
