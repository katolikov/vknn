// Session holds the planned graph, the chosen backends and the caches. Runtime is the thin
// entry point callers actually use.
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "vx/backend.h"
#include "vx/config.h"
#include "vx/graph.h"
#include "vx/profiler.h"
#include "vx/tensor.h"

namespace vx {

/// A named tensor handed in/out of the engine (host side, NCHW canonical, fp32).
struct IOTensor {
  std::string name;
  Shape shape;
  DType dtype = DType::kFloat32;
  std::vector<uint8_t> data;
  float* f32() { return reinterpret_cast<float*>(data.data()); }
  const float* f32() const { return reinterpret_cast<const float*>(data.data()); }
};

/// Owns the planned graph, the chosen backend(s), caches, and the tensor pool.
class Session {
 public:
  ~Session();
  /// Build a session from an ONNX model file.
  static std::unique_ptr<Session> createFromOnnx(const std::string& path, const Config& cfg);
  /// Build from an already-imported graph (testing / surgery).
  static std::unique_ptr<Session> create(Graph&& g, const Config& cfg);

  Status run(const std::vector<IOTensor>& inputs, std::vector<IOTensor>& outputs);

  const Graph& graph() const { return graph_; }
  const Config& config() const { return cfg_; }
  Profiler& profiler() { return profiler_; }
  // Backend assignment per node (for reporting fallbacks).
  std::vector<BackendKind> nodeBackends() const;

  // Per-tensor accessor for layer-dump / debugging (host residency).
  const RtTensor* tensor(const std::string& name) const;

 private:
  Session() = default;
  void plan();  // assign backends, partition into segments, compile
  void reconcileInputs(Segment& seg);

  Graph graph_;
  Config cfg_;
  Profiler profiler_;
  // Declaration order matters for teardown: backends_ (owns the VulkanContext) must be
  // destroyed LAST, after segments_ and pool_ release their device buffers. Members are
  // destroyed in reverse declaration order, so backends_ is declared first here.
  std::vector<std::unique_ptr<Backend>> backends_;  // active, in priority order
  std::map<BackendKind, Backend*> byKind_;
  std::vector<int> nodeBackendIdx_;  // backend index per node
  std::vector<std::unique_ptr<Segment>> segments_;
  std::vector<RtTensor> pool_;
  bool planned_ = false;
};

/// Top-level facade users call.
class Runtime {
 public:
  static std::unique_ptr<Session> load(const std::string& onnxPath, const Config& cfg = {}) {
    return Session::createFromOnnx(onnxPath, cfg);
  }
};

}  // namespace vx
