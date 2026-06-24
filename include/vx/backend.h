// vxrt — Backend abstraction, segment execution model, and per-backend op registries.
//
// Execution model: the Session partitions the topo-ordered node list into maximal
// runs ("segments") of nodes assigned to the same backend. Each backend compiles a
// segment into a Segment object (the Vulkan backend pre-records one command buffer per
// segment; the CPU backend just lists ops). At run time, segments execute in order and
// tensor residency is reconciled at segment boundaries (toHost/toDevice), giving
// pre-recorded GPU command buffers + seamless CPU fallback with minimal host<->device sync.
#pragma once
#include <functional>
#include <map>
#include <memory>
#include <vector>
#include "vx/config.h"
#include "vx/graph.h"
#include "vx/tensor.h"

namespace vx {

class Profiler;

/// Per-run execution context shared with operators.
struct ExecContext {
  std::vector<RtTensor>* pool = nullptr;  // indexed by TensorId
  const Graph* graph = nullptr;
  const Config* config = nullptr;
  Profiler* profiler = nullptr;
  RtTensor& t(TensorId id) { return (*pool)[id]; }
};

class Segment;

/// Abstract backend. Subclass + register to add a backend (see docs/ADDING_A_BACKEND.md).
class Backend {
 public:
  virtual ~Backend() = default;
  virtual BackendKind kind() const = 0;
  virtual const char* name() const = 0;
  /// Whether the backend is usable on this device (false => skip in selection).
  virtual bool available() const = 0;
  /// Capability query used for per-op backend assignment / fallback decisions.
  virtual bool supports(OpType t, DType dt) const = 0;

  /// Ensure tensor `rt` has valid host data (NCHW canonical). Default: assume host already valid.
  virtual void toHost(RtTensor& rt, ExecContext& ctx) {}
  /// Ensure tensor `rt` is resident on this backend (e.g. uploaded+packed). Default: no-op.
  virtual void toDevice(RtTensor& rt, ExecContext& ctx) {}

  /// Compile a contiguous run of nodes (indices into graph.nodes) into a Segment.
  virtual std::unique_ptr<Segment> compileSegment(const std::vector<int>& nodeIdx, Graph& g,
                                                  const Config& cfg) = 0;

  /// Called once after all segments are compiled (flush pipeline/weight/tuning caches to disk).
  virtual void finalize() {}
};

/// An executable run of nodes belonging to one backend.
class Segment {
 public:
  virtual ~Segment() = default;
  virtual void run(ExecContext& ctx) = 0;
  Backend* backend = nullptr;
  bool isFallback = false;  // true if this CPU segment exists because the primary backend
                            // could not run these ops (drives the fallback warning + profiler tag)
  std::vector<int> nodeIdx;
  // tensor ids this segment consumes from outside / produces for outside (boundary set)
  std::vector<TensorId> boundaryInputs;
  std::vector<TensorId> boundaryOutputs;
};

// --------------------------- Backend registry ---------------------------
class BackendRegistry {
 public:
  using Factory = std::function<std::unique_ptr<Backend>()>;
  static BackendRegistry& instance();
  void registerBackend(BackendKind k, Factory f);
  bool has(BackendKind k) const;
  std::unique_ptr<Backend> create(BackendKind k) const;

 private:
  std::map<BackendKind, Factory> factories_;
};

struct BackendRegistrar {
  BackendRegistrar(BackendKind k, BackendRegistry::Factory f) {
    BackendRegistry::instance().registerBackend(k, std::move(f));
  }
};
#define VX_REGISTER_BACKEND(KIND, TYPE)                    \
  static ::vx::BackendRegistrar _vx_backend_reg_##TYPE(    \
      KIND, []() -> std::unique_ptr<::vx::Backend> {       \
        return std::unique_ptr<::vx::Backend>(new TYPE()); \
      })

}  // namespace vx
