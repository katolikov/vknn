#include "vx/session.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <set>
#include <sys/stat.h>
#include "vx/logging.h"
#include "../import/passes.h"

namespace vx {

Session::~Session() = default;

std::unique_ptr<Session> Session::createFromOnnx(const std::string& path, const Config& cfg) {
  Graph g = importOnnx(path);
  return create(std::move(g), cfg);
}

std::unique_ptr<Session> Session::create(Graph&& g, const Config& cfg) {
  auto s = std::unique_ptr<Session>(new Session());
  s->graph_ = std::move(g);
  s->cfg_ = cfg;
  cfg.applyLogLevel();
  s->profiler_.setEnabled(cfg.profile);
  auto t0 = std::chrono::high_resolution_clock::now();
  s->plan();
  auto t1 = std::chrono::high_resolution_clock::now();
  VX_INFO << "Session created in "
          << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms";
  return s;
}

void Session::plan() {
  // --- graph optimization passes (NCHW IR, static batch=1) ---
  runStandardPasses(graph_, 1);
  graph_.topoSort();

  // --- instantiate backends in priority order: primary, fallbacks..., CPU last ---
  std::vector<BackendKind> order;
  order.push_back(cfg_.backend);
  for (auto k : cfg_.fallback) order.push_back(k);
  if (cfg_.allowCpuFallback) order.push_back(BackendKind::kCpu);
  std::set<BackendKind> seen;
  auto& reg = BackendRegistry::instance();
  for (BackendKind k : order) {
    if (seen.count(k)) continue;
    seen.insert(k);
    if (!reg.has(k)) { VX_DEBUG << "backend " << backendName(k) << " not registered"; continue; }
    auto b = reg.create(k);
    if (!b || !b->available()) {
      VX_WARN << "backend " << backendName(k) << " unavailable; skipping";
      continue;
    }
    byKind_[k] = b.get();
    backends_.push_back(std::move(b));
  }
  if (backends_.empty()) throw Error(Status::kRuntimeError, "no usable backend");
  VX_INFO << "Active backends (priority): " << [&] {
    std::string s;
    for (auto& b : backends_) s += std::string(b->name()) + " ";
    return s;
  }();

  // --- init tensor pool, load initializers ---
  pool_.resize(graph_.tensors.size());
  for (size_t i = 0; i < pool_.size(); ++i) {
    pool_[i].id = (TensorId)i;
    pool_[i].shape = graph_.tensors[i].shape;
    pool_[i].dtype = graph_.tensors[i].dtype;
  }
  for (auto& kv : graph_.initializers) {
    RtTensor& rt = pool_[kv.first];
    rt.host = kv.second;
    rt.hostValid = true;
    rt.shape = graph_.tensors[kv.first].shape;
    rt.dtype = graph_.tensors[kv.first].dtype;
  }

  // --- per-node backend assignment (highest-priority backend that supports it) ---
  nodeBackendIdx_.assign(graph_.nodes.size(), -1);
  for (size_t n = 0; n < graph_.nodes.size(); ++n) {
    const Node& nd = graph_.nodes[n];
    DType dt = DType::kFloat32;  // compute dtype at IR level
    int chosen = -1;
    for (size_t bi = 0; bi < backends_.size(); ++bi) {
      if (backends_[bi]->supports(nd.type, dt)) { chosen = (int)bi; break; }
    }
    if (chosen < 0) throw Error(Status::kUnsupported,
        std::string("no backend supports op ") + opTypeName(nd.type) + " (" + nd.name + ")");
    nodeBackendIdx_[n] = chosen;
    // warn if the primary backend couldn't take it
    if (backends_[chosen]->kind() != cfg_.backend &&
        byKind_.count(cfg_.backend) &&
        !byKind_[cfg_.backend]->supports(nd.type, dt)) {
      VX_WARN_THROTTLE(std::string("fallback_") + opTypeName(nd.type), 2)
          << "op " << opTypeName(nd.type) << " (" << nd.name << ") not supported by "
          << backendName(cfg_.backend) << " backend -> falling back to "
          << backends_[chosen]->name() << ". Perf note: this op does not run on the requested backend.";
    }
  }

  // --- partition into maximal same-backend segments ---
  std::vector<std::vector<int>> parts;
  for (size_t n = 0; n < graph_.nodes.size(); ++n) {
    if (parts.empty() || nodeBackendIdx_[n] != nodeBackendIdx_[parts.back().front()])
      parts.push_back({});
    parts.back().push_back((int)n);
  }

  // boundary-set computation: producer map
  std::vector<int> producerSeg(graph_.tensors.size(), -1);
  std::vector<int> nodeToSeg(graph_.nodes.size(), -1);
  for (size_t p = 0; p < parts.size(); ++p)
    for (int ni : parts[p]) {
      nodeToSeg[ni] = (int)p;
      for (TensorId o : graph_.nodes[ni].outputs)
        if (o != kNoTensor) producerSeg[o] = (int)p;
    }
  std::set<TensorId> graphOutputs(graph_.outputs.begin(), graph_.outputs.end());

  for (size_t p = 0; p < parts.size(); ++p) {
    int bi = nodeBackendIdx_[parts[p].front()];
    auto seg = backends_[bi]->compileSegment(parts[p], graph_, cfg_);
    std::set<TensorId> ins, outs;
    std::set<TensorId> internalOut;
    for (int ni : parts[p])
      for (TensorId o : graph_.nodes[ni].outputs) internalOut.insert(o);
    for (int ni : parts[p]) {
      for (TensorId in : graph_.nodes[ni].inputs) {
        if (in == kNoTensor) continue;
        if (!internalOut.count(in)) ins.insert(in);  // produced outside (init/input/other seg)
      }
      for (TensorId o : graph_.nodes[ni].outputs) {
        if (o == kNoTensor) continue;
        // consumed outside this segment?
        bool external = graphOutputs.count(o) > 0;
        if (!external)
          for (size_t q = 0; q < graph_.nodes.size() && !external; ++q)
            if (nodeToSeg[q] != (int)p)
              for (TensorId x : graph_.nodes[q].inputs)
                if (x == o) { external = true; break; }
        if (external) outs.insert(o);
      }
    }
    seg->boundaryInputs.assign(ins.begin(), ins.end());
    seg->boundaryOutputs.assign(outs.begin(), outs.end());
    // tag a CPU segment as a fallback when the configured primary backend isn't CPU.
    if (backends_[bi]->kind() == BackendKind::kCpu && cfg_.backend != BackendKind::kCpu)
      seg->isFallback = true;
    segments_.push_back(std::move(seg));
  }
  for (auto& b : backends_) b->finalize();  // flush pipeline/weight/tuning caches
  planned_ = true;
  VX_INFO << "Planned " << segments_.size() << " segment(s) over " << graph_.nodes.size()
          << " nodes";
}

std::vector<BackendKind> Session::nodeBackends() const {
  std::vector<BackendKind> v;
  for (int bi : nodeBackendIdx_) v.push_back(bi >= 0 ? backends_[bi]->kind() : BackendKind::kCpu);
  return v;
}

const RtTensor* Session::tensor(const std::string& name) const {
  TensorId id = graph_.find(name);
  if (id == kNoTensor) return nullptr;
  return &pool_[id];
}

Status Session::run(const std::vector<IOTensor>& inputs, std::vector<IOTensor>& outputs) {
  ExecContext ctx;
  ctx.pool = &pool_;
  ctx.graph = &graph_;
  ctx.config = &cfg_;
  ctx.profiler = &profiler_;
  profiler_.clear();

  // --- bind inputs ---
  for (const auto& io : inputs) {
    TensorId id = graph_.find(io.name);
    if (id == kNoTensor) {
      // fall back to the single graph input
      if (graph_.inputs.size() == 1) id = graph_.inputs[0];
      else { VX_ERROR << "input not found: " << io.name; return Status::kInvalidArgument; }
    }
    RtTensor& rt = pool_[id];
    rt.shape = io.shape.empty() ? graph_.tensors[id].shape : io.shape;
    rt.dtype = io.dtype;
    rt.host.bytes = io.data;
    rt.hostValid = true;
    rt.deviceValid = false;
  }

  // --- run segments in order ---
  try {
    for (auto& seg : segments_) seg->run(ctx);
  } catch (const std::exception& e) {
    VX_ERROR << "run failed: " << e.what();
    return Status::kRuntimeError;
  }

  // --- layer dump ---
  if (cfg_.layerDump) {
    ::mkdir(cfg_.layerDumpDir.c_str(), 0755);
    for (size_t i = 0; i < pool_.size(); ++i) {
      RtTensor& rt = pool_[i];
      if (!rt.hostValid || graph_.isInitializer((TensorId)i)) continue;
      std::string nm = graph_.tensors[i].name;
      for (char& c : nm) if (c == '/' || c == ':') c = '_';
      std::ofstream f(cfg_.layerDumpDir + "/" + nm + ".bin", std::ios::binary);
      if (f) f.write((const char*)rt.host.bytes.data(), rt.host.bytes.size());
    }
    VX_INFO << "layer dump written to " << cfg_.layerDumpDir;
  }

  // --- collect outputs ---
  outputs.clear();
  for (TensorId oid : graph_.outputs) {
    RtTensor& rt = pool_[oid];
    IOTensor io;
    io.name = graph_.tensors[oid].name;
    io.shape = rt.shape;
    io.dtype = rt.dtype;
    io.data = rt.host.bytes;
    outputs.push_back(std::move(io));
  }
  return Status::kOk;
}

}  // namespace vx
