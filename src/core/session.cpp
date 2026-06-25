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

std::unique_ptr<Session> Session::createFromVxm(const std::string& path, const Config& cfg) {
  Graph g;
  if (!loadGraphBin(g, path)) {
    VX_ERROR << "failed to load .vxm: " << path;
    return nullptr;
  }
  auto s = std::unique_ptr<Session>(new Session());
  s->graphOptimized_ = true;  // passes already baked in
  s->graph_ = std::move(g);
  s->cfg_ = cfg;
  cfg.applyLogLevel();
  s->profiler_.setEnabled(cfg.profile);
  auto t0 = std::chrono::high_resolution_clock::now();
  s->plan();
  auto t1 = std::chrono::high_resolution_clock::now();
  VX_INFO << "Session created from .vxm in "
          << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms";
  return s;
}

bool Session::saveOptimized(const std::string& path) const { return saveGraphBin(graph_, path); }

std::unique_ptr<Session> Session::create(Graph&& g, const Config& cfg) {
  auto s = std::unique_ptr<Session>(new Session());
  s->graph_ = std::move(g);
  s->cfg_ = cfg;
  cfg.applyLogLevel();
  s->profiler_.setEnabled(cfg.profile);
  auto t0 = std::chrono::high_resolution_clock::now();
  s->plan();
  auto t1 = std::chrono::high_resolution_clock::now();
  VX_INFO << "Session created in " << std::chrono::duration<double, std::milli>(t1 - t0).count()
          << " ms";
  return s;
}

void Session::foldTinyGpuIslands() {
  int cpuIdx = -1;
  for (size_t i = 0; i < backends_.size(); ++i)
    if (backends_[i]->kind() == BackendKind::kCpu) cpuIdx = (int)i;
  if (cpuIdx < 0) return;  // no CPU backend to fall back to
  auto isCpu = [&](int ni) { return backends_[nodeBackendIdx_[ni]]->kind() == BackendKind::kCpu; };
  // Approximate per-node work: a conv/gemm output costs Cin*KH*KW per element, everything else ~1.
  auto nodeCost = [&](int ni) -> int64_t {
    const Node& nd = graph_.nodes[ni];
    int64_t outElems = nd.outputs.empty() || nd.outputs[0] == kNoTensor
                           ? 0 : numElements(graph_.desc(nd.outputs[0]).shape);
    if ((nd.type == OpType::kConv || nd.type == OpType::kGemm) && nd.inputs.size() > 1) {
      const Shape& w = graph_.desc(nd.inputs[1]).shape;
      int64_t k = 1;
      for (size_t i = 1; i < w.size(); ++i) k *= w[i];  // Cin*KH*KW (Gemm: K)
      return outElems * std::max<int64_t>(k, 1);
    }
    return outElems;
  };
  const int64_t kKeepOnGpu = 2'000'000;  // work below this isn't worth a CPU<->GPU round trip
  bool changed = true;
  while (changed) {
    changed = false;
    // contiguous same-backend runs over the (topo-sorted) node order
    std::vector<int> runOf(graph_.nodes.size(), -1);
    int nRuns = 0;
    for (size_t n = 0; n < graph_.nodes.size(); ++n) {
      if (n == 0 || nodeBackendIdx_[n] != nodeBackendIdx_[n - 1]) nRuns++;
      runOf[n] = nRuns - 1;
    }
    std::vector<int> producerRun(graph_.tensors.size(), -1);
    std::vector<std::vector<int>> runNodes(nRuns);
    for (size_t n = 0; n < graph_.nodes.size(); ++n) {
      runNodes[runOf[n]].push_back((int)n);
      for (TensorId o : graph_.nodes[n].outputs)
        if (o != kNoTensor) producerRun[o] = runOf[n];
    }
    for (int r = 0; r < nRuns && !changed; ++r) {
      if (isCpu(runNodes[r].front())) continue;  // already CPU
      std::set<TensorId> internal;
      for (int ni : runNodes[r])
        for (TensorId o : graph_.nodes[ni].outputs)
          if (o != kNoTensor) internal.insert(o);
      bool touchesGpu = false;
      int64_t work = 0;
      for (int ni : runNodes[r]) {
        work += nodeCost(ni);
        for (TensorId in : graph_.nodes[ni].inputs)  // fed by another GPU run?
          if (in != kNoTensor && !internal.count(in) && producerRun[in] >= 0 &&
              !isCpu(runNodes[producerRun[in]].front()))
            touchesGpu = true;
      }
      for (size_t q = 0; q < graph_.nodes.size() && !touchesGpu; ++q) {  // consumed by a GPU run?
        if (runOf[q] == r || isCpu((int)q)) continue;
        for (TensorId x : graph_.nodes[q].inputs)
          if (x != kNoTensor && internal.count(x)) { touchesGpu = true; break; }
      }
      if (touchesGpu || work >= kKeepOnGpu) continue;  // connected to GPU work, or heavy -> keep
      bool cpuOk = true;
      for (int ni : runNodes[r])
        if (!backends_[cpuIdx]->supportsNode(graph_, graph_.nodes[ni], DType::kFloat32)) {
          cpuOk = false;
          break;
        }
      if (!cpuOk) continue;
      for (int ni : runNodes[r]) nodeBackendIdx_[ni] = cpuIdx;
      changed = true;  // restart: the fold may have merged neighbours into a new island
    }
  }
}

void Session::plan() {
  // --- graph optimization passes (NCHW IR, static batch=1) ---
  // Skipped when the graph came from a .vxm (passes already applied at save time).
  if (!graphOptimized_) runStandardPasses(graph_, 1);
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
    if (!reg.has(k)) {
      VX_DEBUG << "backend " << backendName(k) << " not registered";
      continue;
    }
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
      if (backends_[bi]->supportsNode(graph_, nd, dt)) {
        chosen = (int)bi;
        break;
      }
    }
    if (chosen < 0)
      throw Error(Status::kUnsupported, std::string("no backend supports op ") +
                                            opTypeName(nd.type) + " (" + nd.name + ")");
    nodeBackendIdx_[n] = chosen;
    // warn if the primary backend couldn't take it
    if (backends_[chosen]->kind() != cfg_.backend && byKind_.count(cfg_.backend) &&
        !byKind_[cfg_.backend]->supportsNode(graph_, nd, dt)) {
      VX_WARN_THROTTLE(std::string("fallback_") + opTypeName(nd.type), 2)
          << "op " << opTypeName(nd.type) << " (" << nd.name << ") not supported by "
          << backendName(cfg_.backend) << " backend -> falling back to "
          << backends_[chosen]->name()
          << ". Perf note: this op does not run on the requested backend.";
    }
  }

  // --- fold tiny GPU "islands" back to CPU ---
  // A maximal run of GPU nodes that is fed only by CPU output and consumed only by CPU (a true island
  // in the dataflow) costs a CPU->GPU->CPU round trip. When that island does little work (a detection
  // head's DFL conv / sigmoid on tiny tensors) the pack/unpack dwarfs the compute and, worse, stresses
  // the boundary path. Run it on the CPU instead. The heavy backbone/head convs are kept on the GPU —
  // they exceed the work threshold, so this never drags real compute off the accelerator.
  foldTinyGpuIslands();

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
                if (x == o) {
                  external = true;
                  break;
                }
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
      if (graph_.inputs.size() == 1)
        id = graph_.inputs[0];
      else {
        VX_ERROR << "input not found: " << io.name;
        return Status::kInvalidArgument;
      }
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
    bool dbg = std::getenv("VXRT_DEBUG_SEG") != nullptr;
    for (size_t si = 0; si < segments_.size(); ++si) {
      if (dbg) VX_INFO << "RUN segment " << si << "/" << segments_.size()
                       << " backend=" << segments_[si]->backend->name();
      segments_[si]->run(ctx);
    }
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
      for (char& c : nm)
        if (c == '/' || c == ':') c = '_';
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

// --- ergonomic API ------------------------------------------------------------------------------

static IOInfo ioInfoOf(const Graph& g, TensorId id) {
  IOInfo info;
  info.name = g.tensors[id].name;
  info.shape = g.tensors[id].shape;
  info.dtype = g.tensors[id].dtype;
  info.elems = numElements(info.shape);
  return info;
}

std::vector<IOInfo> Session::inputInfo() const {
  std::vector<IOInfo> v;
  for (TensorId id : graph_.inputs) v.push_back(ioInfoOf(graph_, id));
  return v;
}

std::vector<IOInfo> Session::outputInfo() const {
  std::vector<IOInfo> v;
  for (TensorId id : graph_.outputs) v.push_back(ioInfoOf(graph_, id));
  return v;
}

Status Session::run(const std::vector<std::vector<float>>& inputData,
                    std::vector<IOTensor>& outputs) {
  if (inputData.size() != graph_.inputs.size()) {
    VX_ERROR << "run: expected " << graph_.inputs.size() << " input(s), got " << inputData.size();
    return Status::kInvalidArgument;
  }
  std::vector<IOTensor> ins(graph_.inputs.size());
  for (size_t i = 0; i < graph_.inputs.size(); ++i) {
    TensorId id = graph_.inputs[i];
    const TensorDesc& d = graph_.tensors[id];
    int64_t want = numElements(d.shape);
    // Allow callers not to know the count (want<=0 for fully-dynamic shapes); otherwise validate.
    if (want > 0 && (int64_t)inputData[i].size() != want) {
      VX_ERROR << "run: input '" << d.name << "' expects " << want << " values, got "
               << inputData[i].size();
      return Status::kInvalidArgument;
    }
    ins[i].name = d.name;
    ins[i].shape = d.shape;
    ins[i].dtype = DType::kFloat32;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(inputData[i].data());
    ins[i].data.assign(p, p + inputData[i].size() * sizeof(float));
  }
  return run(ins, outputs);
}

std::vector<float> Session::infer(const std::vector<float>& input) {
  std::vector<IOTensor> outs;
  if (run({input}, outs) != Status::kOk || outs.empty()) return {};
  const float* o = outs[0].f32();
  return std::vector<float>(o, o + numElements(outs[0].shape));
}

}  // namespace vx
