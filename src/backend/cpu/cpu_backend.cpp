#include "cpu_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include "vknn/logging.h"
#include "vknn/profiler.h"

namespace vknn {

CpuOpRegistry& CpuOpRegistry::instance() {
  static CpuOpRegistry r;
  return r;
}

namespace cpu {
float* allocOut(RtTensor& rt, const Shape& shape) {
  rt.shape = shape;
  rt.dtype = DType::kFloat32;
  rt.host.resizeElems(numElements(shape), DType::kFloat32);
  rt.hostValid = true;
  rt.deviceValid = false;
  return rt.host.f32();
}
int64_t* allocOutI64(RtTensor& rt, const Shape& shape) {
  rt.shape = shape;
  rt.dtype = DType::kInt64;
  rt.host.resizeElems(numElements(shape), DType::kInt64);
  rt.hostValid = true;
  rt.deviceValid = false;
  return rt.host.i64();
}
void applyAct(float* p, int64_t n, ActType act, float lo, float hi) {
  switch (act) {
    case ActType::kRelu:
      for (int64_t i = 0; i < n; ++i)
        p[i] = p[i] > 0 ? p[i] : 0;
      break;
    case ActType::kRelu6:
      for (int64_t i = 0; i < n; ++i) {
        float v = p[i];
        p[i] = v < 0 ? 0 : (v > 6 ? 6 : v);
      }
      break;
    case ActType::kClip:
      for (int64_t i = 0; i < n; ++i) {
        float v = p[i];
        p[i] = v < lo ? lo : (v > hi ? hi : v);
      }
      break;
    case ActType::kHardSwish:
      for (int64_t i = 0; i < n; ++i) {
        float v = p[i];
        p[i] = v * std::min(std::max(v + 3.f, 0.f), 6.f) / 6.f;
      }
      break;
    case ActType::kSiLU:
      for (int64_t i = 0; i < n; ++i)
        p[i] = p[i] / (1.f + std::exp(-p[i]));
      break;
    default:
      break;
  }
}
void copyAs(const RtTensor& X, RtTensor& Y, const Shape& shape) {
  Y.shape = shape;
  Y.dtype = X.dtype;
  Y.host.resizeElems(numElements(shape), X.dtype);
  Y.hostValid = true;
  Y.deviceValid = false;
  std::memcpy(Y.host.bytes.data(), X.host.bytes.data(),
              std::min(Y.host.bytes.size(), X.host.bytes.size()));
}
}  // namespace cpu

// --------------------------- CpuSegment ---------------------------
class CpuSegment : public Segment {
public:
  CpuSegment(const std::vector<int>& idx, Graph& g) : g_(g) {
    nodeIdx = idx;
    for (int i : idx) {
      auto op = CpuOpRegistry::instance().create(g.nodes[i].type);
      ops_.push_back(std::move(op));
    }
  }
  void run(ExecContext& ctx) override {
    for (size_t k = 0; k < nodeIdx.size(); ++k) {
      const Node& node = ctx.graph->nodes[nodeIdx[k]];
      if (ctx.config && ctx.config->debugSegments) {
        std::string sh;
        for (auto t : node.inputs) {
          sh += std::to_string(t) + ":[";
          if (t >= 0) {
            const RtTensor& rt = ctx.t(t);
            for (auto d : rt.shape)
              sh += std::to_string(d) + ",";
            sh += rt.hostValid ? "]h " : "]NOHOST ";
          } else
            sh += "?] ";
        }
        VKNN_INFO << "  cpuop " << opTypeName(node.type) << " '" << node.name << "' ins=" << sh;
      }
      CpuOp* op = ops_[k].get();
      if (!op)
        throw Error(Status::kUnsupported, std::string("no CPU kernel for op ") +
                                              opTypeName(node.type) + " (" + node.name + ")");
      auto t0 = std::chrono::high_resolution_clock::now();
      op->run(node, ctx);
      auto t1 = std::chrono::high_resolution_clock::now();
      if (ctx.profiler && ctx.profiler->enabled()) {
        OpRecord r;
        r.name = node.name;
        r.type = node.type;
        r.backend = backend ? backend->name() : "CPU";
        r.cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        r.fellBack = isFallback;
        ctx.profiler->add(r);
      }
    }
  }

private:
  Graph& g_;
  std::vector<std::unique_ptr<CpuOp>> ops_;
};

// --------------------------- CpuBackend ---------------------------
class CpuBackend : public Backend {
public:
  BackendKind kind() const override { return BackendKind::kCpu; }
  const char* name() const override { return "CPU"; }
  bool available() const override { return true; }
  bool supports(OpType t, DType dt) const override {
    auto& r = CpuOpRegistry::instance();
    if (!r.has(t))
      return false;
    return dt == DType::kFloat32 || dt == DType::kInt64 || dt == DType::kInt32;
  }
  std::unique_ptr<Segment> compileSegment(const std::vector<int>& idx, Graph& g,
                                          const Config&) override {
    auto s = std::make_unique<CpuSegment>(idx, g);
    s->backend = this;
    return s;
  }
};

VKNN_REGISTER_BACKEND(BackendKind::kCpu, CpuBackend);

}  // namespace vknn
