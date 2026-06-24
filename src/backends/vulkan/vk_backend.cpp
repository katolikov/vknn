#include "vk_backend.h"
#include <chrono>
#include <cstdlib>
#include <set>
#include <sys/stat.h>
#if defined(VXRT_ENABLE_NEON) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include "vx/dtype.h"
#include "vx/logging.h"
#include "vx/profiler.h"

namespace vx {

// True once the fp16 shader variants (conv_fp16, dwconv_fp16, ...) are compiled in. The ops
// pick the _fp16 kernels and upload half weights when this and the device feature line up.
bool vxVulkanFp16Available() {
  return true;
}

// ============================ VkOpRegistry ============================
VkOpRegistry& VkOpRegistry::instance() {
  static VkOpRegistry r;
  return r;
}

// ============================ WeightCache ============================
// Binary format: [u32 nWeights]{[u32 klen][key][u32 nfloats][floats]} [u32 nTune]{[u32
// klen][key][i32 val]}
void WeightCache::load(const std::string& path) {
  path_ = path;
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return;
  auto rd32 = [&](uint32_t& v) { return fread(&v, 4, 1, f) == 1; };
  uint32_t nw = 0;
  if (rd32(nw)) {
    for (uint32_t i = 0; i < nw; ++i) {
      uint32_t kl = 0, nf = 0;
      if (!rd32(kl)) break;
      std::string k(kl, 0);
      fread(&k[0], 1, kl, f);
      if (!rd32(nf)) break;
      std::vector<float> d(nf);
      fread(d.data(), 4, nf, f);
      weights_[k] = std::move(d);
    }
    uint32_t nt = 0;
    if (rd32(nt))
      for (uint32_t i = 0; i < nt; ++i) {
        uint32_t kl = 0;
        int32_t val = 0;
        if (!rd32(kl)) break;
        std::string k(kl, 0);
        fread(&k[0], 1, kl, f);
        fread(&val, 4, 1, f);
        tune_[k] = val;
      }
  }
  fclose(f);
  VX_INFO << "WeightCache: loaded " << weights_.size() << " prepacked weights, " << tune_.size()
          << " tuning entries from " << path;
}
void WeightCache::save() const {
  if (path_.empty() || !dirty_) return;
  FILE* f = fopen(path_.c_str(), "wb");
  if (!f) {
    VX_WARN << "WeightCache: cannot write " << path_;
    return;
  }
  auto wr32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
  wr32((uint32_t)weights_.size());
  for (auto& kv : weights_) {
    wr32((uint32_t)kv.first.size());
    fwrite(kv.first.data(), 1, kv.first.size(), f);
    wr32((uint32_t)kv.second.size());
    fwrite(kv.second.data(), 4, kv.second.size(), f);
  }
  wr32((uint32_t)tune_.size());
  for (auto& kv : tune_) {
    wr32((uint32_t)kv.first.size());
    fwrite(kv.first.data(), 1, kv.first.size(), f);
    int32_t v = kv.second;
    fwrite(&v, 4, 1, f);
  }
  fclose(f);
  dirty_ = false;
  VX_INFO << "WeightCache: saved " << weights_.size() << " weights + " << tune_.size()
          << " tuning entries -> " << path_;
}
bool WeightCache::get(const std::string& key, std::vector<float>& out) const {
  auto it = weights_.find(key);
  if (it == weights_.end()) return false;
  out = it->second;
  return true;
}
void WeightCache::put(const std::string& key, const std::vector<float>& data) {
  weights_[key] = data;
  dirty_ = true;
}
int WeightCache::tuned(const std::string& sig, int dflt) const {
  auto it = tune_.find(sig);
  return it == tune_.end() ? dflt : it->second;
}
void WeightCache::setTuned(const std::string& sig, int val) {
  tune_[sig] = val;
  dirty_ = true;
}

// ============================ VulkanBackend ============================
class VulkanBackend : public Backend {
 public:
  VulkanBackend() {
    ctx_ = std::make_unique<vk::VulkanContext>();
    if (ctx_->initialized()) runner_ = std::make_unique<vk::CommandRunner>(*ctx_);
  }
  BackendKind kind() const override { return BackendKind::kVulkan; }
  const char* name() const override { return "Vulkan"; }
  bool available() const override { return ctx_ && ctx_->initialized(); }
  bool supports(OpType t, DType dt) const override {
    if (!available()) return false;
    // Debug/fallback hook: VXRT_DISABLE_VK_OPS="Add,Conv" forces those ops to fall back
    // (used to demonstrate the NEON CPU fallback path, M5).
    if (const char* dis = std::getenv("VXRT_DISABLE_VK_OPS")) {
      std::string list = dis, name = opTypeName(t);
      if (list.find(name) != std::string::npos) return false;
    }
    return VkOpRegistry::instance().has(t);
  }

  // Shape-aware gate: Concat and Binary only run on the GPU for the NC4HW4-friendly cases; other
  // layouts fall back to the (always-correct) CPU op.
  bool supportsNode(const Graph& g, const Node& nd, DType dt) const override {
    if (!supports(nd.type, dt)) return false;
    if (nd.type == OpType::kConcat) {
      const Shape& out = g.desc(nd.outputs[0]).shape;
      int rank = (int)out.size();
      int64_t axis = nd.attr.geti("axis", 1);
      if (axis < 0) axis += rank;
      if (rank != 4 || axis != 1) return false;  // channel concat only
      for (TensorId in : nd.inputs) {
        const Shape& s = g.desc(in).shape;
        if (s.size() != 4 || s[1] % 4 != 0) return false;  // need 4-aligned channel blocks
      }
      return true;
    }
    if (nd.type == OpType::kBinary) {
      if (nd.inputs.size() != 2) return false;
      // constant operands aren't uploaded as device buffers; let the CPU op handle those
      if (g.isInitializer(nd.inputs[0]) || g.isInitializer(nd.inputs[1])) return false;
      const Shape& a = g.desc(nd.inputs[0]).shape;
      const Shape& b = g.desc(nd.inputs[1]).shape;
      if (a == b) return true;
      // channel-broadcast on either operand: one is [N,C,1,1], the other [N,C,H,W] (matching N,C)
      auto bcastOver = [](const Shape& s, const Shape& full) {
        return s.size() == 4 && full.size() == 4 && s[0] == full[0] && s[1] == full[1] &&
               s[2] == 1 && s[3] == 1 && (full[2] > 1 || full[3] > 1);
      };
      return bcastOver(a, b) || bcastOver(b, a);
    }
    return true;
  }

  vk::VulkanContext& ctx() { return *ctx_; }
  vk::CommandRunner& runner() { return *runner_; }
  vk::PipelineCache* pipelineCache(const Config& cfg) {
    if (!cache_) {
      ::mkdir(cfg.cacheDir.c_str(), 0755);
      cache_ = std::make_unique<vk::PipelineCache>(*ctx_, cfg.cacheDir + "/pipeline.bin");
    }
    return cache_.get();
  }
  WeightCache* weightCache(const Config& cfg) {
    if (!wcache_) {
      wcache_ = std::make_unique<WeightCache>();
      if (cfg.cacheWeights)
        wcache_->load(cfg.cacheDir + "/weights.bin");
      else
        wcache_->load("");  // disabled (no path)
    }
    return wcache_.get();
  }
  void saveCaches() {
    if (cache_) cache_->save();
    if (wcache_) wcache_->save();
  }

  bool useFp16(const Config& cfg) const {
    return vxVulkanFp16Available() && ctx_->caps().shaderFloat16 &&
           (cfg.precision == Precision::kFp16 || cfg.precision == Precision::kAuto);
  }

  std::unique_ptr<Segment> compileSegment(const std::vector<int>& idx, Graph& g,
                                          const Config& cfg) override;
  void finalize() override { saveCaches(); }

  // ---- host NCHW fp32  <->  device NC4HW4 (fp32 path; fp16 device buffers handled here) ----
  static void packToBuffer(vk::Buffer* buf, const RtTensor& rt, bool fp16) {
    NCHW x = NCHW::from(rt.shape);
    int64_t Cb = cBlocks(x.c);
    const float* src = rt.host.f32();
    if (fp16) {
      fp16_t* dst = reinterpret_cast<fp16_t*>(buf->host());
      for (int64_t n = 0; n < x.n; ++n)
        for (int64_t cb = 0; cb < Cb; ++cb)
          for (int64_t h = 0; h < x.h; ++h)
            for (int64_t w = 0; w < x.w; ++w) {
              int64_t base = ((((n * Cb + cb) * x.h + h) * x.w) + w) * 4;
              float t[4] = {0, 0, 0, 0};
              for (int l = 0; l < 4; ++l) {
                int64_t c = cb * 4 + l;
                if (c < x.c) t[l] = src[((n * x.c + c) * x.h + h) * x.w + w];
              }
#if defined(VXRT_ENABLE_NEON) && defined(__ARM_NEON)
              // convert the 4 gathered channels to fp16 in one instruction
              vst1_f16(reinterpret_cast<__fp16*>(dst + base), vcvt_f16_f32(vld1q_f32(t)));
#else
              for (int l = 0; l < 4; ++l) dst[base + l] = floatToHalf(t[l]);
#endif
            }
    } else {
      float* dst = reinterpret_cast<float*>(buf->host());
      for (int64_t n = 0; n < x.n; ++n)
        for (int64_t cb = 0; cb < Cb; ++cb)
          for (int64_t h = 0; h < x.h; ++h)
            for (int64_t w = 0; w < x.w; ++w) {
              int64_t base = ((((n * Cb + cb) * x.h + h) * x.w) + w) * 4;
              for (int l = 0; l < 4; ++l) {
                int64_t c = cb * 4 + l;
                dst[base + l] = (c < x.c) ? src[((n * x.c + c) * x.h + h) * x.w + w] : 0.f;
              }
            }
    }
  }
  static void unpackFromBuffer(vk::Buffer* buf, RtTensor& rt, bool fp16) {
    NCHW x = NCHW::from(rt.shape);
    int64_t Cb = cBlocks(x.c);
    rt.host.resizeElems(x.elems(), DType::kFloat32);
    rt.dtype = DType::kFloat32;
    float* dst = rt.host.f32();
    if (fp16) {
      const fp16_t* src = reinterpret_cast<const fp16_t*>(buf->host());
      for (int64_t n = 0; n < x.n; ++n)
        for (int64_t c = 0; c < x.c; ++c)
          for (int64_t h = 0; h < x.h; ++h)
            for (int64_t w = 0; w < x.w; ++w) {
              int64_t cb = c / 4, l = c % 4;
              int64_t sidx = ((((n * Cb + cb) * x.h + h) * x.w) + w) * 4 + l;
              dst[((n * x.c + c) * x.h + h) * x.w + w] = halfToFloat(src[sidx]);
            }
    } else {
      const float* src = reinterpret_cast<const float*>(buf->host());
      for (int64_t n = 0; n < x.n; ++n)
        for (int64_t c = 0; c < x.c; ++c)
          for (int64_t h = 0; h < x.h; ++h)
            for (int64_t w = 0; w < x.w; ++w) {
              int64_t cb = c / 4, l = c % 4;
              int64_t sidx = ((((n * Cb + cb) * x.h + h) * x.w) + w) * 4 + l;
              dst[((n * x.c + c) * x.h + h) * x.w + w] = src[sidx];
            }
    }
    rt.hostValid = true;
  }

 private:
  std::unique_ptr<vk::VulkanContext> ctx_;
  std::unique_ptr<vk::CommandRunner> runner_;
  std::unique_ptr<vk::PipelineCache> cache_;
  std::unique_ptr<WeightCache> wcache_;
};

// ============================ VulkanSegment ============================
class VulkanSegment : public Segment {
 public:
  VulkanSegment(const std::vector<int>& idx, Graph& g, const Config& cfg, VulkanBackend* be)
      : be_(be), g_(g), cfg_(cfg) {
    nodeIdx = idx;
    useFp16_ = be_->useFp16(cfg);
    elemSize_ = useFp16_ ? 2 : 4;

    // 1) allocate device buffers for all activation tensors (non-initializers).
    std::set<TensorId> acts;
    for (int ni : idx) {
      for (TensorId in : g.nodes[ni].inputs)
        if (in != kNoTensor && !g.isInitializer(in)) acts.insert(in);
      for (TensorId o : g.nodes[ni].outputs)
        if (o != kNoTensor) acts.insert(o);
    }
    for (TensorId tid : acts) {
      size_t bytes = (size_t)packedElems(g.tensors[tid].shape) * elemSize_;
      if (bytes == 0) bytes = elemSize_ * 4;
      auto pref = vk::MemPref::kAuto;
      buffers_[tid] = std::make_shared<vk::Buffer>(be_->ctx(), bytes, pref);
    }

    // 2) build env + ops; prepare uploads weights.
    env_.backend = be_;
    env_.ctx = &be_->ctx();
    env_.cache = be_->pipelineCache(cfg);
    env_.weights = be_->weightCache(cfg);
    env_.runner = &be_->runner();
    env_.tuning = cfg.tuning;
    env_.graph = &g;
    env_.config = &cfg;
    env_.useFp16 = useFp16_;
    // per-model weight-cache namespace: FNV-1a over the whole graph (same for every segment of this
    // model, distinct across models) so a shared cacheDir can't return another model's weights.
    {
      uint64_t h = 1469598103934665603ull;
      auto mix = [&](const std::string& s) {
        for (char c : s) { h ^= (uint8_t)c; h *= 1099511628211ull; }
      };
      for (const auto& nd : g.nodes) { mix(nd.name); mix(opTypeName(nd.type)); }
      mix(std::to_string(g.nodes.size()));
      char buf[20];
      snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
      env_.modelTag = buf;
    }
    env_.devBuf = [this](TensorId t) -> vk::Buffer* {
      auto it = buffers_.find(t);
      return it == buffers_.end() ? nullptr : it->second.get();
    };
    for (int ni : idx) {
      auto op = VkOpRegistry::instance().create(g.nodes[ni].type);
      if (!op)
        throw Error(Status::kUnsupported,
                    std::string("no Vulkan kernel for ") + opTypeName(g.nodes[ni].type));
      op->prepare(g.nodes[ni], env_);
      ops_.push_back(std::move(op));
    }

    // 3) timestamp query pool (2 per node). Only when profiling - the extra writes + the implicit
    //    barriers around them aren't free, and we don't want them on the hot path.
    if (be_->ctx().caps().timestampSupported && cfg.profile) {
      VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
      qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
      qi.queryCount = (uint32_t)(idx.size() * 2);
      vkCreateQueryPool(be_->ctx().device(), &qi, nullptr, &queryPool_);
    }

    // 4) pre-record the command buffer for the static graph.
    record();
  }

  ~VulkanSegment() override {
    if (queryPool_) vkDestroyQueryPool(be_->ctx().device(), queryPool_, nullptr);
  }

  void record() {
    cmd_ = be_->runner().allocate();
    be_->runner().begin(cmd_);
    if (queryPool_) vkCmdResetQueryPool(cmd_, queryPool_, 0, (uint32_t)(nodeIdx.size() * 2));
    auto isCopy = [&](int idx) {
      OpType t = g_.nodes[idx].type;
      return t == OpType::kReshape || t == OpType::kFlatten;
    };
    // Precise barriers: each activation tensor has a single writer, so only a read-after-write needs
    // a barrier. Emit one before an op only when it reads a tensor written since the last barrier,
    // letting independent ops (e.g. the parallel branches of an Inception module, or a residual
    // block's downsample and conv1) run without draining the GPU between them. When profiling, keep
    // a barrier after every op so the per-op timestamps aren't polluted by overlap.
    const bool perOpBarrier = (queryPool_ != VK_NULL_HANDLE);
    std::set<TensorId> writtenSinceBarrier;
    bool copySinceBarrier = false;
    for (size_t k = 0; k < nodeIdx.size(); ++k) {
      const Node& node = g_.nodes[nodeIdx[k]];
      bool needBarrier = perOpBarrier;
      if (!needBarrier)
        for (TensorId in : node.inputs)
          if (in != kNoTensor && writtenSinceBarrier.count(in)) { needBarrier = true; break; }
      if (needBarrier) {
        if (copySinceBarrier || isCopy(nodeIdx[k]))
          vk::transferBarrier(cmd_);
        else
          vk::computeBarrier(cmd_);
        writtenSinceBarrier.clear();
        copySinceBarrier = false;
      }
      if (queryPool_)
        vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool_, (uint32_t)(k * 2));
      ops_[k]->record(cmd_, node, env_);
      if (queryPool_)
        vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool_,
                            (uint32_t)(k * 2 + 1));
      for (TensorId o : node.outputs)
        if (o != kNoTensor) writtenSinceBarrier.insert(o);
      if (isCopy(nodeIdx[k])) copySinceBarrier = true;
    }
    // Final barrier so the segment outputs are complete + visible before the host reads them.
    if (copySinceBarrier)
      vk::transferBarrier(cmd_);
    else
      vk::computeBarrier(cmd_);
    be_->runner().end(cmd_);
    recorded_ = true;
  }

  void run(ExecContext& ctx) override {
    const bool timing = std::getenv("VXRT_TIMING") != nullptr;
    auto now = [] { return std::chrono::high_resolution_clock::now(); };
    auto t0 = now();
    // attach boundary buffers to RtTensors (cross-segment residency) + upload inputs.
    for (TensorId tid : boundaryInputs) {
      RtTensor& rt = ctx.t(tid);
      auto bit = buffers_.find(tid);
      if (bit == buffers_.end()) continue;
      if (!rt.device) rt.device = std::make_shared<DeviceStorage>();
      rt.device->buffer = bit->second;
      if (rt.hostValid && !rt.deviceValid) {
        VulkanBackend::packToBuffer(bit->second.get(), rt, useFp16_);
        rt.deviceValid = true;
        rt.deviceFormat = TensorFormat::kNC4HW4;
      }
    }
    auto t1 = now();

    double wall = be_->runner().submitAndWait(cmd_);
    auto t2 = now();

    // download boundary outputs to host.
    for (TensorId tid : boundaryOutputs) {
      auto bit = buffers_.find(tid);
      if (bit == buffers_.end()) continue;
      RtTensor& rt = ctx.t(tid);
      if (!rt.device) rt.device = std::make_shared<DeviceStorage>();
      rt.device->buffer = bit->second;
      rt.deviceValid = true;
      rt.deviceFormat = TensorFormat::kNC4HW4;
      VulkanBackend::unpackFromBuffer(bit->second.get(), rt, useFp16_);
    }
    if (timing) {
      auto t3 = now();
      auto ms = [&](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
      };
      VX_INFO << "timing: pack=" << ms(t0, t1) << "ms submit+gpu=" << wall
              << "ms unpack=" << ms(t2, t3) << "ms";
    }

    // layer-dump: bring every activation back to host for per-layer diffing.
    if (ctx.config && ctx.config->layerDump) {
      for (auto& kv : buffers_) {
        RtTensor& rt = ctx.t(kv.first);
        if (g_.isInitializer(kv.first)) continue;
        VulkanBackend::unpackFromBuffer(kv.second.get(), rt, useFp16_);
      }
    }

    // profiler: per-node GPU time from timestamps + dispatch dims.
    if (ctx.profiler && ctx.profiler->enabled() && queryPool_) {
      std::vector<uint64_t> ts(nodeIdx.size() * 2, 0);
      vkGetQueryPoolResults(be_->ctx().device(), queryPool_, 0, (uint32_t)ts.size(),
                            ts.size() * sizeof(uint64_t), ts.data(), sizeof(uint64_t),
                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
      double period = be_->ctx().caps().timestampPeriod;
      for (size_t k = 0; k < nodeIdx.size(); ++k) {
        const Node& node = g_.nodes[nodeIdx[k]];
        OpRecord r;
        r.name = node.name;
        r.type = node.type;
        r.backend = "Vulkan";
        r.gpuMs = (double)(ts[k * 2 + 1] - ts[k * 2]) * period / 1e6;
        r.cpuMs = 0;
        ctx.profiler->add(r);
      }
      // GPU span (first dispatch start -> last dispatch end) vs the CPU-side submit wall: the
      // difference is barrier bubbles + submit/fence latency, not kernel work.
      double span = (double)(ts.back() - ts.front()) * period / 1e6;
      VX_INFO << "gpu span=" << span << "ms  submit-wall=" << wall << "ms  (gap = overhead)";
    }
  }

 private:
  VulkanBackend* be_;
  Graph& g_;
  const Config& cfg_;
  bool useFp16_ = false;
  int elemSize_ = 4;
  std::map<TensorId, std::shared_ptr<vk::Buffer>> buffers_;
  std::vector<std::unique_ptr<VulkanOp>> ops_;
  VkOpEnv env_;
  VkCommandBuffer cmd_ = VK_NULL_HANDLE;
  VkQueryPool queryPool_ = VK_NULL_HANDLE;
  bool recorded_ = false;
};

std::unique_ptr<Segment> VulkanBackend::compileSegment(const std::vector<int>& idx, Graph& g,
                                                       const Config& cfg) {
  auto s = std::make_unique<VulkanSegment>(idx, g, cfg, this);
  s->backend = this;
  return s;
}

VX_REGISTER_BACKEND(BackendKind::kVulkan, VulkanBackend);

}  // namespace vx
