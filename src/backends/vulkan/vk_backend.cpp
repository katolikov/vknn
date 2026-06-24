#include "vk_backend.h"
#include <chrono>
#include <set>
#include <sys/stat.h>
#include "vx/dtype.h"
#include "vx/logging.h"
#include "vx/profiler.h"

namespace vx {

// ============================ VkOpRegistry ============================
VkOpRegistry& VkOpRegistry::instance() {
  static VkOpRegistry r;
  return r;
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
    return VkOpRegistry::instance().has(t);
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
  void saveCaches() { if (cache_) cache_->save(); }

  bool useFp16(const Config& cfg) const {
    // fp16 device path requires the fp16 shader variants (registered when present).
    extern bool vxVulkanFp16Available();
    return vxVulkanFp16Available() && ctx_->caps().shaderFloat16 &&
           (cfg.precision == Precision::kFp16 || cfg.precision == Precision::kAuto);
  }

  std::unique_ptr<Segment> compileSegment(const std::vector<int>& idx, Graph& g,
                                          const Config& cfg) override;

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
              for (int l = 0; l < 4; ++l) {
                int64_t c = cb * 4 + l;
                float v = (c < x.c) ? src[((n * x.c + c) * x.h + h) * x.w + w] : 0.f;
                dst[base + l] = floatToHalf(v);
              }
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
    env_.graph = &g;
    env_.config = &cfg;
    env_.useFp16 = useFp16_;
    env_.devBuf = [this](TensorId t) -> vk::Buffer* {
      auto it = buffers_.find(t);
      return it == buffers_.end() ? nullptr : it->second.get();
    };
    for (int ni : idx) {
      auto op = VkOpRegistry::instance().create(g.nodes[ni].type);
      if (!op) throw Error(Status::kUnsupported,
          std::string("no Vulkan kernel for ") + opTypeName(g.nodes[ni].type));
      op->prepare(g.nodes[ni], env_);
      ops_.push_back(std::move(op));
    }

    // 3) timestamp query pool (2 per node).
    if (be_->ctx().caps().timestampSupported) {
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
    if (queryPool_)
      vkCmdResetQueryPool(cmd_, queryPool_, 0, (uint32_t)(nodeIdx.size() * 2));
    for (size_t k = 0; k < nodeIdx.size(); ++k) {
      const Node& node = g_.nodes[nodeIdx[k]];
      if (queryPool_)
        vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool_, (uint32_t)(k * 2));
      ops_[k]->record(cmd_, node, env_);
      if (queryPool_)
        vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool_, (uint32_t)(k * 2 + 1));
      vk::computeBarrier(cmd_);
    }
    be_->runner().end(cmd_);
    recorded_ = true;
  }

  void run(ExecContext& ctx) override {
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

    double wall = be_->runner().submitAndWait(cmd_);

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
      (void)wall;
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
