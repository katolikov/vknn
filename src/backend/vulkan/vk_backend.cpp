#include "vk_backend.h"

#include <sys/stat.h>

#include <chrono>
#include <cstdlib>
#include <set>
#if defined(VKNN_ENABLE_NEON) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include "vknn/dtype.h"
#include "vknn/logging.h"
#include "vknn/profiler.h"

namespace vknn {

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
  if (!f)
    return;
  auto rd32 = [&](uint32_t& v) { return fread(&v, 4, 1, f) == 1; };
  uint32_t nw = 0;
  if (rd32(nw)) {
    for (uint32_t i = 0; i < nw; ++i) {
      uint32_t kl = 0, nf = 0;
      if (!rd32(kl))
        break;
      std::string k(kl, 0);
      fread(&k[0], 1, kl, f);
      if (!rd32(nf))
        break;
      std::vector<float> d(nf);
      fread(d.data(), 4, nf, f);
      weights_[k] = std::move(d);
    }
    uint32_t nt = 0;
    if (rd32(nt))
      for (uint32_t i = 0; i < nt; ++i) {
        uint32_t kl = 0;
        int32_t val = 0;
        if (!rd32(kl))
          break;
        std::string k(kl, 0);
        fread(&k[0], 1, kl, f);
        fread(&val, 4, 1, f);
        tune_[k] = val;
      }
  }
  fclose(f);
  VKNN_INFO << "WeightCache: loaded " << weights_.size() << " prepacked weights, " << tune_.size()
          << " tuning entries from " << path;
}
void WeightCache::save() const {
  if (path_.empty() || !dirty_)
    return;
  FILE* f = fopen(path_.c_str(), "wb");
  if (!f) {
    VKNN_WARN << "WeightCache: cannot write " << path_;
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
  VKNN_INFO << "WeightCache: saved " << weights_.size() << " weights + " << tune_.size()
          << " tuning entries -> " << path_;
}
bool WeightCache::get(const std::string& key, std::vector<float>& out) const {
  auto it = weights_.find(key);
  if (it == weights_.end())
    return false;
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
    if (ctx_->initialized())
      runner_ = std::make_unique<vk::CommandRunner>(*ctx_);
  }
  BackendKind kind() const override { return BackendKind::kVulkan; }
  const char* name() const override { return "Vulkan"; }
  bool available() const override { return ctx_ && ctx_->initialized(); }
  bool supports(OpType t, DType dt) const override {
    if (!available())
      return false;
    // Debug/fallback hook: VKNN_DISABLE_VK_OPS="Add,Conv" forces those ops to fall back
    // (used to demonstrate the NEON CPU fallback path, M5).
    if (const char* dis = std::getenv("VKNN_DISABLE_VK_OPS")) {
      std::string list = dis, name = opTypeName(t);
      if (list.find(name) != std::string::npos)
        return false;
    }
    return VkOpRegistry::instance().has(t);
  }

  // Shape-aware gate: Concat and Binary only run on the GPU for the NC4HW4-friendly cases; other
  // layouts fall back to the (always-correct) CPU op.
  bool supportsNode(const Graph& g, const Node& nd, DType dt) const override {
    if (!supports(nd.type, dt))
      return false;
    // Generic N-D ops the GPU runs flat (Transpose/Slice always; Concat/Softmax/Binary/Add either
    // NC4HW4 or flat per the layout pass). The flat row-major kernels handle rank <= 6.
    if (nd.type == OpType::kTranspose || nd.type == OpType::kSlice ||
        nd.type == OpType::kConvertLayout || nd.type == OpType::kConcat ||
        nd.type == OpType::kSoftmax || nd.type == OpType::kSqueeze)
      return true;
    if (nd.type == OpType::kExpand || nd.type == OpType::kTile)
      // flat broadcast/tile gather decodes up to kMaxRank=6 output dims.
      return g.desc(nd.outputs[0]).shape.size() <= 8;
    if (nd.type == OpType::kMatMul) {
      // Batched N-D matmul on the flat row-major path; the kernel decodes up to kMaxRank=6 out
      // dims.
      if (nd.inputs.size() != 2)
        return false;
      return g.desc(nd.outputs[0]).shape.size() <= 8;
    }
    if (nd.type == OpType::kDepthToSpace) {
      // [N,C,H,W] -> [N,C/b^2,H*b,W*b]; flat index-remap kernel. Need 4D and C divisible by b^2.
      const Shape& in = g.desc(nd.inputs[0]).shape;
      int64_t b = nd.attr.geti("blocksize", 1);
      return in.size() == 4 && b >= 1 && in[1] % (b * b) == 0;
    }
    if (nd.type == OpType::kReduce) {
      // flat reduce kernel: one thread per output element, loops the reduced axes. rank <= 6.
      const Shape& in = g.desc(nd.inputs[0]).shape;
      return !in.empty() && in.size() <= 8;
    }
    if (nd.type == OpType::kFusedDwPw) {
      // LDS holds E depthwise outputs (cap 1024). Run ALL eligible fused nodes on the GPU: a
      // partial gate (some fused nodes on CPU) creates a GPU/CPU boundary that mis-handles the
      // fused residual.
      const Shape& in = g.desc(nd.inputs[0]).shape;    // expanded [N,E,H,W]
      const Shape& out = g.desc(nd.outputs[0]).shape;  // [N,Cout,OH,OW]
      if (in.size() != 4 || out.size() != 4)
        return false;
      return in[1] <= 1024;
    }
    if (nd.type == OpType::kFusedSE) {
      // fixed LDS arrays: avg[1024], s1[256]
      const Shape& f = g.desc(nd.inputs[0]).shape;
      const Shape& w1 = g.desc(nd.inputs[1]).shape;
      return f.size() == 4 && f[1] <= 1024 && !w1.empty() && w1[0] <= 256;
    }
    if (nd.type == OpType::kGridSample) {
      // GPU path needs the grid as a raw constant buffer (it can't be NC4HW4-packed); runtime grids
      // and cubic mode fall back to the CPU op.
      if (nd.inputs.size() < 2 || !g.isInitializer(nd.inputs[1]))
        return false;
      const Shape& in = g.desc(nd.inputs[0]).shape;
      if (in.size() != 4)
        return false;
      std::string m = nd.attr.gets("mode", "bilinear");
      return m == "bilinear" || m == "linear" || m == "nearest";
    }
    if (nd.type == OpType::kResize) {
      // GPU kernel resizes spatial dims only; channel/batch resize falls back to the CPU op.
      const Shape& in = g.desc(nd.inputs[0]).shape;
      const Shape& out = g.desc(nd.outputs[0]).shape;
      return in.size() == 4 && out.size() == 4 && in[0] == out[0] && in[1] == out[1];
    }
    if (nd.type == OpType::kLayerNorm) {
      // Flat reduction over the trailing axes; scale (and bias, if present) must be const
      // initializers.
      if (nd.inputs.size() < 2 || !g.isInitializer(nd.inputs[1]))
        return false;
      if (nd.inputs.size() > 2 && nd.inputs[2] != kNoTensor && !g.isInitializer(nd.inputs[2]))
        return false;
      return true;
    }
    if (nd.type == OpType::kWhere || nd.type == OpType::kEqual)
      // flat broadcasting kernels (fixed PC arrays) decode up to kMaxRank=6 output dims.
      return g.desc(nd.outputs[0]).shape.size() <= 8;
    if (nd.type == OpType::kUnsqueeze)
      return true;  // metadata copy on the flat path
    if (nd.type == OpType::kCast) {
      // float->float casts are a no-op copy on the unified-precision buffer; int targets stay CPU.
      DType o = g.desc(nd.outputs[0]).dtype;
      return o == DType::kFloat32 || o == DType::kFloat16;
    }
    if (nd.type == OpType::kGather)
      // flat axis-aware gather; index may be a constant (uploaded) or a runtime float activation
      // (RoPE).
      return nd.inputs.size() >= 2;
    if (nd.type == OpType::kScatterND)
      // flat scatter; index may be a constant or a runtime float activation. Data rank within
      // kMaxRank.
      return nd.inputs.size() >= 3 && g.desc(nd.inputs[0]).shape.size() <= 8;
    if (nd.type == OpType::kEinsum) {
      // Only "i,j->ij" (outer product) has a GPU kernel; other equations use the CPU op.
      std::string eq;
      for (char c : nd.attr.gets("equation", ""))
        if (c != ' ' && c != '\t')
          eq += c;
      return eq == "i,j->ij";
    }
    if (nd.type == OpType::kBatchNorm) {
      // per-channel affine; needs 4D input and the 4 params (gamma/beta/mean/var) as constants.
      if (nd.inputs.size() < 5 || g.desc(nd.inputs[0]).shape.size() != 4)
        return false;
      for (int i = 1; i <= 4; ++i)
        if (!g.isInitializer(nd.inputs[i]))
          return false;
      return true;
    }
    if (nd.type == OpType::kSplit) {
      // NC4HW4 channel split (4D, axis 1, 4-aligned outputs) is a block copy; any other split runs
      // on the flat row-major path (a Slice per output) for rank <= kMaxRank.
      const Shape& in = g.desc(nd.inputs[0]).shape;
      if (in.empty())
        return false;
      int rank = (int)in.size();
      int64_t axis = nd.attr.geti("axis", 0);
      if (axis < 0)
        axis += rank;
      if (rank == 4 && axis == 1) {
        bool aligned = true;
        for (TensorId o : nd.outputs) {
          if (o == kNoTensor)
            continue;
          const Shape& os = g.desc(o).shape;
          if (os.size() != 4 || os[1] % 4 != 0)
            aligned = false;
        }
        if (aligned)
          return true;
      }
      return rank <= 8;
    }
    if (nd.type == OpType::kClip) {
      // const-or-absent scalar bounds (baked into the PC in prepare); runtime bounds fall back.
      for (int i = 1; i < 3 && i < (int)nd.inputs.size(); ++i)
        if (nd.inputs[i] != kNoTensor && !g.isInitializer(nd.inputs[i]))
          return false;
      return true;
    }
    // Add/Binary: 2 inputs required. The NC4HW4 kernel does same-shape + channel-broadcast; the
    // flat kernel (chosen by the layout pass) does everything else incl. constant operands.
    if (nd.type == OpType::kAdd || nd.type == OpType::kBinary)
      return nd.inputs.size() == 2;
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
    if (cache_)
      cache_->save();
    if (wcache_)
      wcache_->save();
  }

  bool useFp16(const Config& cfg) const {
    return vxVulkanFp16Available() && ctx_->caps().shaderFloat16 &&
           (cfg.precision == Precision::kFp16 || cfg.precision == Precision::kAuto);
  }

  std::unique_ptr<Segment> compileSegment(const std::vector<int>& idx, Graph& g,
                                          const Config& cfg) override;
  void finalize() override { saveCaches(); }

  // ---- host NCHW fp32  <->  device NC4HW4 (fp32 path; fp16 device buffers handled here) ----
  static void packToBuffer(vk::Buffer* buf, const RtTensor& rt, bool fp16, bool flat = false) {
    if (flat) {  // host NCHW row-major == the flat device buffer; straight copy (+ fp16 convert)
      int64_t n = numElements(rt.shape);
      const float* src = rt.host.f32();
      if (fp16) {
        fp16_t* dst = reinterpret_cast<fp16_t*>(buf->host());
        for (int64_t i = 0; i < n; ++i)
          dst[i] = floatToHalf(src[i]);
      } else {
        std::memcpy(buf->host(), src, (size_t)n * 4);
      }
      return;
    }
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
                if (c < x.c)
                  t[l] = src[((n * x.c + c) * x.h + h) * x.w + w];
              }
#if defined(VKNN_ENABLE_NEON) && defined(__ARM_NEON)
              // convert the 4 gathered channels to fp16 in one instruction
              vst1_f16(reinterpret_cast<__fp16*>(dst + base), vcvt_f16_f32(vld1q_f32(t)));
#else
              for (int l = 0; l < 4; ++l)
                dst[base + l] = floatToHalf(t[l]);
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
  static void unpackFromBuffer(vk::Buffer* buf, RtTensor& rt, bool fp16, bool flat = false) {
    if (flat) {  // flat device buffer == host NCHW row-major; straight copy (+ fp16 convert)
      int64_t n = numElements(rt.shape);
      rt.host.resizeElems(n, DType::kFloat32);
      rt.dtype = DType::kFloat32;
      float* dst = rt.host.f32();
      if (fp16) {
        halfToFloatBulk(reinterpret_cast<const fp16_t*>(buf->host()), dst, n);
      } else {
        std::memcpy(dst, buf->host(), (size_t)n * 4);
      }
      rt.hostValid = true;
      return;
    }
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
        if (in != kNoTensor && !g.isInitializer(in))
          acts.insert(in);
      // A fused residual (out = act(conv + residual)) is read by record() but isn't in node.inputs,
      // so it needs a buffer too — and may be produced by another segment (boundary input).
      TensorId res = g.nodes[ni].fusedResidual;
      if (res != kNoTensor && !g.isInitializer(res))
        acts.insert(res);
      for (TensorId o : g.nodes[ni].outputs)
        if (o != kNoTensor)
          acts.insert(o);
    }
    // Tensors this segment produces that are read OUTSIDE it (by another segment or as a graph
    // output) get downloaded to host via unpackFromBuffer. The default kAuto memory is
    // write-combined (fast to upload, but CPU READS are uncached and brutally slow -> 150ms on a
    // YOLO head boundary). Allocate those readback buffers as HOST_CACHED so the download is fast;
    // keep the rest as kAuto.
    std::set<int> idxSet(idx.begin(), idx.end());
    std::set<TensorId> readBack(g.outputs.begin(), g.outputs.end());
    {
      std::set<TensorId> produced;
      for (int ni : idx)
        for (TensorId o : g.nodes[ni].outputs)
          if (o != kNoTensor)
            produced.insert(o);
      for (size_t q = 0; q < g.nodes.size(); ++q) {
        if (idxSet.count((int)q))
          continue;
        for (TensorId in : g.nodes[q].inputs)
          if (in != kNoTensor && produced.count(in))
            readBack.insert(in);
      }
    }
    // Debug: VKNN_DUMP_NAMES="substr1,substr2" forces matching tensors to dedicated (un-aliased)
    // readback buffers and dumps them to /data/local/tmp/vxrt/dump after the run — so intermediate
    // activations can be diffed despite the liveness planner reusing buffers. A few tensors only.
    if (const char* dn = std::getenv("VKNN_DUMP_NAMES")) {
      std::string list = dn;
      for (TensorId tid : acts) {
        const std::string& nm = g.tensors[tid].name;
        if (nm.empty())
          continue;
        size_t pos = 0, comma;
        do {
          comma = list.find(',', pos);
          std::string sub = list.substr(pos, comma == std::string::npos ? comma : comma - pos);
          if (!sub.empty() && nm.find(sub) != std::string::npos) {
            readBack.insert(tid);
            dumpTids_.push_back(tid);
            break;
          }
          pos = comma + 1;
        } while (comma != std::string::npos);
      }
    }
    auto actBytes = [&](TensorId tid) -> size_t {
      int64_t elems = g.tensors[tid].gpuFlat ? numElements(g.tensors[tid].shape)
                                             : packedElems(g.tensors[tid].shape);
      size_t b = (size_t)elems * elemSize_;
      return b == 0 ? (size_t)elemSize_ * 4 : b;
    };
    // Liveness buffer planner. One buffer per tensor keeps ALL activations live at once (~11.5GB on
    // the YoNoSplat encoder); the simultaneously-live peak is ~0.17GB. Boundary-in (produced by
    // another segment) and readback (read by host / another segment) tensors get dedicated buffers;
    // internal (produced-and-consumed only here) tensors are pooled by a greedy scan over execution
    // order, reusing a buffer once its previous occupant's last use has passed. The buffer-level
    // barriers in record() make the write-after-read at each reuse point safe.
    std::set<TensorId> producedHere;
    for (int ni : idx)
      for (TensorId o : g.nodes[ni].outputs)
        if (o != kNoTensor)
          producedHere.insert(o);
    for (TensorId tid : acts) {
      bool internal = producedHere.count(tid) && !readBack.count(tid);
      if (internal)
        continue;  // pooled below
      auto pref = readBack.count(tid) ? vk::MemPref::kReadback : vk::MemPref::kAuto;
      buffers_[tid] = std::make_shared<vk::Buffer>(be_->ctx(), actBytes(tid), pref);
    }
    // [firstPos,lastPos] of each internal tensor within this segment's execution order
    std::map<TensorId, int> firstPos, lastPos;
    auto touch = [&](TensorId t, int k) {
      if (t == kNoTensor || !producedHere.count(t) || readBack.count(t))
        return;
      if (!firstPos.count(t))
        firstPos[t] = k;
      lastPos[t] = k;
    };
    for (int k = 0; k < (int)idx.size(); ++k) {
      const Node& nd = g.nodes[idx[k]];
      for (TensorId in : nd.inputs)
        touch(in, k);
      touch(nd.fusedResidual, k);
      for (TensorId o : nd.outputs)
        touch(o, k);
    }
    std::vector<TensorId> order;
    order.reserve(firstPos.size());
    for (auto& kv : firstPos)
      order.push_back(kv.first);
    std::sort(order.begin(), order.end(),
              [&](TensorId a, TensorId b) { return firstPos[a] < firstPos[b]; });
    struct Slot {
      std::shared_ptr<vk::Buffer> buf;
      size_t cap;
      int deadAt;
    };
    std::vector<Slot> busy, freeSlots;
    for (TensorId tid : order) {
      int p = firstPos[tid];
      for (size_t i = 0; i < busy.size();) {
        if (busy[i].deadAt < p) {
          freeSlots.push_back(busy[i]);
          busy[i] = busy.back();
          busy.pop_back();
        } else
          ++i;
      }
      size_t need = actBytes(tid);
      int best = -1;
      for (size_t i = 0; i < freeSlots.size(); ++i)
        if (freeSlots[i].cap >= need && (best < 0 || freeSlots[i].cap < freeSlots[best].cap))
          best = (int)i;
      Slot s;
      if (best >= 0) {
        s = freeSlots[best];
        freeSlots[best] = freeSlots.back();
        freeSlots.pop_back();
      } else {
        s.buf = std::make_shared<vk::Buffer>(be_->ctx(), need, vk::MemPref::kAuto);
        s.cap = need;
      }
      s.deadAt = lastPos[tid];
      buffers_[tid] = s.buf;
      busy.push_back(s);
    }

    // 2) build env + ops; prepare uploads weights.
    env_.backend = be_;
    env_.ctx = &be_->ctx();
    env_.cache = be_->pipelineCache(cfg);
    env_.weights = be_->weightCache(cfg);
    env_.runner = &be_->runner();
    env_.tuning = cfg.tuning;
    env_.winograd = cfg.winograd;
    env_.graph = &g;
    env_.config = &cfg;
    env_.useFp16 = useFp16_;
    // per-model weight-cache namespace: FNV-1a over the whole graph (same for every segment of this
    // model, distinct across models) so a shared cacheDir can't return another model's weights.
    {
      uint64_t h = 1469598103934665603ull;
      auto mix = [&](const std::string& s) {
        for (char c : s) {
          h ^= (uint8_t)c;
          h *= 1099511628211ull;
        }
      };
      for (const auto& nd : g.nodes) {
        mix(nd.name);
        mix(opTypeName(nd.type));
      }
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
    if (queryPool_)
      vkDestroyQueryPool(be_->ctx().device(), queryPool_, nullptr);
  }

  void record() {
    cmd_ = be_->runner().allocate();
    be_->runner().begin(cmd_);
    if (queryPool_)
      vkCmdResetQueryPool(cmd_, queryPool_, 0, (uint32_t)(nodeIdx.size() * 2));
    auto isCopy = [&](int idx) {
      const Node& nn = g_.nodes[idx];
      OpType t = nn.type;
      // A flat split is a compute dispatch (flat_gather); the NC4HW4 split is a buffer copy.
      if (t == OpType::kSplit)
        return nn.outputs.empty() || nn.outputs[0] == kNoTensor || !g_.desc(nn.outputs[0]).gpuFlat;
      // Reshape/Flatten/Squeeze/Unsqueeze/Cast are vkCmdCopyBuffer (transfer-stage writes).
      return t == OpType::kReshape || t == OpType::kFlatten || t == OpType::kSqueeze ||
             t == OpType::kUnsqueeze || t == OpType::kCast;
    };
    // Precise barriers: each activation tensor has a single writer, so only a read-after-write
    // needs a barrier. Emit one before an op only when it reads a tensor written since the last
    // barrier, letting independent ops (e.g. the parallel branches of an Inception module, or a
    // residual block's downsample and conv1) run without draining the GPU between them. When
    // profiling, keep a barrier after every op so the per-op timestamps aren't polluted by overlap.
    const bool perOpBarrier = (queryPool_ != VK_NULL_HANDLE);
    // Hazard tracking is at the BUFFER level, not the tensor level: the liveness planner aliases
    // multiple tensors onto one buffer, so a node that writes a reused buffer has a
    // write-after-read hazard against the previous occupant that a tensor-level check would miss.
    // For non-aliased buffers this is identical to the old per-tensor read-after-write (single
    // writer per buffer), so the independent-op overlap (Inception/YOLO) is preserved.
    std::set<vk::Buffer*> writtenBufs, readBufs;
    auto bufOf = [&](TensorId t) -> vk::Buffer* {
      if (t == kNoTensor)
        return nullptr;
      auto it = buffers_.find(t);
      return it == buffers_.end() ? nullptr : it->second.get();
    };
    bool copySinceBarrier = false;
    for (size_t k = 0; k < nodeIdx.size(); ++k) {
      const Node& node = g_.nodes[nodeIdx[k]];
      bool needBarrier = perOpBarrier;
      if (!needBarrier) {
        for (TensorId in : node.inputs)  // read-after-write
          if (vk::Buffer* b = bufOf(in))
            if (writtenBufs.count(b)) {
              needBarrier = true;
              break;
            }
        if (!needBarrier)
          if (vk::Buffer* b = bufOf(node.fusedResidual))
            if (writtenBufs.count(b))
              needBarrier = true;
        if (!needBarrier)
          for (TensorId o : node.outputs)  // write-after-write / write-after-read (reused buffer)
            if (vk::Buffer* b = bufOf(o))
              if (writtenBufs.count(b) || readBufs.count(b)) {
                needBarrier = true;
                break;
              }
      }
      if (needBarrier) {
        if (copySinceBarrier || isCopy(nodeIdx[k]))
          vk::transferBarrier(cmd_);
        else
          vk::computeBarrier(cmd_);
        writtenBufs.clear();
        readBufs.clear();
        copySinceBarrier = false;
      }
      if (queryPool_)
        vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool_, (uint32_t)(k * 2));
      ops_[k]->record(cmd_, node, env_);
      if (queryPool_)
        vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool_,
                            (uint32_t)(k * 2 + 1));
      for (TensorId in : node.inputs)
        if (vk::Buffer* b = bufOf(in))
          readBufs.insert(b);
      if (vk::Buffer* b = bufOf(node.fusedResidual))
        readBufs.insert(b);
      for (TensorId o : node.outputs)
        if (vk::Buffer* b = bufOf(o))
          writtenBufs.insert(b);
      if (isCopy(nodeIdx[k]))
        copySinceBarrier = true;
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
    const bool timing = std::getenv("VKNN_TIMING") != nullptr;
    auto now = [] { return std::chrono::high_resolution_clock::now(); };
    auto t0 = now();
    // attach boundary buffers to RtTensors (cross-segment residency) + upload inputs.
    // Each segment owns a SEPARATE buffer per tensor, so a boundary input must be (re)packed into
    // THIS segment's buffer unless that exact buffer already holds the data. The old `!deviceValid`
    // guard checked only "some buffer has it" — when a tensor was produced by an earlier GPU
    // segment (deviceValid set, but pointing at that segment's buffer) a later GPU consumer skipped
    // the pack and read its own never-written buffer (zeros). Seen as YOLO's P3/P4 class logits
    // collapsing.
    for (TensorId tid : boundaryInputs) {
      RtTensor& rt = ctx.t(tid);
      auto bit = buffers_.find(tid);
      if (bit == buffers_.end())
        continue;
      bool alreadyHere = rt.deviceValid && rt.device && rt.device->buffer == bit->second;
      bool flat = g_.desc(tid).gpuFlat;
      if (!rt.device)
        rt.device = std::make_shared<DeviceStorage>();
      rt.device->buffer = bit->second;
      if (rt.hostValid && !alreadyHere) {
        VulkanBackend::packToBuffer(bit->second.get(), rt, useFp16_, flat);
        rt.deviceValid = true;
        rt.deviceFormat = flat ? TensorFormat::kNCHW : TensorFormat::kNC4HW4;
      }
    }
    auto t1 = now();

    double wall = be_->runner().submitAndWait(cmd_);
    auto t2 = now();

    // download boundary outputs to host.
    for (TensorId tid : boundaryOutputs) {
      auto bit = buffers_.find(tid);
      if (bit == buffers_.end())
        continue;
      RtTensor& rt = ctx.t(tid);
      bool flat = g_.desc(tid).gpuFlat;
      if (!rt.device)
        rt.device = std::make_shared<DeviceStorage>();
      rt.device->buffer = bit->second;
      rt.deviceValid = true;
      rt.deviceFormat = flat ? TensorFormat::kNCHW : TensorFormat::kNC4HW4;
      VulkanBackend::unpackFromBuffer(bit->second.get(), rt, useFp16_, flat);
    }
    if (timing) {
      auto t3 = now();
      auto ms = [&](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
      };
      VKNN_INFO << "timing: pack=" << ms(t0, t1) << "ms submit+gpu=" << wall
              << "ms unpack=" << ms(t2, t3) << "ms";
    }

    // VKNN_DUMP_NAMES targeted dump: write the named tensors (dedicated buffers) to disk for
    // diffing.
    if (!dumpTids_.empty()) {
      ::mkdir("/data/local/tmp/vxrt/dump", 0755);
      for (TensorId tid : dumpTids_) {
        auto bit = buffers_.find(tid);
        if (bit == buffers_.end())
          continue;
        RtTensor& rt = ctx.t(tid);
        VulkanBackend::unpackFromBuffer(bit->second.get(), rt, useFp16_, g_.desc(tid).gpuFlat);
        std::string nm = g_.tensors[tid].name;
        for (char& c : nm)
          if (c == '/' || c == ':')
            c = '_';
        FILE* f = fopen(("/data/local/tmp/vxrt/dump/" + nm + ".bin").c_str(), "wb");
        if (f) {
          fwrite(rt.host.bytes.data(), 1, rt.host.bytes.size(), f);
          fclose(f);
        }
      }
    }
    // layer-dump: bring every activation back to host for per-layer diffing.
    if (ctx.config && ctx.config->layerDump) {
      for (auto& kv : buffers_) {
        RtTensor& rt = ctx.t(kv.first);
        if (g_.isInitializer(kv.first))
          continue;
        VulkanBackend::unpackFromBuffer(kv.second.get(), rt, useFp16_, g_.desc(kv.first).gpuFlat);
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
      VKNN_INFO << "gpu span=" << span << "ms  submit-wall=" << wall << "ms  (gap = overhead)";
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
  std::vector<TensorId> dumpTids_;  // VKNN_DUMP_NAMES debug: tensors to dump after the run
};

std::unique_ptr<Segment> VulkanBackend::compileSegment(const std::vector<int>& idx, Graph& g,
                                                       const Config& cfg) {
  auto s = std::make_unique<VulkanSegment>(idx, g, cfg, this);
  s->backend = this;
  return s;
}

VKNN_REGISTER_BACKEND(BackendKind::kVulkan, VulkanBackend);

}  // namespace vknn
