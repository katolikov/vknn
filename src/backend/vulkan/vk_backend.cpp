#include "vk_backend.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sys/stat.h>
#if defined(VKNN_ENABLE_NEON) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include "ops/boundary_convert.h"
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
    VkOpRegistry &VkOpRegistry::instance() {
        static VkOpRegistry r;
        return r;
    }

    // ============================ WeightCache ============================
    // Binary format: [u32 nWeights]{[u32 klen][key][u32 nfloats][floats]} [u32 nTune]{[u32
    // klen][key][i32 val]}
    void WeightCache::loadBytes(const uint8_t *data, size_t n, bool keepWeights, bool keepTune) {
        enabled_     = keepWeights;
        tuneEnabled_ = keepTune;
        size_t off   = 0;
        auto   rd32 = [&](uint32_t &v) {
            if (off + 4 > n)
            {
                return false;
            }
            std::memcpy(&v, data + off, 4);
            off += 4;
            return true;
        };
        uint32_t nw = 0;
        if (rd32(nw))
        {
            for (uint32_t i = 0; i < nw; ++i)
            {
                uint32_t kl = 0, nf = 0;
                if (!rd32(kl) || off + kl > n)
                {
                    break;
                }
                std::string k((const char *) data + off, kl);
                off += kl;
                if (!rd32(nf) || off + (size_t) nf * 4 > n)
                {
                    break;
                }
                if (keepWeights) // else advance past the blob without materializing it (saves RAM)
                {
                    std::vector<float> d(nf);
                    std::memcpy(d.data(), data + off, (size_t) nf * 4);
                    weights_[k] = std::move(d);
                }
                off += (size_t) nf * 4;
            }
            uint32_t nt = 0;
            if (rd32(nt))
            {
                for (uint32_t i = 0; i < nt; ++i)
                {
                    uint32_t kl  = 0;
                    int32_t  val = 0;
                    if (!rd32(kl) || off + kl > n)
                    {
                        break;
                    }
                    std::string k((const char *) data + off, kl);
                    off += kl;
                    if (off + 4 > n)
                    {
                        break;
                    }
                    std::memcpy(&val, data + off, 4);
                    off += 4;
                    if (keepTune)
                    {
                        tune_[k] = val;
                    }
                }
            }
        }
        VKNN_INFO << "WeightCache: loaded " << weights_.size() << " prepacked weights, " << tune_.size() << " tuning entries";
    }
    std::vector<uint8_t> WeightCache::serialize() const {
        std::vector<uint8_t> out;
        auto                 wr32 = [&](uint32_t v) {
            const uint8_t *p = (const uint8_t *) &v;
            out.insert(out.end(), p, p + 4);
        };
        auto wrBytes = [&](const void *p, size_t bytes) {
            const uint8_t *b = (const uint8_t *) p;
            out.insert(out.end(), b, b + bytes);
        };
        wr32((uint32_t) weights_.size());
        for (auto &kv: weights_)
        {
            wr32((uint32_t) kv.first.size());
            wrBytes(kv.first.data(), kv.first.size());
            wr32((uint32_t) kv.second.size());
            wrBytes(kv.second.data(), kv.second.size() * 4);
        }
        wr32(tuneEnabled_ ? (uint32_t) tune_.size() : 0u);
        if (tuneEnabled_)
        {
            for (auto &kv: tune_)
            {
                wr32((uint32_t) kv.first.size());
                wrBytes(kv.first.data(), kv.first.size());
                int32_t v = kv.second;
                wrBytes(&v, 4);
            }
        }
        return out;
    }
    bool WeightCache::get(const std::string &key, std::vector<float> &out) const {
        auto it = weights_.find(key);
        if (it == weights_.end())
        {
            return false;
        }
        out = it->second;
        return true;
    }
    void WeightCache::put(const std::string &key, const std::vector<float> &data) {
        weights_[key] = data;
        dirty_        = true;
    }
    int WeightCache::tuned(const std::string &sig, int dflt) const {
        auto it = tune_.find(sig);
        return it == tune_.end() ? dflt : it->second;
    }
    void WeightCache::setTuned(const std::string &sig, int val) {
        tune_[sig] = val;
        dirty_     = true;
    }

    // ============================ VulkanBackend ============================
    class VulkanBackend: public Backend {
      public:
        VulkanBackend() {
            ctx_ = std::make_unique<vk::VulkanContext>();
            if (ctx_->initialized())
            {
                runner_ = std::make_unique<vk::CommandRunner>(*ctx_);
            }
        }
        BackendKind kind() const override {
            return BackendKind::Vulkan;
        }
        const char *name() const override {
            return "Vulkan";
        }
        bool available() const override {
            return ctx_ && ctx_->initialized();
        }
        void configure(const Config &cfg) override {
            disabledOps_ = cfg.disableVkOps;
        }
        bool supports(OpType t, DType dt) const override {
            if (!available())
            {
                return false;
            }
            // Debug/fallback hook: Config::disableVkOps="Add,Conv" forces those ops to fall back
            // to the CPU path.
            if (!disabledOps_.empty() && disabledOps_.find(opTypeName(t)) != std::string::npos)
            {
                return false;
            }
            return VkOpRegistry::instance().has(t);
        }

        // Shape-aware gate: Concat and Binary only run on the GPU for the NC4HW4-friendly cases; other
        // layouts fall back to the (always-correct) CPU op.
        bool supportsNode(const Graph &g, const Node &nd, DType dt) const override {
            if (!supports(nd.type, dt))
            {
                return false;
            }
            // Generic N-D ops the GPU runs flat (Transpose/Slice always; Concat/Softmax/Binary/Add either
            // NC4HW4 or flat per the layout pass). The flat row-major kernels handle rank <= 6.
            if (nd.type == OpType::Transpose || nd.type == OpType::Slice || nd.type == OpType::ConvertLayout || nd.type == OpType::Concat || nd.type == OpType::Softmax || nd.type == OpType::Squeeze)
            {
                return true;
            }
            if (nd.type == OpType::Expand || nd.type == OpType::Tile)
            {
                // flat broadcast/tile gather decodes up to kMaxRank=6 output dims.
                return g.desc(nd.outputs[0]).shape.size() <= 8;
            }
            if (nd.type == OpType::Pad)
            {
                // Flat pad runs on the GPU only for static pads + a supported mode (else CPU). Mirrors
                // gpuFlatNode so a GPU-assigned Pad is always marked flat by the layout pass.
                if (g.desc(nd.outputs[0]).shape.size() > 8)
                {
                    return false;
                }
                std::string mode = nd.attr.gets("mode", "constant");
                if (mode != "constant" && mode != "edge" && mode != "reflect")
                {
                    return false;
                }
                bool padsKnown = !nd.attr.getints("pads").empty() || (nd.inputs.size() > 1 && nd.inputs[1] != kNoTensor && g.isInitializer(nd.inputs[1]));
                if (!padsKnown)
                {
                    return false;
                }
                return !(nd.inputs.size() > 2 && nd.inputs[2] != kNoTensor && !g.isInitializer(nd.inputs[2]));
            }
            if (nd.type == OpType::MatMul)
            {
                // Batched N-D matmul on the flat row-major path; the kernel decodes up to kMaxRank=6 out
                // dims. Two operands, or three when a rank-1 bias is fused in (the _bias kernel binds it
                // as a 4th buffer).
                if (!(nd.inputs.size() == 2 || (nd.inputs.size() == 3 && nd.fusedBias != kNoTensor)))
                {
                    return false;
                }
                return g.desc(nd.outputs[0]).shape.size() <= 8;
            }
            if (nd.type == OpType::DepthToSpace)
            {
                // [N,C,H,W] -> [N,C/b^2,H*b,W*b]; flat index-remap kernel. Need 4D and C divisible by b^2.
                const Shape &in = g.desc(nd.inputs[0]).shape;
                int64_t      b  = nd.attr.geti("blocksize", 1);
                return in.size() == 4 && b >= 1 && in[1] % (b * b) == 0;
            }
            if (nd.type == OpType::Reduce)
            {
                // flat reduce kernel: one thread per output element, loops the reduced axes. rank <= 6.
                const Shape &in = g.desc(nd.inputs[0]).shape;
                return !in.empty() && in.size() <= 8;
            }
            if (nd.type == OpType::FusedDwPw)
            {
                // LDS holds E depthwise outputs (cap 1024). Run ALL eligible fused nodes on the GPU: a
                // partial gate (some fused nodes on CPU) creates a GPU/CPU boundary that mis-handles the
                // fused residual.
                const Shape &in  = g.desc(nd.inputs[0]).shape;  // expanded [N,E,H,W]
                const Shape &out = g.desc(nd.outputs[0]).shape; // [N,Cout,OH,OW]
                if (in.size() != 4 || out.size() != 4)
                {
                    return false;
                }
                return in[1] <= 1024;
            }
            if (nd.type == OpType::FusedSE)
            {
                // fixed LDS arrays: avg[1024], s1[256]
                const Shape &f  = g.desc(nd.inputs[0]).shape;
                const Shape &w1 = g.desc(nd.inputs[1]).shape;
                return f.size() == 4 && f[1] <= 1024 && !w1.empty() && w1[0] <= 256;
            }
            if (nd.type == OpType::GridSample)
            {
                // GPU path needs the grid as a raw constant buffer (it can't be NC4HW4-packed); runtime grids
                // and cubic mode fall back to the CPU op.
                if (nd.inputs.size() < 2 || !g.isInitializer(nd.inputs[1]))
                {
                    return false;
                }
                const Shape &in = g.desc(nd.inputs[0]).shape;
                if (in.size() != 4)
                {
                    return false;
                }
                std::string m = nd.attr.gets("mode", "bilinear");
                return m == "bilinear" || m == "linear" || m == "nearest";
            }
            if (nd.type == OpType::Resize)
            {
                // GPU kernel resizes spatial dims only; channel/batch resize falls back to the CPU op.
                const Shape &in  = g.desc(nd.inputs[0]).shape;
                const Shape &out = g.desc(nd.outputs[0]).shape;
                return in.size() == 4 && out.size() == 4 && in[0] == out[0] && in[1] == out[1];
            }
            if (nd.type == OpType::LayerNorm)
            {
                // Flat reduction over the trailing axes; scale (and bias, if present) must be const
                // initializers.
                if (nd.inputs.size() < 2 || !g.isInitializer(nd.inputs[1]))
                {
                    return false;
                }
                if (nd.inputs.size() > 2 && nd.inputs[2] != kNoTensor && !g.isInitializer(nd.inputs[2]))
                {
                    return false;
                }
                return true;
            }
            if (nd.type == OpType::Where || nd.type == OpType::Equal)
            {
                // flat broadcasting kernels (fixed PC arrays) decode up to kMaxRank=6 output dims.
                return g.desc(nd.outputs[0]).shape.size() <= 8;
            }
            if (nd.type == OpType::Unsqueeze)
            {
                return true; // metadata copy on the flat path
            }
            if (nd.type == OpType::Cast)
            {
                // float->float casts are a no-op copy on the unified-precision buffer; int targets stay CPU.
                DType o = g.desc(nd.outputs[0]).dtype;
                return o == DType::Float32 || o == DType::Float16;
            }
            if (nd.type == OpType::Gather)
            {
                // flat axis-aware gather; index may be a constant (uploaded) or a runtime float activation
                // (RoPE).
                return nd.inputs.size() >= 2;
            }
            if (nd.type == OpType::ScatterND)
            {
                // flat scatter; index may be a constant or a runtime float activation. Data rank within
                // kMaxRank.
                return nd.inputs.size() >= 3 && g.desc(nd.inputs[0]).shape.size() <= 8;
            }
            if (nd.type == OpType::Einsum)
            {
                // Only "i,j->ij" (outer product) has a GPU kernel; other equations use the CPU op.
                std::string eq;
                for (char c: nd.attr.gets("equation", ""))
                {
                    if (c != ' ' && c != '\t')
                    {
                        eq += c;
                    }
                }
                return eq == "i,j->ij";
            }
            if (nd.type == OpType::BatchNorm)
            {
                // per-channel affine; needs 4D input and the 4 params (gamma/beta/mean/var) as constants.
                if (nd.inputs.size() < 5 || g.desc(nd.inputs[0]).shape.size() != 4)
                {
                    return false;
                }
                for (int i = 1; i <= 4; ++i)
                {
                    if (!g.isInitializer(nd.inputs[i]))
                    {
                        return false;
                    }
                }
                return true;
            }
            if (nd.type == OpType::Split)
            {
                // NC4HW4 channel split (4D, axis 1, 4-aligned outputs) is a block copy; any other split runs
                // on the flat row-major path (a Slice per output) for rank <= kMaxRank.
                const Shape &in = g.desc(nd.inputs[0]).shape;
                if (in.empty())
                {
                    return false;
                }
                int     rank = (int) in.size();
                int64_t axis = nd.attr.geti("axis", 0);
                if (axis < 0)
                {
                    axis += rank;
                }
                if (rank == 4 && axis == 1)
                {
                    bool aligned = true;
                    for (TensorId o: nd.outputs)
                    {
                        if (o == kNoTensor)
                        {
                            continue;
                        }
                        const Shape &os = g.desc(o).shape;
                        if (os.size() != 4 || os[1] % 4 != 0)
                        {
                            aligned = false;
                        }
                    }
                    if (aligned)
                    {
                        return true;
                    }
                }
                return rank <= 8;
            }
            if (nd.type == OpType::Clip)
            {
                // const-or-absent scalar bounds (baked into the PC in prepare); runtime bounds fall back.
                for (int i = 1; i < 3 && i < (int) nd.inputs.size(); ++i)
                {
                    if (nd.inputs[i] != kNoTensor && !g.isInitializer(nd.inputs[i]))
                    {
                        return false;
                    }
                }
                return true;
            }
            // Add/Binary: 2 inputs required. The NC4HW4 kernel does same-shape + channel-broadcast; the
            // flat kernel (chosen by the layout pass) does everything else incl. constant operands.
            if (nd.type == OpType::Add || nd.type == OpType::Binary)
            {
                return nd.inputs.size() == 2;
            }
            return true;
        }

        vk::VulkanContext &ctx() {
            return *ctx_;
        }
        vk::CommandRunner &runner() {
            return *runner_;
        }
        // Read cfg.cacheFile once and split it into the pipeline + weight sections. Empty cacheFile
        // (e.g. a session built from an in-memory graph) leaves both sections empty -> caches stay
        // in-memory only and saveCaches() is a no-op.
        void loadUnified(const Config &cfg) {
            if (unifiedLoaded_)
            {
                return;
            }
            unifiedLoaded_ = true;
            cacheFile_     = cfg.cacheFile;
            savePipeline_  = cfg.cachesPipeline();
            if (cacheFile_.empty())
            {
                return;
            }
            std::ifstream f(cacheFile_, std::ios::binary);
            if (!f)
            {
                return;
            }
            loadedBytes_.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            const uint8_t *p = loadedBytes_.data();
            size_t         n = loadedBytes_.size();
            if (n < 8 || std::memcmp(p, "VKNNCAC1", 8) != 0)
            {
                return; // absent / unrecognized -> regenerate
            }
            size_t off  = 8;
            auto   rd32 = [&](uint32_t &v) {
                if (off + 4 > n)
                {
                    return false;
                }
                std::memcpy(&v, p + off, 4);
                off += 4;
                return true;
            };
            uint32_t pl = 0;
            if (rd32(pl) && off + pl <= n)
            {
                pipeInit_.assign((const char *) p + off, (const char *) p + off + pl);
                off += pl;
                uint32_t wl = 0;
                if (rd32(wl) && off + wl <= n)
                {
                    weightInit_.assign(p + off, p + off + wl);
                }
            }
        }
        vk::PipelineCache *pipelineCache(const Config &cfg) {
            if (!cache_)
            {
                loadUnified(cfg);
                cache_ = std::make_unique<vk::PipelineCache>(*ctx_, cfg.cachesPipeline() ? pipeInit_ : std::vector<char> {});
            }
            return cache_.get();
        }
        WeightCache *weightCache(const Config &cfg) {
            if (!wcache_)
            {
                loadUnified(cfg);
                wcache_          = std::make_unique<WeightCache>();
                bool keepWeights = cfg.cachesWeights() && !cacheFile_.empty();
                bool keepTune    = cfg.cachesTuning() && !cacheFile_.empty();
                wcache_->loadBytes(weightInit_.data(), weightInit_.size(), keepWeights, keepTune);
            }
            return wcache_.get();
        }
        // Write the unified cache file, but only when the serialized cache differs from what was loaded
        // (so an unchanged warm session leaves the file untouched). Called from Session::updateCache().
        void saveCaches() {
            if (cacheFile_.empty())
            {
                return;
            }
            std::vector<char>    pipe = (cache_ && savePipeline_) ? cache_->getData() : pipeInit_;
            // serialize() writes the weights only when CacheMode::Full retained them and the autotune
            // only when the mode is Tune or Full, so this one call covers every combination.
            std::vector<uint8_t> w = wcache_ ? wcache_->serialize() : weightInit_;
            std::vector<uint8_t> out;
            auto                 wr32 = [&](uint32_t v) {
                const uint8_t *b = (const uint8_t *) &v;
                out.insert(out.end(), b, b + 4);
            };
            const char magic[8] = {'V', 'K', 'N', 'N', 'C', 'A', 'C', '1'};
            out.insert(out.end(), magic, magic + 8);
            wr32((uint32_t) pipe.size());
            out.insert(out.end(), pipe.begin(), pipe.end());
            wr32((uint32_t) w.size());
            out.insert(out.end(), w.begin(), w.end());
            if (out == loadedBytes_)
            {
                return; // unchanged
            }
            std::ofstream f(cacheFile_, std::ios::binary | std::ios::trunc);
            if (!f)
            {
                VKNN_WARN << "cannot write cache file " << cacheFile_;
                return;
            }
            f.write((const char *) out.data(), (std::streamsize) out.size());
            loadedBytes_ = out;
            VKNN_INFO << "Saved cache (" << out.size() << " bytes: pipeline " << pipe.size() << " + weights " << w.size() << ") -> " << cacheFile_;
        }

        bool useFp16(const Config &cfg) const {
            return vxVulkanFp16Available() && ctx_->caps().shaderFloat16 && (cfg.precision == Precision::Fp16 || cfg.precision == Precision::Auto);
        }

        std::unique_ptr<Segment> compileSegment(const std::vector<int> &idx, Graph &g, const Config &cfg) override;
        void                     finalize() override {
            saveCaches();
        }

        // ---- host NCHW fp32  <->  device NC4HW4 (fp32 path; fp16 device buffers handled here) ----
        static void packToBuffer(vk::Buffer *buf, const RtTensor &rt, bool fp16, bool flat = false) {
            if (flat)
            { // host NCHW row-major == the flat device buffer; straight copy (+ fp16 convert)
                int64_t      n   = numElements(rt.shape);
                const float *src = rt.host.f32();
                if (fp16)
                {
                    fp16_t *dst = reinterpret_cast<fp16_t *>(buf->host());
                    for (int64_t i = 0; i < n; ++i)
                    {
                        dst[i] = floatToHalf(src[i]);
                    }
                } else
                {
                    std::memcpy(buf->host(), src, (size_t) n * 4);
                }
                return;
            }
            NCHW         x   = NCHW::from(rt.shape);
            int64_t      Cb  = cBlocks(x.c);
            const float *src = rt.host.f32();
            if (fp16)
            {
                fp16_t *dst = reinterpret_cast<fp16_t *>(buf->host());
                for (int64_t n = 0; n < x.n; ++n)
                {
                    for (int64_t cb = 0; cb < Cb; ++cb)
                    {
                        for (int64_t h = 0; h < x.h; ++h)
                        {
                            for (int64_t w = 0; w < x.w; ++w)
                            {
                                int64_t base = ((((n * Cb + cb) * x.h + h) * x.w) + w) * 4;
                                float   t[4] = {0, 0, 0, 0};
                                for (int l = 0; l < 4; ++l)
                                {
                                    int64_t c = cb * 4 + l;
                                    if (c < x.c)
                                    {
                                        t[l] = src[((n * x.c + c) * x.h + h) * x.w + w];
                                    }
                                }
#if defined(VKNN_ENABLE_NEON) && defined(__ARM_NEON)
                                // convert the 4 gathered channels to fp16 in one instruction
                                vst1_f16(reinterpret_cast<__fp16 *>(dst + base), vcvt_f16_f32(vld1q_f32(t)));
#else
                                for (int l = 0; l < 4; ++l)
                                {
                                    dst[base + l] = floatToHalf(t[l]);
                                }
#endif
                            }
                        }
                    }
                }
            } else
            {
                float *dst = reinterpret_cast<float *>(buf->host());
                for (int64_t n = 0; n < x.n; ++n)
                {
                    for (int64_t cb = 0; cb < Cb; ++cb)
                    {
                        for (int64_t h = 0; h < x.h; ++h)
                        {
                            for (int64_t w = 0; w < x.w; ++w)
                            {
                                int64_t base = ((((n * Cb + cb) * x.h + h) * x.w) + w) * 4;
                                for (int l = 0; l < 4; ++l)
                                {
                                    int64_t c     = cb * 4 + l;
                                    dst[base + l] = (c < x.c) ? src[((n * x.c + c) * x.h + h) * x.w + w] : 0.f;
                                }
                            }
                        }
                    }
                }
            }
        }
        static void unpackFromBuffer(vk::Buffer *buf, RtTensor &rt, bool fp16, bool flat = false) {
            if (flat)
            { // flat device buffer == host NCHW row-major; straight copy (+ fp16 convert)
                int64_t n = numElements(rt.shape);
                rt.host.resizeElems(n, DType::Float32);
                rt.dtype   = DType::Float32;
                float *dst = rt.host.f32();
                if (fp16)
                {
                    halfToFloatBulk(reinterpret_cast<const fp16_t *>(buf->host()), dst, n);
                } else
                {
                    std::memcpy(dst, buf->host(), (size_t) n * 4);
                }
                rt.hostValid = true;
                return;
            }
            NCHW    x  = NCHW::from(rt.shape);
            int64_t Cb = cBlocks(x.c);
            rt.host.resizeElems(x.elems(), DType::Float32);
            rt.dtype   = DType::Float32;
            float *dst = rt.host.f32();
            if (fp16)
            {
                const fp16_t *src = reinterpret_cast<const fp16_t *>(buf->host());
                for (int64_t n = 0; n < x.n; ++n)
                {
                    for (int64_t c = 0; c < x.c; ++c)
                    {
                        for (int64_t h = 0; h < x.h; ++h)
                        {
                            for (int64_t w = 0; w < x.w; ++w)
                            {
                                int64_t cb = c / 4, l = c % 4;
                                int64_t sidx                             = ((((n * Cb + cb) * x.h + h) * x.w) + w) * 4 + l;
                                dst[((n * x.c + c) * x.h + h) * x.w + w] = halfToFloat(src[sidx]);
                            }
                        }
                    }
                }
            } else
            {
                const float *src = reinterpret_cast<const float *>(buf->host());
                for (int64_t n = 0; n < x.n; ++n)
                {
                    for (int64_t c = 0; c < x.c; ++c)
                    {
                        for (int64_t h = 0; h < x.h; ++h)
                        {
                            for (int64_t w = 0; w < x.w; ++w)
                            {
                                int64_t cb = c / 4, l = c % 4;
                                int64_t sidx                             = ((((n * Cb + cb) * x.h + h) * x.w) + w) * 4 + l;
                                dst[((n * x.c + c) * x.h + h) * x.w + w] = src[sidx];
                            }
                        }
                    }
                }
            }
            rt.hostValid = true;
        }

      private:
        std::unique_ptr<vk::VulkanContext> ctx_;
        std::unique_ptr<vk::CommandRunner> runner_;
        std::unique_ptr<vk::PipelineCache> cache_;
        std::unique_ptr<WeightCache>       wcache_;
        std::string                        disabledOps_; // Config::disableVkOps (debug op-fallback list)
        // Unified per-model cache file (cfg.cacheFile): one file bundling the pipeline + weight/tuning
        // blobs, read once and split into pipeInit_/weightInit_, rewritten by saveCaches() only when the
        // serialized cache differs from what was loaded.
        std::string          cacheFile_;
        std::vector<char>    pipeInit_;
        std::vector<uint8_t> weightInit_, loadedBytes_;
        bool                 unifiedLoaded_ = false, savePipeline_ = true;
    };

    // ============================ VulkanSegment ============================
    class VulkanSegment: public Segment {
      public:
        VulkanSegment(const std::vector<int> &idx, Graph &g, const Config &cfg, VulkanBackend *be): be_(be), g_(g), cfg_(cfg) {
            nodeIdx   = idx;
            useFp16_  = be_->useFp16(cfg);
            elemSize_ = useFp16_ ? 2 : 4;

            // 1) allocate device buffers for all activation tensors (non-initializers).
            std::set<TensorId> acts;
            for (int ni: idx)
            {
                for (TensorId in: g.nodes[ni].inputs)
                {
                    if (in != kNoTensor && !g.isInitializer(in))
                    {
                        acts.insert(in);
                    }
                }
                // A fused residual (out = act(conv + residual)) is read by record() but isn't in node.inputs,
                // so it needs a buffer too — and may be produced by another segment (boundary input).
                TensorId res = g.nodes[ni].fusedResidual;
                if (res != kNoTensor && !g.isInitializer(res))
                {
                    acts.insert(res);
                }
                for (TensorId o: g.nodes[ni].outputs)
                {
                    if (o != kNoTensor)
                    {
                        acts.insert(o);
                    }
                }
            }
            // Tensors this segment produces that are read OUTSIDE it (by another segment or as a graph
            // output) get downloaded to host via unpackFromBuffer. The default kAuto memory is
            // write-combined (fast to upload, but CPU READS are uncached and brutally slow -> 150ms on a
            // YOLO head boundary). Allocate those readback buffers as HOST_CACHED so the download is fast;
            // keep the rest as kAuto.
            std::set<int>      idxSet(idx.begin(), idx.end());
            std::set<TensorId> readBack(g.outputs.begin(), g.outputs.end());
            {
                std::set<TensorId> produced;
                for (int ni: idx)
                {
                    for (TensorId o: g.nodes[ni].outputs)
                    {
                        if (o != kNoTensor)
                        {
                            produced.insert(o);
                        }
                    }
                }
                for (size_t q = 0; q < g.nodes.size(); ++q)
                {
                    if (idxSet.count((int) q))
                    {
                        continue;
                    }
                    for (TensorId in: g.nodes[q].inputs)
                    {
                        if (in != kNoTensor && produced.count(in))
                        {
                            readBack.insert(in);
                        }
                    }
                }
            }
            // Debug: Config::dumpTensors="substr1,substr2" forces matching tensors to dedicated (un-aliased)
            // readback buffers and dumps them to /data/local/tmp/vxrt/dump after the run — so intermediate
            // activations can be diffed despite the liveness planner reusing buffers. A few tensors only.
            if (!cfg_.dumpTensors.empty())
            {
                std::string list = cfg_.dumpTensors;
                for (TensorId tid: acts)
                {
                    const std::string &nm = g.tensors[tid].name;
                    if (nm.empty())
                    {
                        continue;
                    }
                    size_t pos = 0, comma;
                    do
                    {
                        comma           = list.find(',', pos);
                        std::string sub = list.substr(pos, comma == std::string::npos ? comma : comma - pos);
                        if (!sub.empty() && nm.find(sub) != std::string::npos)
                        {
                            readBack.insert(tid);
                            dumpTids_.push_back(tid);
                            break;
                        }
                        pos = comma + 1;
                    } while (comma != std::string::npos);
                }
            }
            auto actBytes = [&](TensorId tid) -> size_t {
                int64_t elems = g.tensors[tid].gpuFlat ? numElements(g.tensors[tid].shape) : packedElems(g.tensors[tid].shape);
                int     es    = g.tensors[tid].storeFp32 ? 4 : elemSize_; // selective-fp32 tensors keep 4-byte storage
                size_t  b     = (size_t) elems * es;
                return b == 0 ? (size_t) elemSize_ * 4 : b;
            };
            // Liveness buffer planner. One buffer per tensor keeps ALL activations live at once (~11.5GB on
            // the YoNoSplat encoder); the simultaneously-live peak is ~0.17GB. Boundary-in (produced by
            // another segment) and readback (read by host / another segment) tensors get dedicated buffers;
            // internal (produced-and-consumed only here) tensors are pooled by a greedy scan over execution
            // order, reusing a buffer once its previous occupant's last use has passed. The buffer-level
            // barriers in record() make the write-after-read at each reuse point safe.
            std::set<TensorId> producedHere;
            for (int ni: idx)
            {
                for (TensorId o: g.nodes[ni].outputs)
                {
                    if (o != kNoTensor)
                    {
                        producedHere.insert(o);
                    }
                }
            }
            for (TensorId tid: acts)
            {
                // storeFp32 tensors get a dedicated buffer (never pooled): the liveness pool aliases by
                // byte size only, so a 4-byte tensor must not share a slot sized for 2-byte neighbours.
                bool internal = producedHere.count(tid) && !readBack.count(tid) && !g.tensors[tid].storeFp32;
                if (internal)
                {
                    continue; // pooled below
                }
                auto pref     = readBack.count(tid) ? vk::MemPref::kReadback : vk::MemPref::kAuto;
                buffers_[tid] = std::make_shared<vk::Buffer>(be_->ctx(), actBytes(tid), pref, 0, /*zeroInit=*/true);
            }
            // [firstPos,lastPos] of each internal tensor within this segment's execution order
            std::map<TensorId, int> firstPos, lastPos;
            auto                    touch = [&](TensorId t, int k) {
                if (t == kNoTensor || !producedHere.count(t) || readBack.count(t) || g.tensors[t].storeFp32)
                {
                    return; // dedicated (storeFp32) and boundary tensors are not pooled
                }
                if (!firstPos.count(t))
                {
                    firstPos[t] = k;
                }
                lastPos[t] = k;
            };
            for (int k = 0; k < (int) idx.size(); ++k)
            {
                const Node &nd = g.nodes[idx[k]];
                for (TensorId in: nd.inputs)
                {
                    touch(in, k);
                }
                touch(nd.fusedResidual, k);
                for (TensorId o: nd.outputs)
                {
                    touch(o, k);
                }
            }
            std::vector<TensorId> order;
            order.reserve(firstPos.size());
            for (auto &kv: firstPos)
            {
                order.push_back(kv.first);
            }
            std::sort(order.begin(), order.end(), [&](TensorId a, TensorId b) {
                return firstPos[a] < firstPos[b];
            });
            struct Slot {
                std::shared_ptr<vk::Buffer> buf;
                size_t                      cap;
                int                         deadAt;
            };
            std::vector<Slot> busy, freeSlots;
            for (TensorId tid: order)
            {
                int p = firstPos[tid];
                for (size_t i = 0; i < busy.size();)
                {
                    if (busy[i].deadAt < p)
                    {
                        freeSlots.push_back(busy[i]);
                        busy[i] = busy.back();
                        busy.pop_back();
                    } else
                    {
                        ++i;
                    }
                }
                size_t need = actBytes(tid);
                int    best = -1;
                for (size_t i = 0; i < freeSlots.size(); ++i)
                {
                    if (freeSlots[i].cap >= need && (best < 0 || freeSlots[i].cap < freeSlots[best].cap))
                    {
                        best = (int) i;
                    }
                }
                Slot s;
                if (best >= 0)
                {
                    s               = freeSlots[best];
                    freeSlots[best] = freeSlots.back();
                    freeSlots.pop_back();
                } else
                {
                    s.buf = std::make_shared<vk::Buffer>(be_->ctx(), need, vk::MemPref::kAuto, 0, /*zeroInit=*/true);
                    s.cap = need;
                }
                s.deadAt      = lastPos[tid];
                buffers_[tid] = s.buf;
                busy.push_back(s);
            }

            // 2) build env + ops; prepare uploads weights.
            env_.backend  = be_;
            env_.ctx      = &be_->ctx();
            env_.cache    = be_->pipelineCache(cfg);
            env_.weights  = be_->weightCache(cfg);
            env_.runner   = &be_->runner();
            env_.tuning   = (Mode) cfg.hint(Hint::Tuning, (int) Mode::Fast);
            env_.winograd = (Mode) cfg.hint(Hint::Winograd, (int) Mode::Auto);
            env_.graph    = &g;
            env_.config   = &cfg;
            env_.useFp16  = useFp16_;
            env_.baseFp16 = useFp16_; // segment-wide precision; useFp16_ is overridden per-node below for storeFp32 nodes
            // per-model weight-cache namespace: FNV-1a over the whole graph (same for every segment of this
            // model, distinct across models) so a shared cacheDir can't return another model's weights.
            {
                uint64_t h   = 1469598103934665603ull;
                auto     mix = [&](const std::string &s) {
                    for (char c: s)
                    {
                        h ^= (uint8_t) c;
                        h *= 1099511628211ull;
                    }
                };
                for (const auto &nd: g.nodes)
                {
                    mix(nd.name);
                    mix(opTypeName(nd.type));
                }
                mix(std::to_string(g.nodes.size()));
                char buf[20];
                snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) h);
                env_.modelTag = buf;
            }
            { // per-GPU autotune namespace: vendor/device/driver identify the kernel-timing target.
                const auto &c = be_->ctx().caps();
                char        g[40];
                snprintf(g, sizeof(g), "%04x%04x-%08x", c.vendorID, c.deviceID, c.driverVersion);
                env_.gpuTag = g;
            }
            env_.devBuf = [this](TensorId t) -> vk::Buffer * {
                auto it = buffers_.find(t);
                return it == buffers_.end() ? nullptr : it->second.get();
            };
            for (int ni: idx)
            {
                auto op = VkOpRegistry::instance().create(g.nodes[ni].type);
                if (!op)
                {
                    throw Error(Status::Unsupported, std::string("no Vulkan kernel for ") + opTypeName(g.nodes[ni].type));
                }
                // A storeFp32 node (its output kept in fp32) selects its fp32 kernel variant + uploads its
                // weights fp32; ConvertDtype reads the precision per tensor and ignores this.
                env_.useFp16 = nodeFp32(g.nodes[ni]) ? false : useFp16_;
                op->prepare(g.nodes[ni], env_);
                ops_.push_back(std::move(op));
            }
            env_.useFp16 = useFp16_;

            // 3) timestamp query pool (2 per node). Only when profiling - the extra writes + the implicit
            //    barriers around them aren't free, and we don't want them on the hot path.
            if (be_->ctx().caps().timestampSupported && cfg.profile)
            {
                VkQueryPoolCreateInfo qi {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
                qi.queryType  = VK_QUERY_TYPE_TIMESTAMP;
                qi.queryCount = (uint32_t) (idx.size() * 2);
                vkCreateQueryPool(be_->ctx().device(), &qi, nullptr, &queryPool_);
            }

            // 4) pre-record the command buffer for the static graph.
            record();
        }

        ~VulkanSegment() override {
            if (queryPool_)
            {
                vkDestroyQueryPool(be_->ctx().device(), queryPool_, nullptr);
            }
        }

        // A node runs in fp32 (selects its fp32 kernel + 4-byte buffers) when its output is storeFp32.
        bool nodeFp32(const Node &nd) const {
            return !nd.outputs.empty() && nd.outputs[0] != kNoTensor && g_.tensors[nd.outputs[0]].storeFp32;
        }

        void record() {
            cmd_ = be_->runner().allocate();
            be_->runner().begin(cmd_);
            if (queryPool_)
            {
                vkCmdResetQueryPool(cmd_, queryPool_, 0, (uint32_t) (nodeIdx.size() * 2));
            }
            // Declared-format zero-copy inputs: convert each caller dma-buf (declared layout/dtype) into
            // this segment's device-native boundary buffer, then a barrier before the ops read it.
            {
                bool any = false;
                for (const auto &kv: convert_)
                {
                    if (!kv.second.isInput)
                    {
                        continue;
                    }
                    const ConvertBinding &c = kv.second;
                    if (!conv_)
                    {
                        conv_ = std::make_unique<BoundaryConvert>();
                    }
                    conv_->record(cmd_, *env_.ctx, env_.cache, c.imported.get(), buffers_[kv.first].get(), c.shape, c.declFmt, c.declDtype, c.devFmt, c.devDtype);
                    any = true;
                }
                if (any)
                {
                    vk::computeBarrier(cmd_);
                }
            }
            auto isCopy = [&](int idx) {
                const Node &nn = g_.nodes[idx];
                OpType      t  = nn.type;
                // A flat split is a compute dispatch (flat_gather); the NC4HW4 split is a buffer copy.
                if (t == OpType::Split)
                {
                    return nn.outputs.empty() || nn.outputs[0] == kNoTensor || !g_.desc(nn.outputs[0]).gpuFlat;
                }
                // Reshape/Flatten/Squeeze/Unsqueeze/Cast are vkCmdCopyBuffer (transfer-stage writes).
                return t == OpType::Reshape || t == OpType::Flatten || t == OpType::Squeeze || t == OpType::Unsqueeze || t == OpType::Cast;
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
            // For non-aliased buffers this reduces to per-tensor read-after-write (single writer per
            // buffer), so independent-op overlap (Inception/YOLO) is preserved.
            std::set<vk::Buffer *> writtenBufs, readBufs;
            auto                   bufOf = [&](TensorId t) -> vk::Buffer                   *{
                if (t == kNoTensor)
                {
                    return nullptr;
                }
                auto it = buffers_.find(t);
                return it == buffers_.end() ? nullptr : it->second.get();
            };
            bool copySinceBarrier = false;
            for (size_t k = 0; k < nodeIdx.size(); ++k)
            {
                const Node &node        = g_.nodes[nodeIdx[k]];
                bool        needBarrier = perOpBarrier;
                if (!needBarrier)
                {
                    for (TensorId in: node.inputs) // read-after-write
                    {
                        if (vk::Buffer *b = bufOf(in))
                        {
                            if (writtenBufs.count(b))
                            {
                                needBarrier = true;
                                break;
                            }
                        }
                    }
                    if (!needBarrier)
                    {
                        if (vk::Buffer *b = bufOf(node.fusedResidual))
                        {
                            if (writtenBufs.count(b))
                            {
                                needBarrier = true;
                            }
                        }
                    }
                    if (!needBarrier)
                    {
                        for (TensorId o: node.outputs) // write-after-write / write-after-read (reused buffer)
                        {
                            if (vk::Buffer *b = bufOf(o))
                            {
                                if (writtenBufs.count(b) || readBufs.count(b))
                                {
                                    needBarrier = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                if (needBarrier)
                {
                    if (copySinceBarrier || isCopy(nodeIdx[k]))
                    {
                        vk::transferBarrier(cmd_);
                    } else
                    {
                        vk::computeBarrier(cmd_);
                    }
                    writtenBufs.clear();
                    readBufs.clear();
                    copySinceBarrier = false;
                }
                if (queryPool_)
                {
                    vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool_, (uint32_t) (k * 2));
                }
                env_.useFp16 = nodeFp32(node) ? false : useFp16_; // match the variant chosen in prepare()
                ops_[k]->record(cmd_, node, env_);
                if (queryPool_)
                {
                    vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool_, (uint32_t) (k * 2 + 1));
                }
                for (TensorId in: node.inputs)
                {
                    if (vk::Buffer *b = bufOf(in))
                    {
                        readBufs.insert(b);
                    }
                }
                if (vk::Buffer *b = bufOf(node.fusedResidual))
                {
                    readBufs.insert(b);
                }
                for (TensorId o: node.outputs)
                {
                    if (vk::Buffer *b = bufOf(o))
                    {
                        writtenBufs.insert(b);
                    }
                }
                if (isCopy(nodeIdx[k]))
                {
                    copySinceBarrier = true;
                }
                // Split the segment into multiple command buffers so no single submit runs long
                // enough to trip the GPU watchdog (a ~20s single submit on this driver gets reset
                // silently, zeroing the unexecuted tail). The submit fence between chunks is a full
                // barrier, so buffer reuse stays correct across the boundary. Config::maxSubmitNodes
                // controls the chunk size (0 disables). Only when not profiling.
                const int chunkNodes = cfg_.maxSubmitNodes;
                if (!queryPool_ && chunkNodes > 0 && (k + 1) % chunkNodes == 0 && k + 1 < nodeIdx.size())
                {
                    be_->runner().end(cmd_);
                    cmds_.push_back(cmd_);
                    cmd_ = be_->runner().allocate();
                    be_->runner().begin(cmd_);
                    writtenBufs.clear();
                    readBufs.clear();
                    copySinceBarrier = false;
                }
            }
            // Final barrier so the segment outputs are complete + visible before the host reads them.
            if (copySinceBarrier)
            {
                vk::transferBarrier(cmd_);
            } else
            {
                vk::computeBarrier(cmd_);
            }
            // Declared-format zero-copy outputs: convert the device-native boundary buffer into each
            // caller dma-buf (declared layout/dtype), then a barrier before the host reads it.
            {
                bool any = false;
                for (const auto &kv: convert_)
                {
                    if (kv.second.isInput)
                    {
                        continue;
                    }
                    const ConvertBinding &c = kv.second;
                    if (!conv_)
                    {
                        conv_ = std::make_unique<BoundaryConvert>();
                    }
                    conv_->record(cmd_, *env_.ctx, env_.cache, buffers_[kv.first].get(), c.imported.get(), c.shape, c.devFmt, c.devDtype, c.declFmt, c.declDtype);
                    any = true;
                }
                if (any)
                {
                    vk::computeBarrier(cmd_);
                }
            }
            be_->runner().end(cmd_);
            cmds_.push_back(cmd_);
            recorded_        = true;
            recordedConvert_ = convert_;
        }

        void run(ExecContext &ctx) override {
            const bool timing = cfg_.timing;
            auto       now    = [] {
                return std::chrono::high_resolution_clock::now();
            };
            auto t0 = now();
            // --- zero-copy: bind a caller dma-buf fd (rt.dmaBufFd) as the boundary GPU buffer so the GPU
            //     reads/writes it directly. Re-record the command buffer when the bound-buffer set changes;
            //     the imported buffer is cached by fd, so a reused dma-buf re-records once. Import failure
            //     keeps the pooled buffer (the copy path). Boundary I/O buffers are dedicated (not
            //     pool-aliased), so swapping them is safe.
            {
                bool reRecord = false;
                convert_.clear();
                auto rebind = [&](TensorId tid, bool isInput) {
                    auto bit = buffers_.find(tid);
                    if (bit == buffers_.end())
                    {
                        return;
                    }
                    if (!origBoundary_.count(tid))
                    {
                        origBoundary_[tid] = bit->second; // snapshot the pooled boundary buffer
                    }
                    std::shared_ptr<vk::Buffer> want = origBoundary_[tid];
                    RtTensor                   &rt   = ctx.t(tid);
                    int                         fd   = rt.dmaBufFd;
                    if (fd >= 0)
                    {
                        bool         flat    = g_.desc(tid).gpuFlat;
                        TensorFormat devFmt  = flat ? TensorFormat::NCHW : TensorFormat::NC4HW4;
                        DType        devDt   = useFp16_ ? DType::Float16 : DType::Float32;
                        TensorFormat declFmt = rt.dmaBufFormat;
                        DType        declDt  = rt.dmaBufDtype;
                        bool         direct  = declFmt == TensorFormat::Auto || (declFmt == devFmt && declDt == devDt);
                        NCHW         x       = NCHW::from(rt.shape.empty() ? g_.tensors[tid].shape : rt.shape);
                        // Import sized for what the dma-buf actually holds: the device-native bytes for a
                        // direct bind, the declared-format bytes for a convert. Re-import when this
                        // tensor's fd or size changes.
                        size_t needB = direct ? origBoundary_[tid]->bytes() : (size_t) (formatElems(declFmt, x) * dtypeSize(declDt));
                        if (needB > 0)
                        {
                            uint64_t  id    = dmaBufId(fd);
                            Imported &imp   = imported_[tid];
                            bool      stale = !imp.buf || imp.bytes != needB || (id != 0 ? imp.id != id : imp.fd != fd);
                            if (stale)
                            {
                                vk::Buffer *b = vk::Buffer::importDmaBufFd(be_->ctx(), fd, needB);
                                imp           = {id, fd, needB, b ? std::shared_ptr<vk::Buffer>(b) : nullptr};
                                if (!b)
                                {
                                    // No dma-buf import on this device: zero-copy can't be honored. The
                                    // pooled buffer holds no caller data, so the result for this input is
                                    // invalid — surface it rather than read silently undefined memory.
                                    VKNN_WARN_THROTTLE("zerocopy-import-fail", 1) << "dma-buf import failed for '" << g_.tensors[tid].name << "' (device lacks dma-buf import); zero-copy unavailable";
                                }
                            }
                            if (imp.buf)
                            {
                                if (direct)
                                {
                                    want = imp.buf; // declared == device-native: bind the fd directly
                                } else
                                {
                                    // declared != device-native: keep the pooled boundary buffer; the GPU
                                    // converts between the imported buffer and it (recorded in record()).
                                    ConvertBinding cb;
                                    cb.imported   = imp.buf;
                                    cb.isInput    = isInput;
                                    cb.shape      = x;
                                    cb.declFmt    = declFmt;
                                    cb.declDtype  = declDt;
                                    cb.devFmt     = devFmt;
                                    cb.devDtype   = devDt;
                                    convert_[tid] = cb;
                                }
                            }
                        }
                    }
                    if (bit->second != want)
                    {
                        bit->second = want;
                        reRecord    = true;
                    }
                };
                for (TensorId tid: boundaryInputs)
                {
                    rebind(tid, true);
                }
                for (TensorId tid: boundaryOutputs)
                {
                    rebind(tid, false);
                }
                if (!sameConvert(convert_, recordedConvert_))
                {
                    reRecord = true;
                }
                if (reRecord)
                {
                    if (!cmds_.empty())
                    {
                        vkFreeCommandBuffers(be_->ctx().device(), be_->runner().pool(), (uint32_t) cmds_.size(), cmds_.data());
                        cmds_.clear();
                    }
                    record();
                }
            }
            // attach boundary buffers to RtTensors (cross-segment residency) + upload inputs.
            // Each segment owns a SEPARATE buffer per tensor, so a boundary input must be (re)packed into
            // THIS segment's buffer unless that exact buffer already holds the data. Matching on the exact
            // buffer (not just rt.deviceValid) is required: a tensor produced by an earlier GPU segment is
            // deviceValid but points at that segment's buffer, so this segment must repack into its own.
            for (TensorId tid: boundaryInputs)
            {
                RtTensor &rt  = ctx.t(tid);
                auto      bit = buffers_.find(tid);
                if (bit == buffers_.end())
                {
                    continue;
                }
                bool alreadyHere = rt.deviceValid && rt.device && rt.device->buffer == bit->second;
                bool flat        = g_.desc(tid).gpuFlat;
                if (!rt.device)
                {
                    rt.device = std::make_shared<DeviceStorage>();
                }
                rt.device->buffer = bit->second;
                if (rt.dmaBufFd >= 0)
                {
                    // zero-copy: the GPU reads the caller's dma-buf directly (device-native bytes); no pack.
                    rt.deviceValid  = true;
                    rt.deviceFormat = flat ? TensorFormat::NCHW : TensorFormat::NC4HW4;
                } else if (rt.hostValid && !alreadyHere)
                {
                    VulkanBackend::packToBuffer(bit->second.get(), rt, useFp16_, flat);
                    rt.deviceValid  = true;
                    rt.deviceFormat = flat ? TensorFormat::NCHW : TensorFormat::NC4HW4;
                }
            }
            auto t1 = now();

            double wall = 0;
            for (VkCommandBuffer c: cmds_)
            {
                wall += be_->runner().submitAndWait(c);
            }
            auto t2 = now();

            // download boundary outputs to host.
            for (TensorId tid: boundaryOutputs)
            {
                auto bit = buffers_.find(tid);
                if (bit == buffers_.end())
                {
                    continue;
                }
                RtTensor &rt   = ctx.t(tid);
                bool      flat = g_.desc(tid).gpuFlat;
                if (!rt.device)
                {
                    rt.device = std::make_shared<DeviceStorage>();
                }
                rt.device->buffer = bit->second;
                rt.deviceValid    = true;
                rt.deviceFormat   = flat ? TensorFormat::NCHW : TensorFormat::NC4HW4;
                if (rt.dmaBufFd < 0)
                {
                    VulkanBackend::unpackFromBuffer(bit->second.get(), rt, useFp16_ && !g_.tensors[tid].storeFp32, flat);
                }
                // else: the GPU wrote device-native bytes straight into the caller's dma-buf; caller reads it.
            }
            if (timing)
            {
                auto t3 = now();
                auto ms = [&](auto a, auto b) {
                    return std::chrono::duration<double, std::milli>(b - a).count();
                };
                VKNN_INFO << "timing: pack=" << ms(t0, t1) << "ms submit+gpu=" << wall << "ms unpack=" << ms(t2, t3) << "ms";
            }

            // Config::dumpTensors targeted dump: write the named tensors (dedicated buffers) to disk for
            // diffing.
            if (!dumpTids_.empty())
            {
                ::mkdir("/data/local/tmp/vxrt/dump", 0755);
                for (TensorId tid: dumpTids_)
                {
                    auto bit = buffers_.find(tid);
                    if (bit == buffers_.end())
                    {
                        continue;
                    }
                    RtTensor &rt = ctx.t(tid);
                    VulkanBackend::unpackFromBuffer(bit->second.get(), rt, useFp16_ && !g_.tensors[tid].storeFp32, g_.desc(tid).gpuFlat);
                    std::string nm = g_.tensors[tid].name;
                    for (char &c: nm)
                    {
                        if (c == '/' || c == ':')
                        {
                            c = '_';
                        }
                    }
                    FILE *f = fopen(("/data/local/tmp/vxrt/dump/" + nm + ".bin").c_str(), "wb");
                    if (f)
                    {
                        fwrite(rt.host.bytes.data(), 1, rt.host.bytes.size(), f);
                        fclose(f);
                    }
                }
            }
            // layer-dump: bring every activation back to host for per-layer diffing.
            if (ctx.config && ctx.config->layerDump)
            {
                for (auto &kv: buffers_)
                {
                    RtTensor &rt = ctx.t(kv.first);
                    if (g_.isInitializer(kv.first))
                    {
                        continue;
                    }
                    VulkanBackend::unpackFromBuffer(kv.second.get(), rt, useFp16_ && !g_.tensors[kv.first].storeFp32, g_.desc(kv.first).gpuFlat);
                }
            }

            // profiler: per-node GPU time from timestamps + dispatch dims.
            if (ctx.profiler && ctx.profiler->enabled() && queryPool_)
            {
                std::vector<uint64_t> ts(nodeIdx.size() * 2, 0);
                vkGetQueryPoolResults(be_->ctx().device(), queryPool_, 0, (uint32_t) ts.size(), ts.size() * sizeof(uint64_t), ts.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                double period = be_->ctx().caps().timestampPeriod;
                for (size_t k = 0; k < nodeIdx.size(); ++k)
                {
                    const Node &node = g_.nodes[nodeIdx[k]];
                    OpRecord    r;
                    r.name    = node.name;
                    r.type    = node.type;
                    r.backend = "Vulkan";
                    r.gpuMs   = (double) (ts[k * 2 + 1] - ts[k * 2]) * period / 1e6;
                    r.cpuMs   = 0;
                    ctx.profiler->add(r);
                }
                // GPU span (first dispatch start -> last dispatch end) vs the CPU-side submit wall: the
                // difference is barrier bubbles + submit/fence latency, not kernel work.
                double span = (double) (ts.back() - ts.front()) * period / 1e6;
                VKNN_INFO << "gpu span=" << span << "ms  submit-wall=" << wall << "ms  (gap = overhead)";
            }
        }

      private:
        VulkanBackend                                  *be_;
        Graph                                          &g_;
        const Config                                   &cfg_;
        bool                                            useFp16_  = false;
        int                                             elemSize_ = 4;
        std::map<TensorId, std::shared_ptr<vk::Buffer>> buffers_;
        std::vector<std::unique_ptr<VulkanOp>>          ops_;
        VkOpEnv                                         env_;
        VkCommandBuffer                                 cmd_ = VK_NULL_HANDLE;
        std::vector<VkCommandBuffer> cmds_; // chunked submits (one entry unless the segment is split for the GPU watchdog; see Config::maxSubmitNodes)
        VkQueryPool                  queryPool_ = VK_NULL_HANDLE;
        bool                         recorded_  = false;
        std::vector<TensorId>        dumpTids_; // Config::dumpTensors debug: tensors to dump after the run
        // Zero-copy: each boundary tensor's pooled buffer (the fallback) and its imported dma-buf. The
        // import is kept per boundary tensor and refreshed when that tensor's dma-buf or required size
        // changes. Identity is the dma-buf's (device, inode) from fstat, not the fd: fd numbers are
        // recycled by the OS, so keying on the raw fd would alias a reused number to a stale buffer.
        std::map<TensorId, std::shared_ptr<vk::Buffer>> origBoundary_;
        struct Imported {
            uint64_t                    id    = 0; // dma-buf (dev,inode) hash (0 = unknown, fall back to fd)
            int                         fd    = -1;
            size_t                      bytes = 0;
            std::shared_ptr<vk::Buffer> buf;
        };
        std::map<TensorId, Imported> imported_;
        static uint64_t              dmaBufId(int fd) {
            struct stat st;
            if (::fstat(fd, &st) != 0)
            {
                return 0;
            }
            return ((uint64_t) st.st_dev * 1099511628211ull) ^ (uint64_t) st.st_ino; // (dev,inode) -> stable id
        }
        // Declared-format zero-copy: boundary tensors whose declared dma-buf layout/dtype differs from the
        // device-native boundary, so the GPU converts between the imported buffer and the pooled boundary
        // buffer instead of binding the fd directly. `convert_` is rebuilt each run; `recordedConvert_` is
        // what the current command buffer encodes (a change re-records).
        struct ConvertBinding {
            std::shared_ptr<vk::Buffer> imported;
            bool                        isInput = true;
            NCHW                        shape;
            TensorFormat                declFmt   = TensorFormat::NCHW;
            DType                       declDtype = DType::Float32;
            TensorFormat                devFmt    = TensorFormat::NCHW;
            DType                       devDtype  = DType::Float32;
        };
        std::map<TensorId, ConvertBinding> convert_, recordedConvert_;
        std::unique_ptr<BoundaryConvert>   conv_;
        static bool                        sameConvert(const std::map<TensorId, ConvertBinding> &a, const std::map<TensorId, ConvertBinding> &b) {
            if (a.size() != b.size())
            {
                return false;
            }
            for (const auto &kv: a)
            {
                auto it = b.find(kv.first);
                if (it == b.end())
                {
                    return false;
                }
                const ConvertBinding &x = kv.second, &y = it->second;
                if (x.imported.get() != y.imported.get() || x.isInput != y.isInput || x.declFmt != y.declFmt || x.declDtype != y.declDtype || x.devFmt != y.devFmt || x.devDtype != y.devDtype ||
                    x.shape.n != y.shape.n || x.shape.c != y.shape.c || x.shape.h != y.shape.h || x.shape.w != y.shape.w)
                {
                    return false;
                }
            }
            return true;
        }
    };

    std::unique_ptr<Segment> VulkanBackend::compileSegment(const std::vector<int> &idx, Graph &g, const Config &cfg) {
        auto s     = std::make_unique<VulkanSegment>(idx, g, cfg, this);
        s->backend = this;
        return s;
    }

    VKNN_REGISTER_BACKEND(BackendKind::Vulkan, VulkanBackend);

} // namespace vknn
