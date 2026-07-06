#include "vk_backend.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sys/stat.h>
#include <unistd.h>
#if defined(VKNN_ENABLE_NEON) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include "core/vk_gates.h" // vkNodeGate/vkKernelDeclared (shared capability model)
#include "import/passes.h" // readI64Param (raster-core view-eligibility diagnostic)
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
    void WeightCache::loadFrom(const CacheVariant &v) {
        weights_ = v.weights;
        tune_.clear();
        for (const auto &kv: v.tune)
        {
            tune_[kv.first] = (int) kv.second;
        }
        enabled_ = true;
        dirty_   = false;
        VKNN_INFO << "WeightCache: loaded " << weights_.size() << " prepacked weights, " << tune_.size() << " tuning entries";
    }
    void WeightCache::writeInto(CacheVariant &v) const {
        v.weights = weights_;
        v.tune.clear();
        for (const auto &kv: tune_)
        {
            v.tune[kv.first] = (int32_t) kv.second;
        }
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
    /// The Vulkan backend orchestrator. Owns the device context + command runner, the op registry gate
    /// (supports/supportsNode decide which nodes run on the GPU vs fall back to the CPU op), the shared
    /// pipeline and content-addressed constant pools, and the multi-variant model cache. One instance
    /// serves exactly one model (one Session): the loaded cache document and selected variant key stay
    /// valid for its whole lifetime. Per-node execution and buffer planning live in VulkanSegment.
    class VulkanBackend: public Backend {
      public:
        // The queue priority is applied at device/queue creation, so it must be known here (before
        // configure() runs) - the backend factory passes the session Config for exactly this.
        explicit VulkanBackend(const Config &cfg = {}) {
            ctx_ = std::make_unique<vk::VulkanContext>(cfg.priority);
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
            // to the CPU path. Entries match whole op-type names ("Conv" leaves ConvTranspose,
            // ConvGemm and ConvertLayout on the GPU).
            if (!disabledOps_.empty() && Config::listContains(disabledOps_, opTypeName(t)))
            {
                return false;
            }
            bool registered = VkOpRegistry::instance().has(t);
            // vkKernelDeclared (core) mirrors this registry for builds without the Vulkan backend
            // (the host support report); a disagreement means the table missed a kernel add/remove.
            if (registered != vkKernelDeclared(t))
            {
                VKNN_WARN_THROTTLE(std::string("vk_kernel_decl_") + opTypeName(t), 1) << "vkKernelDeclared(" << opTypeName(t) << ") disagrees with the live registry (" << (registered ? "registered" : "missing")
                                                                                      << ") - update src/core/vk_gates.cpp";
            }
            return registered;
        }

        // Shape-aware gate behind per-node assignment. The shape/attribute logic is vkNodeGate
        // (src/core/vk_gates.cpp) — a pure function shared with the host support report — behind
        // the availability/disable/registry pre-checks, which name their refusal here.
        bool supportsNode(const Graph &g, const Node &nd, DType dt, std::string *whyNot = nullptr) const override {
            if (!supports(nd.type, dt))
            {
                if (whyNot)
                {
                    if (!available())
                    {
                        *whyNot = "vulkan backend unavailable";
                    } else if (!disabledOps_.empty() && Config::listContains(disabledOps_, opTypeName(nd.type)))
                    {
                        *whyNot = std::string(opTypeName(nd.type)) + ": disabled via Config::disableVkOps";
                    } else
                    {
                        *whyNot = "no vulkan kernel registered";
                    }
                }
                return false;
            }
            return vkNodeGate(g, nd, whyNot);
        }

        vk::VulkanContext &ctx() {
            return *ctx_;
        }
        vk::CommandRunner &runner() {
            return *runner_;
        }
        // The cache-affecting configuration that keys a cache variant (see cache_codec.h). Two configs
        // with an equal key produce identical compiled artifacts and share a variant.
        static CacheVariant variantKey(const Config &cfg) {
            CacheVariant k;
            k.precision       = cfg.precision == Precision::High ? "high" : cfg.precision == Precision::Normal ? "normal" : "low";
            k.flatLayout      = cfg.flatLayout();
            k.gpuIslandFold   = cfg.gpuIslandFold();
            k.fp32Tensors     = cfg.fp32Tensors;
            k.winograd        = cfg.hint(Hint::Winograd, (int) Mode::Auto);
            k.winogradVariant = cfg.hint(Hint::WinogradVariant, 0);
            k.winogradUnit    = cfg.hint(Hint::WinogradUnit, 0);
            k.directConv3x3   = cfg.hint(Hint::DirectConv3x3, 0);
            return k;
        }
        // Load + validate the model cache once, selecting the variant for this config. Caching is
        // always-on: a valid file's matching variant primes the pipeline + weight caches for a warm
        // start; a missing/invalid file (or a config with no cached variant yet) starts empty and this
        // variant is built at load and appended on save. Whole-file guards: format + kernel hash
        // (embedded SPIR-V) + device (vendor/device/driver + pipeline-cache UUID) + model hash. An
        // in-memory graph (empty cacheFile) or cfg.noCache stays memory-only.
        void loadCache(const Config &cfg, const std::string &modelHash) {
            if (cacheLoaded_)
            {
                return;
            }
            cacheLoaded_ = true;
            noCache_     = cfg.noCache;
            cacheFile_   = cfg.cacheFile;
            curKey_      = variantKey(cfg);

            const auto &caps        = ctx_->caps();
            cacheDoc_               = CacheDoc {};
            cacheDoc_.format        = kCacheFormat;
            cacheDoc_.kernelHash    = embeddedShadersHash();
            cacheDoc_.vendorId      = caps.vendorID;
            cacheDoc_.deviceId      = caps.deviceID;
            cacheDoc_.driverVersion = caps.driverVersion;
            cacheDoc_.pipelineCacheUUID.assign(caps.pipelineCacheUUID, caps.pipelineCacheUUID + sizeof(caps.pipelineCacheUUID));
            cacheDoc_.model = modelHash;

            std::vector<char>   pipeInit;
            const CacheVariant *matched = nullptr;
            if (!noCache_ && !cacheFile_.empty())
            {
                std::ifstream f(cacheFile_, std::ios::binary);
                if (f)
                {
                    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                    CacheDoc             loaded;
                    if (cacheDecode(bytes.data(), bytes.size(), loaded) && loaded.format == cacheDoc_.format &&
                        loaded.kernelHash == cacheDoc_.kernelHash && loaded.vendorId == cacheDoc_.vendorId &&
                        loaded.deviceId == cacheDoc_.deviceId && loaded.driverVersion == cacheDoc_.driverVersion &&
                        loaded.pipelineCacheUUID == cacheDoc_.pipelineCacheUUID && loaded.model == cacheDoc_.model)
                    {
                        cacheDoc_    = std::move(loaded); // keep every cached variant
                        loadedBytes_ = std::move(bytes);
                        matched      = cacheDoc_.findVariant(curKey_);
                        if (matched)
                        {
                            pipeInit.assign(matched->pipeline.begin(), matched->pipeline.end());
                        }
                    } else
                    {
                        VKNN_INFO << "cache " << cacheFile_ << " does not match this device/model/kernels -> recompiling";
                    }
                }
            }
            cache_  = std::make_unique<vk::PipelineCache>(*ctx_, pipeInit);
            wcache_ = std::make_unique<WeightCache>();
            if (matched)
            {
                wcache_->loadFrom(*matched);
            } else
            {
                wcache_->reset(!noCache_ && !cacheFile_.empty());
            }
        }
        vk::PipelineCache *pipelineCache() noexcept {
            return cache_.get();
        }
        WeightCache *weightCache() noexcept {
            return wcache_.get();
        }
        // Update this config's variant in the loaded document and rewrite the cache file, but only when
        // the serialized bytes changed (an unchanged warm session leaves the file untouched). Called from
        // Session::updateCache() at teardown.
        void saveCaches() {
            if (noCache_ || cacheFile_.empty() || !cache_ || !wcache_)
            {
                return;
            }
            CacheVariant      v    = curKey_;
            std::vector<char> pipe = cache_->getData();
            v.pipeline.assign(pipe.begin(), pipe.end());
            wcache_->writeInto(v);
            bool replaced = false;
            for (auto &e: cacheDoc_.variants)
            {
                if (e.sameKey(v))
                {
                    e        = std::move(v);
                    replaced = true;
                    break;
                }
            }
            if (!replaced)
            {
                cacheDoc_.variants.push_back(std::move(v));
            }
            std::vector<uint8_t> out = cacheEncode(cacheDoc_);
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
            VKNN_INFO << "Saved cache (" << out.size() << " bytes, " << cacheDoc_.variants.size() << " variant(s)) -> " << cacheFile_;
        }

        bool useFp16(const Config &cfg) const {
            // Normal is fp16 storage too; markFp32 keeps only the selective set in fp32. High is full fp32.
            return vxVulkanFp16Available() && ctx_->caps().shaderFloat16 && (cfg.precision == Precision::Low || cfg.precision == Precision::Normal);
        }

        std::unique_ptr<Segment> compileSegment(const std::vector<int> &idx, Graph &g, const Config &cfg) override;
        void                     finalize() override {
            saveCaches();
        }

        // One VkPipeline (+ shader module + layout) per distinct kernel configuration, shared by every
        // node that requests it. Driver pipeline memory and creation time then scale with the number of
        // DISTINCT kernels in the model, not the node count (a transformer has hundreds of MatMul nodes
        // but a handful of matmul kernel configs).
        std::shared_ptr<vk::ComputePipeline> sharedPipeline(const std::string &name, uint32_t numBuffers, uint32_t pushConstBytes, const std::vector<uint32_t> &spec, VkPipelineCache cache) {
            std::string key = name;
            key += '|';
            key += std::to_string(numBuffers);
            key += '|';
            key += std::to_string(pushConstBytes);
            for (uint32_t s: spec)
            {
                key += ',';
                key += std::to_string(s);
            }
            auto it = pipePool_.find(key);
            if (it != pipePool_.end())
            {
                return it->second;
            }
            auto p = std::make_shared<vk::ComputePipeline>(*ctx_, name, numBuffers, pushConstBytes, spec, cache);
            pipePool_[key] = p;
            return p;
        }

        // Content-addressed device buffer for small parameter blocks: identical bytes share one
        // allocation (weak-held, so it frees with its last user at segment teardown).
        std::shared_ptr<vk::Buffer> uploadPooled(const void *data, size_t bytes) {
            std::string key(reinterpret_cast<const char *>(data), bytes);
            auto        it = constPool_.find(key);
            if (it != constPool_.end())
            {
                if (auto b = it->second.lock())
                {
                    return b;
                }
            }
            auto b = std::make_shared<vk::Buffer>(*ctx_, bytes, vk::MemPref::kAuto);
            b->upload(data, bytes);
            constPool_[key] = b;
            return b;
        }

        // Device-weight pool: one uploaded copy of a weight/bias/transformed-weight buffer shared by
        // every op instance (and every plan bucket) that references the same weight-cache key at the same
        // precision. `make` runs on a miss (host-cache consult + prepack + upload); a hit returns the
        // shared buffer with no upload. Weakly held (frees with its last user), so a single-bucket model
        // keeps today's allocation count. See vk_weight_pool.h.
        std::shared_ptr<vk::Buffer> acquireWeight(const std::string &key, bool fp16, const std::function<std::shared_ptr<vk::Buffer>()> &make) {
            return weightPool_.acquire(key, fp16, make);
        }

        // ---- host NCHW fp32  <->  device NC4HW4 (fp32 path; fp16 device buffers handled here) ----
        // NC4HW4 groups channels into blocks of four laid out as [N, Cblock, H, W, 4]: the four channels
        // of a block are the innermost contiguous axis, so one (n,cb,h,w) location owns a 4-lane vector at
        // `base = (((n*Cb + cb)*H + h)*W + w) * 4` and lane l holds logical channel c = cb*4 + l. A channel
        // count not divisible by four pads the final block's unused lanes with zero on pack, and those
        // padding lanes are dropped on unpack. The flat path skips all of this: a gpuFlat tensor stores
        // plain NCHW row-major, matching host layout byte-for-byte (fp16 conversion aside).
        static void packToBuffer(vk::Buffer *buf, const RtTensor &rt, bool fp16, bool flat = false) {
            // An int64 boundary input (a shape/index tensor produced by a CPU op, e.g. the Cast-from-int64
            // shape path) has int64 host bytes. The device carries it as compute-precision float, so decode
            // the int64 lanes into a scratch fp32 vector once and pack from that; the magnitudes are small
            // (shape/index values), so the fp32 round-trip is exact. A fp32 host tensor packs directly.
            std::vector<float> i64Scratch;
            const float       *hostSrc = rt.host.f32();
            if (rt.dtype == DType::Int64)
            {
                int64_t        n  = numElements(rt.shape);
                const int64_t *xi = rt.host.i64();
                i64Scratch.resize((size_t) std::max<int64_t>(n, 0));
                for (int64_t i = 0; i < n; ++i)
                {
                    i64Scratch[(size_t) i] = (float) xi[i];
                }
                hostSrc = i64Scratch.data();
            }
            if (flat)
            { // host NCHW row-major == the flat device buffer; straight copy (+ fp16 convert)
                int64_t      n   = numElements(rt.shape);
                const float *src = hostSrc;
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
            const float *src = hostSrc;
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
        // Inverse of packToBuffer: gather each logical channel c back out of NC4HW4 by its block cb = c/4
        // and lane l = c%4, so the source index is `sidx = (((n*Cb + cb)*H + h)*W + w) * 4 + l`. Always
        // produces fp32 host data (rt.dtype set to Float32); readbackOutput does any final dtype convert.
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

        // Download a FLAT (NCHW row-major) graph output straight into the model's declared output dtype,
        // skipping the fp16->fp32->declared double-convert that unpackFromBuffer (always fp32) followed by
        // readbackOutput would do. Only valid for terminal graph outputs: inter-segment boundaries are
        // re-uploaded by packToBuffer, which reads rt.host as fp32, so they keep the fp32 unpack. rt.dtype
        // is set to what rt.host now holds so readbackOutput takes its dst==rt.dtype memcpy fast path.
        static void downloadFlatOutput(vk::Buffer *buf, RtTensor &rt, bool deviceFp16, DType declared) {
            int64_t n = numElements(rt.shape);
            if (deviceFp16 && declared == DType::Float16)
            { // fp16 device -> fp16 output: straight copy, no conversion
                rt.host.resizeElems(n, DType::Float16);
                std::memcpy(rt.host.bytes.data(), buf->host(), (size_t) n * 2);
                rt.dtype = DType::Float16;
            } else if (!deviceFp16 && declared == DType::Float32)
            { // fp32 device -> fp32 output: straight copy
                rt.host.resizeElems(n, DType::Float32);
                std::memcpy(rt.host.bytes.data(), buf->host(), (size_t) n * 4);
                rt.dtype = DType::Float32;
            } else if (declared == DType::Float16)
            { // fp32 device -> fp16 output
                rt.host.resizeElems(n, DType::Float16);
                fp16_t      *d = reinterpret_cast<fp16_t *>(rt.host.bytes.data());
                const float *s = reinterpret_cast<const float *>(buf->host());
                for (int64_t i = 0; i < n; ++i)
                {
                    d[i] = floatToHalf(s[i]);
                }
                rt.dtype = DType::Float16;
            } else
            { // integer / other declared dtype: decode to fp32, readbackOutput does the final convert
                rt.host.resizeElems(n, DType::Float32);
                float *d = rt.host.f32();
                if (deviceFp16)
                {
                    halfToFloatBulk(reinterpret_cast<const fp16_t *>(buf->host()), d, n);
                } else
                {
                    std::memcpy(d, buf->host(), (size_t) n * 4);
                }
                rt.dtype = DType::Float32;
            }
            rt.hostValid = true;
        }

      private:
        std::unique_ptr<vk::VulkanContext> ctx_;
        std::unique_ptr<vk::CommandRunner> runner_;
        std::unique_ptr<vk::PipelineCache> cache_;
        std::unique_ptr<WeightCache>       wcache_;
        std::string                        disabledOps_; // Config::disableVkOps (debug op-fallback list)
        // Multi-variant per-model cache file (cfg.cacheFile). loadCache() reads + validates it into
        // cacheDoc_ and selects the variant matching curKey_; saveCaches() updates that variant and
        // rewrites the file only when the serialized bytes (loadedBytes_) change. cacheLoaded_ makes
        // loadCache idempotent across this model's segments: one VulkanBackend serves exactly one model
        // (one Session), so the single loaded document + curKey_ stay valid for its whole lifetime.
        std::string          cacheFile_;
        CacheDoc             cacheDoc_;
        CacheVariant         curKey_;
        std::vector<uint8_t> loadedBytes_;
        bool                 cacheLoaded_ = false;
        bool                 noCache_     = false;

        std::map<std::string, std::shared_ptr<vk::ComputePipeline>> pipePool_;   // sharedPipeline()
        std::map<std::string, std::weak_ptr<vk::Buffer>>            constPool_;  // uploadPooled()
        DeviceWeightPool<vk::Buffer>                                weightPool_; // acquireWeight() — shared across plan buckets
    };

    std::shared_ptr<vk::ComputePipeline> VkOpEnv::pipeline(const std::string &shaderName, uint32_t numBuffers, uint32_t pushConstBytes, const std::vector<uint32_t> &specData) const {
        return backend->sharedPipeline(shaderName, numBuffers, pushConstBytes, specData, cache ? cache->handle() : VK_NULL_HANDLE);
    }

    std::shared_ptr<vk::Buffer> VkOpEnv::uploadPooled(const void *data, size_t bytes) const {
        return backend->uploadPooled(data, bytes);
    }

    std::shared_ptr<vk::Buffer> VkOpEnv::acquireWeight(const std::string &key, bool fp16, std::function<std::shared_ptr<vk::Buffer>()> make) const {
        return backend->acquireWeight(key, fp16, make);
    }

    // ============================ VulkanSegment ============================
    /// A pre-recorded, statically-planned run of one contiguous GPU node range. The constructor does all
    /// the up-front work: allocate device activation buffers with a greedy liveness pool (internal tensors
    /// share buffers once their last use has passed; boundary/readback tensors get dedicated buffers),
    /// alias pure-copy outputs onto their input's buffer, prepare() every op (which prepacks + uploads
    /// weights), and pre-record the command buffer(s) with precise buffer-level barriers. run() then only
    /// (re)binds zero-copy dma-buf / staging boundaries, packs inputs, submits, and unpacks outputs — the
    /// static dispatch stream is reused across runs and re-recorded only when a boundary buffer changes.
    class VulkanSegment: public Segment {
      public:
        VulkanSegment(const std::vector<int> &idx, Graph &g, const Config &cfg, VulkanBackend *be): be_(be), g_(g), cfg_(cfg) {
            nodeIdx   = idx;
            useFp16_  = be_->useFp16(cfg);
            elemSize_ = useFp16_ ? 2 : 4;
            graphInputs_.insert(g.inputs.begin(), g.inputs.end());

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
            // Geometry-as-metadata: a pure-copy op copies its input verbatim, so alias the output onto the
            // input's buffer and skip the copy dispatch (record() checks src==dst). A Reshape/Squeeze/Unsqueeze
            // qualifies whenever input and output share layout + byte size (the layout pass guarantees this,
            // else it inserts a convert). A full-range unit-step Slice (start=0, step=1 on every axis) is the
            // same thing — an x[:] no-op that survives import; the byte-size guard below proves same shape, so
            // start=0 + step=1 makes it a verbatim copy. Restricted to an internal (poolable) output so
            // readback/boundary tensors keep their own buffer; the root input's liveness is extended to the
            // output's consumers because the liveness scan resolves aliases below.
            std::map<TensorId, TensorId> aliasRoot;
            auto                         resolveAlias = [&](TensorId t) {
                for (int hop = 0; hop < 64 && aliasRoot.count(t); ++hop)
                {
                    t = aliasRoot[t];
                }
                return t;
            };
            auto isIdentitySlice = [&](const Node &nd) {
                if (nd.type != OpType::Slice)
                {
                    return false;
                }
                auto starts = readI64Param(g, nd, "starts", 1);
                if (starts.empty())
                {
                    return false; // params must be visible (static) to prove the slice is a no-op
                }
                for (int64_t s: starts)
                {
                    if (s != 0)
                    {
                        return false;
                    }
                }
                for (int64_t s: readI64Param(g, nd, "steps", 4))
                {
                    if (s != 1)
                    {
                        return false; // a non-unit or negative step reorders elements, not a copy
                    }
                }
                return true;
            };
            for (int ni: idx)
            {
                const Node &nd = g.nodes[ni];
                // A geometry op carrying a fused pointwise epilogue (pw_steps) is NOT a pure copy — its
                // kernel must run to apply the chain, so it can't be aliased/skipped.
                bool pureCopy = !nd.attr.has("pw_steps") && (nd.type == OpType::Reshape || nd.type == OpType::Squeeze || nd.type == OpType::Unsqueeze || isIdentitySlice(nd));
                if (!pureCopy || nd.inputs.empty() || nd.outputs.empty())
                {
                    continue;
                }
                TensorId in = nd.inputs[0], out = nd.outputs[0];
                if (in == kNoTensor || out == kNoTensor || g.isInitializer(in))
                {
                    continue;
                }
                if (!producedHere.count(out) || readBack.count(out) || g.tensors[out].storeFp32 || g.tensors[in].storeFp32)
                {
                    continue; // output must be internal; skip fp32-pinned tensors
                }
                if (actBytes(in) != actBytes(out) || g.tensors[in].gpuFlat != g.tensors[out].gpuFlat)
                {
                    continue; // only a byte-for-byte, same-layout reshape can share the buffer
                }
                aliasRoot[out] = resolveAlias(in);
            }
            // [firstPos,lastPos] of each internal tensor within this segment's execution order
            std::map<TensorId, int> firstPos, lastPos;
            auto                    touch = [&](TensorId t, int k) {
                t = resolveAlias(t); // an aliased tensor lives in its root's buffer; extend the root's span
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
                // Best-fit: reuse the smallest freed slot that still fits, so a large freed buffer is kept
                // available for a later large tensor instead of being spent (and grown) on a small one. A
                // reused slot keeps its existing (larger-or-equal) capacity; only a miss allocates anew.
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
            // Point each aliased pure-copy output at its root's buffer (the root is dedicated- or pool-
            // allocated above); record() then skips the copy since src and dst resolve to the same buffer.
            for (auto &kv: aliasRoot)
            {
                auto it = buffers_.find(resolveAlias(kv.second));
                if (it != buffers_.end())
                {
                    buffers_[kv.first] = it->second;
                }
            }

            // Flat-geometry view-eligibility diagnostic (opt-in: --debug-segments). Classifies each flat
            // Slice/Concat/Transpose as offset-view eligible (its output is one contiguous sub-range of the
            // input), strided (needs a gather), or a Concat disjoint write. Emits greppable [rc-diag] lines
            // that per-node profile ms attributes against. Read-only; no behaviour change.
            if (cfg_.debugSegments)
            {
                auto shp = [](const Shape &s) {
                    std::string o = "[";
                    for (size_t i = 0; i < s.size(); ++i)
                    {
                        o += (i ? "," : "") + std::to_string(s[i]);
                    }
                    return o + "]";
                };
                int slV = 0, slS = 0, co = 0, trI = 0, trS = 0;
                for (int ni: idx)
                {
                    const Node &nd = g.nodes[ni];
                    if (nd.outputs.empty() || nd.outputs[0] == kNoTensor || !g.desc(nd.outputs[0]).gpuFlat)
                    {
                        continue;
                    }
                    if (nd.type == OpType::Slice)
                    {
                        Shape in = g.desc(nd.inputs[0]).shape, out = g.desc(nd.outputs[0]).shape;
                        int   r  = (int) in.size();
                        auto  steps    = readI64Param(g, nd, "steps", 4);
                        bool  unitStep = true;
                        for (auto s: steps)
                        {
                            if (s != 1)
                            {
                                unitStep = false;
                            }
                        }
                        int slicedAx = -1, nSliced = 0;
                        for (int ax = 0; ax < r && ax < (int) out.size(); ++ax)
                        {
                            if (out[ax] != in[ax])
                            {
                                slicedAx = ax;
                                nSliced++;
                            }
                        }
                        bool contig = unitStep && nSliced <= 1 && slicedAx >= 0;
                        for (int k = 0; k < slicedAx; ++k)
                        {
                            if (in[k] != 1)
                            {
                                contig = false; // an outer dim >1 makes the slice several disjoint chunks
                            }
                        }
                        (contig ? slV : slS)++;
                        VKNN_INFO << "[rc-diag] Slice " << (contig ? "VIEW " : "strd ") << nd.name << " in" << shp(in) << "->out" << shp(out) << " ax=" << slicedAx << " step1=" << unitStep << " " << actBytes(nd.outputs[0]) << "B";
                    } else if (nd.type == OpType::Concat)
                    {
                        co++;
                        VKNN_INFO << "[rc-diag] Concat " << nd.name << " axis=" << nd.attr.geti("axis", 1) << " nin=" << nd.inputs.size() << " " << actBytes(nd.outputs[0]) << "B";
                    } else if (nd.type == OpType::Transpose)
                    {
                        const auto &perm  = nd.attr.getints("perm");
                        bool        ident = true;
                        for (size_t k = 0; k < perm.size(); ++k)
                        {
                            if (perm[k] != (int64_t) k)
                            {
                                ident = false; // a real permutation needs strided reads (Stage B)
                            }
                        }
                        (ident ? trI : trS)++;
                        VKNN_INFO << "[rc-diag] Transpose " << (ident ? "VIEW " : "strd ") << nd.name << " " << actBytes(nd.outputs[0]) << "B";
                    }
                }
                VKNN_INFO << "[rc-diag] SUMMARY Slice: " << slV << " view / " << slS << " strided | Concat: " << co << " | Transpose: " << trI << " ident / " << trS << " strided";
            }

            // 2) build env + ops; prepare uploads weights.
            env_.backend  = be_;
            env_.ctx      = &be_->ctx();
            env_.runner   = &be_->runner();
            env_.tuning   = cfg.tuning;
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
            // Load + validate the model cache now that the model hash is known, then hand the primed
            // pipeline + weight caches to the env. loadCache is idempotent across this model's segments.
            be_->loadCache(cfg, env_.modelTag);
            env_.cache   = be_->pipelineCache();
            env_.weights = be_->weightCache();
            env_.devBuf  = [this](TensorId t) -> vk::Buffer * {
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

            VKNN_INFO << "vk memory after segment build: live " << (vk::Buffer::liveBytes() >> 20) << " MB / " << vk::Buffer::liveCount() << " buffers (peak " << (vk::Buffer::peakBytes() >> 20) << " MB / "
                      << vk::Buffer::peakCount() << ", host rss " << hostRssMb() << " MB)";
        }

        // Process resident-set size in MB (0 where /proc is unavailable). The vk buffer totals only
        // cover device allocations; on a UMA device the process RSS is what the OOM killer sees.
        static size_t hostRssMb() {
#ifdef __linux__
            if (FILE *f = fopen("/proc/self/statm", "r"))
            {
                long pages = 0, rss = 0;
                if (fscanf(f, "%ld %ld", &pages, &rss) == 2)
                {
                    fclose(f);
                    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE) >> 20;
                }
                fclose(f);
            }
#endif
            return 0;
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
            // Push-descriptor writes a node records = one per bound storage buffer. A fused
            // pointwise/epilogue kernel always binds the plan SSBO plus the fixed kPwMaxOperands
            // operand slots and kPwMaxOuts extra output streams on top of its own inputs/outputs; a
            // plain op binds just those. Concat dispatches once per concatenated part, re-binding
            // the full set each time. Accumulated per command buffer, this drives the
            // maxSubmitBindings split below.
            auto bindEstimate = [&](const Node &nd) -> int {
                int pwExtra = (nd.type == OpType::FusedPointwise || nd.attr.has("pw_steps")) ? 1 + kPwMaxOperands + kPwMaxOuts : 0;
                if (nd.type == OpType::Concat)
                {
                    return (int) pwCoreInputs(nd) * (2 + pwExtra);
                }
                return (int) nd.inputs.size() + (int) nd.outputs.size() + pwExtra;
            };
            int nodesSinceSplit = 0, bindsSinceSplit = 0;
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
                // Split the segment into multiple command buffers so no single submit (a) runs long
                // enough to trip the GPU watchdog (a ~20s submit on this driver gets reset silently,
                // zeroing the unexecuted tail) or (b) records more push-descriptor writes than the
                // driver holds (maxSubmitBindings; a newer driver corrupts the recording past its cap).
                // The submit fence between chunks is a full barrier, so buffer reuse stays correct across
                // the boundary. Only when not profiling.
                nodesSinceSplit++;
                bindsSinceSplit += bindEstimate(node);
                const int chunkNodes = cfg_.maxSubmitNodes, chunkBinds = cfg_.maxSubmitBindings;
                const bool splitNodes = chunkNodes > 0 && nodesSinceSplit >= chunkNodes;
                const bool splitBinds = chunkBinds > 0 && bindsSinceSplit >= chunkBinds;
                if (!queryPool_ && (splitNodes || splitBinds) && k + 1 < nodeIdx.size())
                {
                    be_->runner().end(cmd_);
                    cmds_.push_back(cmd_);
                    cmd_ = be_->runner().allocate();
                    be_->runner().begin(cmd_);
                    writtenBufs.clear();
                    readBufs.clear();
                    copySinceBarrier = false;
                    nodesSinceSplit  = 0;
                    bindsSinceSplit  = 0;
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
                                std::unique_ptr<vk::Buffer> b = vk::Buffer::importDmaBufFd(be_->ctx(), fd, needB);
                                imp                           = {id, fd, needB, std::shared_ptr<vk::Buffer>(std::move(b))};
                                if (!imp.buf)
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
                // Default-path GPU image conversion: for each 8-bit graph input NOT bound to a dma-buf this
                // run (and not already handled by the dma-buf rebind), stand up a persistent staging buffer
                // and a boundary_convert(staging[declared] -> boundary[device-native]) so the raw caller
                // bytes are converted on the GPU. The staging buffer's stable identity keeps this a one-time
                // re-record. Skipped when a dma-buf fd is present (zero-copy wins) or the graph is not
                // whole-GPU (ioGpuConvert off -> host packToBuffer path).
                if (ioGpuConvert)
                {
                    for (TensorId tid: boundaryInputs)
                    {
                        if (!graphInputs_.count(tid) || buffers_.find(tid) == buffers_.end())
                        {
                            continue;
                        }
                        RtTensor &rt = ctx.t(tid);
                        if (rt.dmaBufFd >= 0 || convert_.count(tid))
                        {
                            continue; // zero-copy dma-buf (direct or its own convert) takes precedence
                        }
                        if (rt.dtype != DType::UInt8 && rt.dtype != DType::Int8)
                        {
                            continue; // only the raw 8-bit image inputs the session stashed as declared dtype
                        }
                        bool         flat    = g_.desc(tid).gpuFlat;
                        TensorFormat devFmt  = flat ? TensorFormat::NCHW : TensorFormat::NC4HW4;
                        DType        devDt   = (useFp16_ && !g_.tensors[tid].storeFp32) ? DType::Float16 : DType::Float32;
                        TensorFormat declFmt = TensorFormat::NCHW; // caller image layout
                        DType        declDt  = rt.dtype;
                        NCHW         x       = NCHW::from(rt.shape.empty() ? g_.tensors[tid].shape : rt.shape);
                        auto        &st      = stagingIn_[tid];
                        size_t       need    = (size_t) (formatElems(declFmt, x) * dtypeSize(declDt));
                        if (!st || st->bytes() != need)
                        {
                            st = std::make_shared<vk::Buffer>(be_->ctx(), need, vk::MemPref::kAuto);
                        }
                        ConvertBinding cb;
                        cb.imported   = st;
                        cb.isInput    = true;
                        cb.shape      = x;
                        cb.declFmt    = declFmt;
                        cb.declDtype  = declDt;
                        cb.devFmt     = devFmt;
                        cb.devDtype   = devDt;
                        convert_[tid] = cb;
                    }
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
                auto sit = stagingIn_.find(tid);
                if (rt.dmaBufFd >= 0)
                {
                    // zero-copy: the GPU reads the caller's dma-buf directly (device-native bytes); no pack.
                    rt.deviceValid  = true;
                    rt.deviceFormat = flat ? TensorFormat::NCHW : TensorFormat::NC4HW4;
                } else if (sit != stagingIn_.end() && convert_.count(tid))
                {
                    // GPU image convert: raw memcpy the caller's declared bytes into the staging buffer; the
                    // recorded boundary_convert dispatch turns them into the device-native boundary. No host
                    // uint8->fp32->fp16 pack. The convert writes bit->second (the boundary), read by the ops.
                    std::memcpy(sit->second->host(), rt.host.bytes.data(), std::min(sit->second->bytes(), rt.host.bytes.size()));
                    rt.deviceValid  = true;
                    rt.deviceFormat = flat ? TensorFormat::NCHW : TensorFormat::NC4HW4;
                } else if (rt.hostValid && !alreadyHere)
                {
                    // The Vulkan device represents an integer tensor as its float value (index/shape ops
                    // upload int64 indices decoded to float), but rt.host for an int64/int32 boundary
                    // tensor holds raw integer bytes. packToBuffer reads host as fp32, so decode the
                    // integer host to fp32 first; a Float32 host packs directly. Without this, an int64
                    // boundary input crossing into a Vulkan segment (e.g. attention_mask when a mid-graph
                    // CPU island splits the graph) is reinterpreted as fp32 and comes out ~0.
                    if (rt.dtype == DType::Int64 || rt.dtype == DType::Int32)
                    {
                        RtTensor f32 = rt;
                        f32.dtype    = DType::Float32;
                        int64_t n    = numElements(rt.shape);
                        f32.host.resizeElems(n, DType::Float32);
                        float *d = f32.host.f32();
                        if (rt.dtype == DType::Int64)
                        {
                            const int64_t *s = rt.host.i64();
                            for (int64_t i = 0; i < n; ++i)
                            {
                                d[i] = (float) s[i];
                            }
                        } else
                        {
                            const int32_t *s = reinterpret_cast<const int32_t *>(rt.host.bytes.data());
                            for (int64_t i = 0; i < n; ++i)
                            {
                                d[i] = (float) s[i];
                            }
                        }
                        VulkanBackend::packToBuffer(bit->second.get(), f32, useFp16_, flat);
                    } else
                    {
                        VulkanBackend::packToBuffer(bit->second.get(), rt, useFp16_, flat);
                    }
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
            std::set<TensorId> graphOut(g_.outputs.begin(), g_.outputs.end());
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
                    bool deviceFp16 = useFp16_ && !g_.tensors[tid].storeFp32;
                    if (flat && graphOut.count(tid))
                    { // terminal graph output: convert straight to the declared dtype (skip fp32 round trip)
                        VulkanBackend::downloadFlatOutput(bit->second.get(), rt, deviceFp16, g_.tensors[tid].dtype);
                    } else
                    {
                        VulkanBackend::unpackFromBuffer(bit->second.get(), rt, deviceFp16, flat);
                    }
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
        // Default-path (non-dma-buf) GPU image I/O: a persistent host-visible staging buffer per 8-bit graph
        // input. The caller's raw bytes are memcpy'd in each run and a recorded boundary_convert turns them
        // into the device-native boundary — no host uint8->fp32->fp16 pack. Allocated once, stable identity.
        std::map<TensorId, std::shared_ptr<vk::Buffer>> stagingIn_;
        std::set<TensorId>                              graphInputs_; // g_.inputs, for the staging-input gate
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
        auto s           = std::make_unique<VulkanSegment>(idx, g, cfg, this);
        s->backend       = this;
        s->compiledGraph = &g;
        return s;
    }

    VKNN_REGISTER_BACKEND(BackendKind::Vulkan, VulkanBackend);

} // namespace vknn
