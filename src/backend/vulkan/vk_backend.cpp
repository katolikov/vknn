#include "vk_backend.h"
#include "core/boundary_pack.h"   // parallel canonical<->boundary layout/precision conversion
#include "core/fused_attention.h" // kFaHd (FusedAttention device-cap gate)
#include "core/vk_gates.h"        // vkNodeGate/vkKernelDeclared (shared capability model)
#include "vk_op_env.h"
#include "vk_segment.h"
#include "vknn/dtype.h"
#include "vknn/logging.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

namespace vknn {

    namespace {
        // Staging-buffer bound for VulkanBackend::stageWeightToDevice: large enough that a multi-GiB
        // weight set moves in tens of fenced submits, small enough to be a rounding error next to the
        // weights themselves.
        constexpr size_t kWeightStagingBufferBytes = (size_t) 64 << 20;

        // True once the fp16 shader variants (conv_fp16, dwconv_fp16, ...) are compiled in. The ops
        // pick the _fp16 kernels and upload half weights when this and the device feature line up.
        bool vxVulkanFp16Available() {
            return true;
        }
    } // namespace

    // ============================ VulkanBackend ============================
    VulkanBackend::VulkanBackend(const Config &cfg) {
        ctx_ = std::make_unique<vk::VulkanContext>(cfg.priority);
        if (ctx_->initialized())
        {
            runner_ = std::make_unique<vk::CommandRunner>(*ctx_);
        }
    }

    void VulkanBackend::configure(const Config &cfg) {
        disabledOps_ = cfg.disableVkOps;
    }

    bool VulkanBackend::supports(OpType t, DType dt) const {
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
            VKNN_WARN_THROTTLE(std::string("vk_kernel_decl_") + opTypeName(t), 1) << "vkKernelDeclared(" << opTypeName(t) << ") disagrees with the live registry (" << (registered ? "registered" : "missing") << ") - update src/core/vk_gates.cpp";
        }
        return registered;
    }

    bool VulkanBackend::supportsNode(const Graph &g, const Node &nd, DType dt, std::string *whyNot) const {
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
        // FusedAttention's device-dependent requirements: the kernel stages the row's scores in
        // shared memory (kFaSharedBytes) and runs a 256-invocation workgroup. vkNodeGate holds
        // the device-free checks; these need the live caps.
        if (nd.type == OpType::FusedAttention && ctx_)
        {
            const auto &caps = ctx_->caps();
            // Per-node shared usage: sQ[G*hd] + sScores[G*chunk] + sRed[256] + sAcc[hd] fp32,
            // with the op's own chunk = min(4096/G, 256). Refuse only when THIS node's staging
            // exceeds the device budget (the kFaMax* caps bound it, but a device with a small
            // shared budget can still run a small-group model).
            const auto &dims      = nd.attr.getints(kFaDims);
            const auto &ks        = nd.attr.getints(kFaKStride);
            const auto &vs        = nd.attr.getints(kFaVStride);
            int64_t     groupSize = 1;
            for (size_t d = 0; d < dims.size(); ++d)
            {
                if (dims[d] > 1 && d < ks.size() && ks[d] == 0 && d < vs.size() && vs[d] == 0)
                {
                    groupSize = dims[d];
                    break;
                }
            }
            const int64_t hd          = nd.attr.geti(kFaHd);
            const int64_t chunk       = std::min<int64_t>(4096 / std::max<int64_t>(groupSize, 1), 256);
            const int64_t sharedBytes = (groupSize * hd + groupSize * chunk + 256 + hd) * 4;
            if ((int64_t) caps.maxSharedMemory < sharedBytes)
            {
                if (whyNot)
                {
                    *whyNot = "FusedAttention: device shared memory below the kernel's score staging";
                }
                return false;
            }
            if (caps.maxWorkGroupInvocations < 256)
            {
                if (whyNot)
                {
                    *whyNot = "FusedAttention: device workgroup limit below 256 invocations";
                }
                return false;
            }
        }
        return vkNodeGate(g, nd, whyNot);
    }

    CacheVariant VulkanBackend::variantKey(const Config &cfg) {
        CacheVariant k;
        k.precision       = cfg.precision == Precision::High ? "high" : cfg.precision == Precision::Normal ? "normal" : "low";
        k.flatLayout      = cfg.flatLayout();
        k.gpuIslandFold   = cfg.gpuIslandFold();
        k.matmulViewFold  = cfg.matmulViewFold();
        k.ropeFusion      = cfg.ropeFusion();
        k.fusedAttention  = cfg.fusedAttention();
        k.kvConcatFold    = cfg.kvConcatFold();
        k.fp32Tensors     = cfg.fp32Tensors;
        k.winograd        = cfg.hint(Hint::Winograd, (int) Mode::Auto);
        k.winogradVariant = cfg.hint(Hint::WinogradVariant, 0);
        k.winogradUnit    = cfg.hint(Hint::WinogradUnit, 0);
        k.directConv3x3   = cfg.hint(Hint::DirectConv3x3, 0);
        k.splitKConv      = cfg.hint(Hint::SplitKConv, (int) Mode::Auto);
        k.coopmatGemm     = cfg.hint(Hint::CoopmatGemm, (int) Mode::Auto);
        k.kvCacheQuant    = cfg.kvCacheQuantMode();
        return k;
    }

    void VulkanBackend::loadCache(const Config &cfg, const std::string &modelHash) {
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
                if (cacheDecode(bytes.data(), bytes.size(), loaded) && loaded.format == cacheDoc_.format && loaded.kernelHash == cacheDoc_.kernelHash &&
                    loaded.vendorId == cacheDoc_.vendorId && loaded.deviceId == cacheDoc_.deviceId && loaded.driverVersion == cacheDoc_.driverVersion &&
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
        cache_ = std::make_unique<vk::PipelineCache>(*ctx_, pipeInit);
        // Baseline from the driver's own serialization of what was just restored, not from pipeInit: the
        // two differ by however the driver re-encodes, and a warm session must compare equal so it skips
        // the flush entirely.
        savedPipelineBytes_ = cache_->currentBytes();
        wcache_             = std::make_unique<WeightCache>();
        if (matched)
        {
            wcache_->loadFrom(*matched);
        } else
        {
            wcache_->reset(!noCache_ && !cacheFile_.empty());
        }
    }

    void VulkanBackend::saveCaches() {
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
            wcache_->markSaved();
            savedPipelineBytes_ = pipe.size();
            return; // unchanged
        }
        // A cache path may name a directory that does not exist yet (one directory holding every model's
        // cache); create the chain so the first write lands instead of warning on every session.
        for (size_t i = cacheFile_.find('/'); i != std::string::npos; i = cacheFile_.find('/', i + 1))
        {
            if (i > 0)
            {
                ::mkdir(cacheFile_.substr(0, i).c_str(), 0755); // EEXIST is the common case and is ignored
            }
        }
        // Write a per-process temp file and atomically rename it over the target, so a crash or a
        // second concurrent writer mid-write leaves the existing cache intact instead of a truncated
        // (corrupt) file. loadedBytes_ advances only after the write is confirmed and the rename lands.
        const std::string tmp = cacheFile_ + ".tmp." + std::to_string((long) getpid());
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f)
            {
                VKNN_WARN << "cannot write cache file " << tmp;
                return;
            }
            f.write((const char *) out.data(), (std::streamsize) out.size());
            f.flush();
            if (!f)
            {
                VKNN_WARN << "failed writing cache file " << tmp;
                f.close();
                std::remove(tmp.c_str());
                return;
            }
        }
        if (std::rename(tmp.c_str(), cacheFile_.c_str()) != 0)
        {
            VKNN_WARN << "cannot replace cache file " << cacheFile_;
            std::remove(tmp.c_str());
            return;
        }
        loadedBytes_        = out;
        savedPipelineBytes_ = pipe.size();
        wcache_->markSaved();
        VKNN_INFO << "Saved cache (" << out.size() << " bytes, " << cacheDoc_.variants.size() << " variant(s)) -> " << cacheFile_;
    }

    bool VulkanBackend::useFp16(const Config &cfg) const {
        // Normal is fp16 storage too; markFp32 keeps only the selective set in fp32. High is full fp32.
        return vxVulkanFp16Available() && ctx_->caps().shaderFloat16 && (cfg.precision == Precision::Low || cfg.precision == Precision::Normal);
    }

    bool VulkanBackend::useFp16Arith(const Config &cfg) const {
        // Low additionally carries the REDUCTION in fp16, not just storage: half the accumulator
        // registers and packed-fp16 math, at roughly 8 dB of SNR on a 288-term conv reduction. Normal
        // keeps fp32 accumulation, so the bit-exact-with-the-CPU-oracle guarantee lives there.
        return useFp16(cfg) && cfg.precision == Precision::Low;
    }

    std::shared_ptr<vk::ComputePipeline> VulkanBackend::sharedPipeline(const std::string &name, uint32_t numBuffers, uint32_t pushConstBytes, const std::vector<uint32_t> &spec, VkPipelineCache cache, uint32_t requiredSubgroupSize) {
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
        if (requiredSubgroupSize > 0)
        {
            key += "|sg";
            key += std::to_string(requiredSubgroupSize);
        }
        auto it = pipePool_.find(key);
        if (it != pipePool_.end())
        {
            return it->second;
        }
        auto p         = std::make_shared<vk::ComputePipeline>(*ctx_, name, numBuffers, pushConstBytes, spec, cache, requiredSubgroupSize);
        pipePool_[key] = p;
        return p;
    }

    std::shared_ptr<vk::Buffer> VulkanBackend::uploadPooled(const void *data, size_t bytes) {
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

    std::shared_ptr<vk::Buffer> VulkanBackend::acquireWeight(const std::string &key, bool fp16, const std::function<std::shared_ptr<vk::Buffer>()> &make) {
        return weightPool_.acquire(key, fp16, make);
    }

    std::shared_ptr<vk::Buffer> VulkanBackend::stageWeightToDevice(const void *src, size_t srcBytes, size_t bufferBytes) {
        auto           destination = std::make_shared<vk::Buffer>(*ctx_, bufferBytes, vk::MemPref::kDeviceOnly);
        const uint8_t *sourceBytes = static_cast<const uint8_t *>(src);
        for (size_t offset = 0; offset < srcBytes;)
        {
            const size_t chunkBytes = std::min(srcBytes - offset, kWeightStagingBufferBytes);
            if (!weightStaging_ || weightStaging_->bytes() < chunkBytes)
            {
                weightStaging_.reset(); // release before growing so two staging allocations never coexist
                weightStaging_ = std::make_unique<vk::Buffer>(*ctx_, chunkBytes, vk::MemPref::kAuto);
            }
            weightStaging_->upload(sourceBytes + offset, chunkBytes);
            const VkBufferCopy region {0, offset, chunkBytes};
            runner_->oneShot([&](VkCommandBuffer cmd) {
                vkCmdCopyBuffer(cmd, weightStaging_->handle(), destination->handle(), 1, &region);
            });
            offset += chunkBytes;
        }
        return destination;
    }

    void VulkanBackend::packToBuffer(vk::Buffer *buf, const RtTensor &rt, bool fp16, bool flat, int threads) {
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
            int64_t n = numElements(rt.shape);
            if (fp16)
            {
                boundary::packFlatFp16(hostSrc, reinterpret_cast<fp16_t *>(buf->host()), n, threads);
            } else
            {
                std::memcpy(buf->host(), hostSrc, (size_t) n * 4);
            }
            buf->flushAfterWrite(); // writes go through host() above, not upload()
            return;
        }
        boundary::packNc4(hostSrc, buf->host(), NCHW::from(rt.shape), fp16, threads);
        buf->flushAfterWrite(); // writes go through host() above, not upload()
    }

    void VulkanBackend::unpackFromBuffer(vk::Buffer *buf, RtTensor &rt, bool fp16, bool flat, int threads) {
        buf->invalidateForRead(); // reads go through host() below, not download()
        if (flat)
        { // flat device buffer == host NCHW row-major; straight copy (+ fp16 convert)
            int64_t n = numElements(rt.shape);
            rt.host.resizeElems(n, DType::Float32);
            rt.dtype   = DType::Float32;
            float *dst = rt.host.f32();
            if (fp16)
            {
                boundary::unpackFlatFp16(reinterpret_cast<const fp16_t *>(buf->host()), dst, n, threads);
            } else
            {
                std::memcpy(dst, buf->host(), (size_t) n * 4);
            }
            rt.hostValid = true;
            return;
        }
        NCHW x = NCHW::from(rt.shape);
        rt.host.resizeElems(x.elems(), DType::Float32);
        rt.dtype = DType::Float32;
        boundary::unpackNc4(buf->host(), rt.host.f32(), x, fp16, threads);
        rt.hostValid = true;
    }

    void VulkanBackend::downloadFlatOutput(vk::Buffer *buf, RtTensor &rt, bool deviceFp16, DType declared, int threads, int64_t srcElemOffset, int64_t elemCount) {
        // Full output when elemCount < 0; a single flat row (elemCount elements from srcElemOffset)
        // when the caller sliced it (setOutputRow — prefill logits). rt.shape is left unchanged; the
        // Session emits the sliced io.shape and reads exactly `n` elements from rt.host.
        buf->invalidateForRead(); // reads go through host() below, not download()
        int64_t        n   = elemCount >= 0 ? elemCount : numElements(rt.shape);
        const uint8_t *src = reinterpret_cast<const uint8_t *>(buf->host()) + (size_t) srcElemOffset * (deviceFp16 ? 2 : 4);
        if (deviceFp16 && declared == DType::Float16)
        { // fp16 device -> fp16 output: straight copy, no conversion
            rt.host.resizeElems(n, DType::Float16);
            std::memcpy(rt.host.bytes.data(), src, (size_t) n * 2);
            rt.dtype = DType::Float16;
        } else if (!deviceFp16 && declared == DType::Float32)
        { // fp32 device -> fp32 output: straight copy
            rt.host.resizeElems(n, DType::Float32);
            std::memcpy(rt.host.bytes.data(), src, (size_t) n * 4);
            rt.dtype = DType::Float32;
        } else if (declared == DType::Float16)
        { // fp32 device -> fp16 output
            rt.host.resizeElems(n, DType::Float16);
            boundary::packFlatFp16(reinterpret_cast<const float *>(src), reinterpret_cast<fp16_t *>(rt.host.bytes.data()), n, threads);
            rt.dtype = DType::Float16;
        } else
        { // integer / other declared dtype: decode to fp32, readbackOutput does the final convert
            rt.host.resizeElems(n, DType::Float32);
            float *d = rt.host.f32();
            if (deviceFp16)
            {
                boundary::unpackFlatFp16(reinterpret_cast<const fp16_t *>(src), d, n, threads);
            } else
            {
                std::memcpy(d, src, (size_t) n * 4);
            }
            rt.dtype = DType::Float32;
        }
        rt.hostValid = true;
    }

    std::unique_ptr<Segment> VulkanBackend::compileSegment(const std::vector<int> &idx, Graph &g, const Config &cfg) {
        auto s           = std::make_unique<VulkanSegment>(idx, g, cfg, this);
        s->backend       = this;
        s->compiledGraph = &g;
        return s;
    }

    VKNN_REGISTER_BACKEND(BackendKind::Vulkan, VulkanBackend);

} // namespace vknn
