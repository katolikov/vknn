// Vulkan backend orchestrator: device context, op gate, shared pools, model cache.
#pragma once
#include "core/cache_codec.h"
#include "vk_buffer.h"
#include "vk_command.h"
#include "vk_context.h"
#include "vk_pipeline.h"
#include "vk_weight_cache.h"
#include "vk_weight_pool.h"
#include "vknn/backend.h"
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vknn {

    /// The Vulkan backend orchestrator. Owns the device context + command runner, the op registry gate
    /// (supports/supportsNode decide which nodes run on the GPU vs fall back to the CPU op), the shared
    /// pipeline and content-addressed constant pools, and the multi-variant model cache. One instance
    /// serves exactly one model (one Session): the loaded cache document and selected variant key stay
    /// valid for its whole lifetime. Per-node execution and buffer planning live in VulkanSegment.
    class VulkanBackend: public Backend {
      public:
        // The queue priority is applied at device/queue creation, so it must be known here (before
        // configure() runs) - the backend factory passes the session Config for exactly this.
        explicit VulkanBackend(const Config &cfg = {});
        BackendKind kind() const override {
            return BackendKind::Vulkan;
        }
        const char *name() const override {
            return "Vulkan";
        }
        bool available() const override {
            return ctx_ && ctx_->initialized();
        }
        void configure(const Config &cfg) override;
        bool supports(OpType t, DType dt) const override;

        // Shape-aware gate behind per-node assignment. The shape/attribute logic is vkNodeGate
        // (src/core/vk_gates.cpp) — a pure function shared with the host support report — behind
        // the availability/disable/registry pre-checks, which name their refusal here.
        bool supportsNode(const Graph &g, const Node &nd, DType dt, std::string *whyNot = nullptr) const override;

        vk::VulkanContext &ctx() {
            return *ctx_;
        }
        vk::CommandRunner &runner() {
            return *runner_;
        }
        // The cache-affecting configuration that keys a cache variant (see cache_codec.h). Two configs
        // with an equal key produce identical compiled artifacts and share a variant.
        static CacheVariant variantKey(const Config &cfg);
        // Load + validate the model cache once, selecting the variant for this config. Caching is
        // always-on: a valid file's matching variant primes the pipeline + weight caches for a warm
        // start; a missing/invalid file (or a config with no cached variant yet) starts empty and this
        // variant is built at load and appended on save. Whole-file guards: format + kernel hash
        // (embedded SPIR-V) + device (vendor/device/driver + pipeline-cache UUID) + model hash. An
        // in-memory graph (empty cacheFile) or cfg.noCache stays memory-only.
        void               loadCache(const Config &cfg, const std::string &modelHash);
        vk::PipelineCache *pipelineCache() noexcept {
            return cache_.get();
        }
        WeightCache *weightCache() noexcept {
            return wcache_.get();
        }
        // Update this config's variant in the loaded document and rewrite the cache file, but only when
        // the serialized bytes changed (an unchanged warm session leaves the file untouched). Called from
        // Session::updateCache() at teardown.
        void saveCaches();

        bool useFp16(const Config &cfg) const;

        std::unique_ptr<Segment> compileSegment(const std::vector<int> &idx, Graph &g, const Config &cfg) override;
        void                     finalize() override {
            saveCaches();
        }
        // New cache content is either a prepacked weight / autotune pick (the weight cache's dirty mark)
        // or a driver-compiled pipeline (the blob grew since the last save). Both are checked, because a
        // model whose weights all take the flat upload path leaves the weight cache clean while still
        // paying for every pipeline compile. Neither changed means the file already holds this session's
        // work, so the flush costs one size query and no encode.
        void flushNewCacheWork() override {
            if ((wcache_ && wcache_->dirty()) || (cache_ && cache_->currentBytes() != savedPipelineBytes_))
            {
                saveCaches();
            }
        }

        // One VkPipeline (+ shader module + layout) per distinct kernel configuration, shared by every
        // node that requests it. Driver pipeline memory and creation time then scale with the number of
        // DISTINCT kernels in the model, not the node count (a transformer has hundreds of MatMul nodes
        // but a handful of matmul kernel configs).
        std::shared_ptr<vk::ComputePipeline> sharedPipeline(const std::string &name, uint32_t numBuffers, uint32_t pushConstBytes, const std::vector<uint32_t> &spec, VkPipelineCache cache, uint32_t requiredSubgroupSize = 0);

        // Content-addressed device buffer for small parameter blocks: identical bytes share one
        // allocation (weak-held, so it frees with its last user at segment teardown).
        std::shared_ptr<vk::Buffer> uploadPooled(const void *data, size_t bytes);

        // Device-weight pool: one uploaded copy of a weight/bias/transformed-weight buffer shared by
        // every op instance (and every plan bucket) that references the same weight-cache key at the same
        // precision. `make` runs on a miss (host-cache consult + prepack + upload); a hit returns the
        // shared buffer with no upload. Weakly held (frees with its last user), so a single-bucket model
        // keeps today's allocation count. See vk_weight_pool.h.
        std::shared_ptr<vk::Buffer> acquireWeight(const std::string &key, bool fp16, const std::function<std::shared_ptr<vk::Buffer>()> &make);

        // Fill a device-only buffer with a weight payload through the persistent staging buffer.
        //
        // Weights must not live in host-mapped memory: some UMA drivers cap per-process HOST_VISIBLE
        // allocations far below the device budget (one mobile driver caps near ~4.4 GiB while the heap
        // reports 7.8 GiB free), so a large-weight model whose weights are MemPref::kAuto
        // (DEVICE_LOCAL + HOST_VISIBLE) exhausts that cap and vkAllocateMemory fails with
        // VK_ERROR_OUT_OF_HOST_MEMORY long before the heap is full. Weight bytes are written once at
        // upload and never host-read again, so the destination is kDeviceOnly (allocated from a
        // non-host-visible type where one exists) and is filled by a bounded staged copy: memcpy into
        // the reusable host-visible staging buffer, then one fenced one-shot vkCmdCopyBuffer per
        // chunk of kWeightStagingBufferBytes. Synchronous by design — this is load-time code. A
        // kDeviceOnly buffer is never mapped even on a UMA device whose every memory type is
        // host-visible, so the staged copy is the only way its bytes arrive; the result is
        // byte-identical to a direct host write.
        std::shared_ptr<vk::Buffer> stageWeightToDevice(const void *src, size_t srcBytes, size_t bufferBytes);

        // ---- host NCHW fp32  <->  device NC4HW4 (fp32 path; fp16 device buffers handled here) ----
        // NC4HW4 groups channels into blocks of four laid out as [N, Cblock, H, W, 4]: the four channels
        // of a block are the innermost contiguous axis, so one (n,cb,h,w) location owns a 4-lane vector at
        // `base = (((n*Cb + cb)*H + h)*W + w) * 4` and lane l holds logical channel c = cb*4 + l. A channel
        // count not divisible by four pads the final block's unused lanes with zero on pack, and those
        // padding lanes are dropped on unpack. The flat path skips all of this: a gpuFlat tensor stores
        // plain NCHW row-major, matching host layout byte-for-byte (fp16 conversion aside).
        static void packToBuffer(vk::Buffer *buf, const RtTensor &rt, bool fp16, bool flat = false, int threads = 1);
        // Inverse of packToBuffer: gather each logical channel c back out of NC4HW4 by its block cb = c/4
        // and lane l = c%4, so the source index is `sidx = (((n*Cb + cb)*H + h)*W + w) * 4 + l`. Always
        // produces fp32 host data (rt.dtype set to Float32); readbackOutput does any final dtype convert.
        static void unpackFromBuffer(vk::Buffer *buf, RtTensor &rt, bool fp16, bool flat = false, int threads = 1);

        // Download a FLAT (NCHW row-major) graph output straight into the model's declared output dtype,
        // skipping the fp16->fp32->declared double-convert that unpackFromBuffer (always fp32) followed by
        // readbackOutput would do. Only valid for terminal graph outputs: inter-segment boundaries are
        // re-uploaded by packToBuffer, which reads rt.host as fp32, so they keep the fp32 unpack. rt.dtype
        // is set to what rt.host now holds so readbackOutput takes its dst==rt.dtype memcpy fast path.
        static void downloadFlatOutput(vk::Buffer *buf, RtTensor &rt, bool deviceFp16, DType declared, int threads = 1, int64_t srcElemOffset = 0, int64_t elemCount = -1);

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
        // Pipeline-blob size the cache file already holds, so flushNewCacheWork can tell driver compiles
        // since the last save from a warm session that compiled nothing.
        size_t savedPipelineBytes_ = 0;
        bool   cacheLoaded_        = false;
        bool   noCache_            = false;

        std::map<std::string, std::shared_ptr<vk::ComputePipeline>> pipePool_;   // sharedPipeline()
        std::map<std::string, std::weak_ptr<vk::Buffer>>            constPool_;  // uploadPooled()
        DeviceWeightPool<vk::Buffer>                                weightPool_; // acquireWeight() — shared across plan buckets
        // stageWeightToDevice() staging buffer, bounded by kWeightStagingBufferBytes and kept for the
        // backend's lifetime: constant operands also upload at RECORD time (operandBuf in ops'
        // record()), and segments re-record when a boundary buffer changes, so uploads outlive load.
        std::unique_ptr<vk::Buffer> weightStaging_;
    };

} // namespace vknn
