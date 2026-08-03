#include "vk_buffer.h"
#include "core/allocation_budget.h"
#include <atomic>
#include <string>
#include <unistd.h>

namespace vknn { namespace vk {

    namespace {

        // Every buffer is a compute storage buffer and a copy endpoint: Reshape/Flatten/Squeeze lower
        // to vkCmdCopyBuffer, and host<->device staging (vk_image.cpp) copies through these.
        constexpr VkBufferUsageFlags kBaseBufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        // Sentinel meaning "no restriction on memory-type bits", used before a dma-buf fd query
        // narrows the allowed set.
        constexpr uint32_t kAnyMemoryType = 0xffffffffu;

        // Right-shift turning a byte count into mebibytes for readable diagnostics.
        constexpr unsigned kBytesToMiBShift = 20;

        // Process-wide allocation accounting. vkAllocateMemory is one-per-Buffer, so the live count
        // tracks the driver's maxMemoryAllocationCount budget as well as the byte totals.
        std::atomic<size_t> gLiveCount {0};
        std::atomic<size_t> gLiveBytes {0};
        std::atomic<size_t> gPeakCount {0};
        std::atomic<size_t> gPeakBytes {0};

        // Said once per process: a device either has a cached readback type or it does not.
        std::atomic<bool> gReadbackUncachedWarned {false};
        // The warning names a process-wide condition, so it is worth saying once, not per buffer.
        // The threshold itself is allocationCountNearLimit (core/allocation_budget.h), which the
        // host build can test -- no Vulkan source is compiled there.
        std::atomic<bool> gAllocCountWarned {false};

        // Raise an atomic high-water mark to `v` if it currently sits lower.
        void raisePeak(std::atomic<size_t> &peak, size_t v) noexcept {
            size_t p = peak.load();
            while (v > p && !peak.compare_exchange_weak(p, v))
            {
            }
        }

    } // namespace

    size_t Buffer::liveCount() noexcept {
        return gLiveCount.load();
    }
    size_t Buffer::liveBytes() noexcept {
        return gLiveBytes.load();
    }
    size_t Buffer::peakCount() noexcept {
        return gPeakCount.load();
    }
    size_t Buffer::peakBytes() noexcept {
        return gPeakBytes.load();
    }

    size_t Buffer::deviceLocalFreeBytes(VulkanContext &ctx) noexcept {
        if (!ctx.caps().memoryBudget)
        {
            return 0;
        }
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2         props {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
        props.pNext = &budget;
        vkGetPhysicalDeviceMemoryProperties2(ctx.physicalDevice(), &props);
        size_t      freeBytes = 0;
        const auto &mp        = ctx.memProps();
        for (uint32_t heap = 0; heap < mp.memoryHeapCount; ++heap)
        {
            if (!(mp.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT))
            {
                continue;
            }
            // A driver may report usage above its own budget under memory pressure; that heap
            // contributes nothing rather than wrapping the unsigned subtraction.
            if (budget.heapBudget[heap] > budget.heapUsage[heap])
            {
                freeBytes += (size_t) (budget.heapBudget[heap] - budget.heapUsage[heap]);
            }
        }
        return freeBytes;
    }

    bool Buffer::isHostVisible(uint32_t typeIdx) const noexcept {
        return (ctx_.memProps().memoryTypes[typeIdx].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    }

    uint32_t Buffer::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) const {
        const auto &mp = ctx_.memProps();
        // First pass: honor both the required and the avoided property bits.
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        {
            if (!(typeBits & (1u << i)))
            {
                continue;
            }
            VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
            if ((f & want) == want && (f & avoid) == 0)
            {
                return i;
            }
        }
        // Second pass: the avoided bits are only a preference (e.g. write-combined over cached), so
        // drop them rather than fail when no ideal type exists.
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        {
            if (!(typeBits & (1u << i)))
            {
                continue;
            }
            if ((mp.memoryTypes[i].propertyFlags & want) == want)
            {
                return i;
            }
        }
        throw Error(Status::RuntimeError, "no memory type satisfies the required properties");
    }

    std::string Buffer::allocFailureDetail(VkDeviceSize size, uint32_t typeIdx) const {
        std::string detail = "vkAllocateMemory(" + std::to_string(size >> kBytesToMiBShift) + " MiB)";
        // VK_EXT_memory_budget exposes the driver's live view of each heap, turning an opaque OOM into
        // an actionable one (how much the heap allows vs. how much is already resident).
        if (ctx_.caps().memoryBudget)
        {
            VkPhysicalDeviceMemoryBudgetPropertiesEXT budget {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
            VkPhysicalDeviceMemoryProperties2         props {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
            props.pNext = &budget;
            vkGetPhysicalDeviceMemoryProperties2(ctx_.physicalDevice(), &props);
            const uint32_t heap = ctx_.memProps().memoryTypes[typeIdx].heapIndex;
            detail += ", heap " + std::to_string(heap) + " budget " + std::to_string(budget.heapBudget[heap] >> kBytesToMiBShift) + " MiB / in use " +
                      std::to_string(budget.heapUsage[heap] >> kBytesToMiBShift) + " MiB";
        }
        detail += ", vknn live " + std::to_string(gLiveBytes.load() >> kBytesToMiBShift) + " MiB across " + std::to_string(gLiveCount.load()) + " buffers";
        // An allocation can fail because the driver is out of ALLOCATIONS rather than out of memory:
        // every Buffer owns one vkAllocateMemory, and a graph spends them on weights, activation
        // buffers and each distinct fused-pointwise plan. Naming the count against the device's
        // limit separates that case from a genuine byte shortage, which the heap budget above
        // reports and which would otherwise be the only reading available.
        const uint32_t allocLimit = ctx_.caps().maxMemoryAllocationCount;
        if (allocLimit != 0)
        {
            detail += " (peak " + std::to_string(gPeakCount.load()) + " of the device's " + std::to_string(allocLimit) + " allowed allocations)";
        }
        return detail;
    }

    void Buffer::account() noexcept {
        accounted_       = true;
        const size_t cnt = ++gLiveCount;
        const size_t byt = gLiveBytes += bytes_;
        raisePeak(gPeakCount, cnt);
        raisePeak(gPeakBytes, byt);
        // Say it while the run is still healthy: past this fraction of the device's allocation
        // count, the next failure is likelier to be the count than the heap, and the two need
        // different fixes. Reported once per process.
        const uint32_t allocLimit = ctx_.caps().maxMemoryAllocationCount;
        if (allocationCountNearLimit(cnt, allocLimit) && !gAllocCountWarned.exchange(true))
        {
            VKNN_WARN << "vk allocations: " << cnt << " live of the device's " << allocLimit << " allowed (>= " << kAllocCountHighWaterPercent << "%). Every buffer costs one vkAllocateMemory; further allocations may fail on COUNT while the heap still has room.";
        }
    }

    void Buffer::destroy() noexcept {
        // vkFreeMemory implicitly unmaps a mapped allocation, so owned host-visible memory is the only
        // case that needs an explicit unmap; imported dma-buf memory is left mapped and released in one
        // step together with the driver-owned fd at vkFreeMemory. A view's mapping is borrowed from its
        // root (never unmapped here) and its mem_ stays null, so a view releases only its VkBuffer.
        if (mapped_ && !imported_ && !viewParent_)
        {
            vkUnmapMemory(ctx_.device(), mem_);
        }
        mapped_ = nullptr;
        if (buf_ != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(ctx_.device(), buf_, nullptr);
            buf_ = VK_NULL_HANDLE;
        }
        if (mem_ != VK_NULL_HANDLE)
        {
            vkFreeMemory(ctx_.device(), mem_, nullptr);
            mem_ = VK_NULL_HANDLE;
        }
        if (accounted_)
        {
            --gLiveCount;
            gLiveBytes -= bytes_;
            accounted_ = false;
        }
    }

    Buffer::Buffer(VulkanContext &ctx, size_t bytes, MemPref pref, VkBufferUsageFlags extraUsage, bool zeroInit, bool allowSubBufferViews):
        ctx_(ctx), bytes_(bytes), allowsViews_(allowSubBufferViews) {
        // A throwing constructor does not run the destructor, so any partially-created handle is
        // reclaimed here before the exception propagates.
        try
        {
            VkBufferCreateInfo bi {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.size        = bytes;
            bi.usage       = kBaseBufferUsage | extraUsage;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VK_CHECK(vkCreateBuffer(ctx_.device(), &bi, nullptr, &buf_));

            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(ctx_.device(), buf_, &req);

            VkMemoryPropertyFlags want  = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            VkMemoryPropertyFlags avoid = 0;
            switch (pref)
            {
                case MemPref::kAuto:
                    want |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                    avoid = VK_MEMORY_PROPERTY_HOST_CACHED_BIT; // write-combined is the faster upload path on UMA
                    break;
                case MemPref::kReadback:
                    want |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
                    break;
                case MemPref::kDeviceOnly:
                    // Prefer a type WITHOUT host visibility: host-mappable allocations count against a
                    // per-process driver budget that GPU-only weights and scratch must not consume
                    // (see VulkanBackend::stageWeightToDevice). The second findMemoryType pass drops
                    // the avoid bit, so a device whose every type is host-visible still allocates.
                    avoid = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
                    break;
            }

            uint32_t typeIdx;
            try
            { typeIdx = findMemoryType(req.memoryTypeBits, want, avoid); } catch (const Error &)
            {
                // No device-local host-visible type (a discrete GPU rather than UMA): settle for any
                // host-coherent mapping so the buffer is still CPU-reachable, just across the bus.
                typeIdx = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            }
            // A readback buffer that did not get HOST_CACHED is mapped write-combined: the host can
            // write it at full speed but READS from it are uncached and run several times slower,
            // which is the whole cost of unpacking an output. The fallback above can land there
            // silently, so the one place that knows says it -- once, naming the shortfall rather
            // than leaving a slow unpack to be blamed on the kernels.
            if (pref == MemPref::kReadback && (ctx_.memProps().memoryTypes[typeIdx].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) == 0 &&
                !gReadbackUncachedWarned.exchange(true))
            {
                VKNN_WARN << "readback buffers are not HOST_CACHED on this device (memory type " << typeIdx << "): host reads of outputs are uncached, so unpack costs several times what the copy should.";
            }

            VkMemoryAllocateInfo ai {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.allocationSize  = req.size;
            ai.memoryTypeIndex = typeIdx;
            // A dedicated allocation backs the buffer with its own device memory instead of a
            // suballocation, which the driver can place and evict more freely — worthwhile for the
            // large weight/activation buffers this allocator hands out. A view-hosting buffer must
            // skip the hint: dedicated memory may legally bind only the buffer it was dedicated to.
            VkMemoryDedicatedAllocateInfo dedicated {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
            if (ctx_.caps().dedicatedAllocation && !allowSubBufferViews)
            {
                dedicated.buffer = buf_;
                ai.pNext         = &dedicated;
            }

            VkResult ar = vkAllocateMemory(ctx_.device(), &ai, nullptr, &mem_);
            if (ar != VK_SUCCESS)
            {
                throw Error(Status::RuntimeError, allocFailureDetail(req.size, typeIdx) + " -> " + vkResultStr(ar));
            }
            memTypeIndex_ = typeIdx;
            memSize_      = req.size;
            VK_CHECK(vkBindBufferMemory(ctx_.device(), buf_, mem_, 0));

            // A kDeviceOnly buffer is never mapped, even when the chosen type IS host-visible: on a UMA
            // device every memory type carries HOST_VISIBLE, so selecting a type cannot keep weights out
            // of the driver's per-process host-mapping budget — only declining to MAP them can. Its
            // contents arrive by device copy (VulkanBackend::stageWeightToDevice) and host()/upload()/
            // download() stay unavailable, as the MemPref contract states.
            if (isHostVisible(typeIdx) && pref != MemPref::kDeviceOnly)
            {
                void *p = nullptr;
                VK_CHECK(vkMapMemory(ctx_.device(), mem_, 0, VK_WHOLE_SIZE, 0, &p));
                mapped_ = p;
                if (zeroInit)
                {
                    std::memset(mapped_, 0, bytes_);
                }
            } else if (zeroInit)
            { throw Error(Status::InvalidArgument, "zeroInit requires a host-mapped buffer; kDeviceOnly cannot be memset from the host"); }
            account();
        } catch (...)
        {
            destroy();
            throw;
        }
    }

    Buffer::Buffer(VulkanContext &ctx, std::shared_ptr<Buffer> parent, size_t byteOffset, size_t bytes): ctx_(ctx), bytes_(bytes) {
        try
        {
            if (!parent)
            {
                throw Error(Status::InvalidArgument, "sub-buffer view needs a parent buffer");
            }
            // Resolve to the memory-owning root: a view of a view binds into the same root memory at
            // the accumulated offset, and retaining the root's shared_ptr keeps the memory alive for
            // exactly as long as any view exists.
            std::shared_ptr<Buffer> root = std::move(parent);
            size_t                  off  = byteOffset;
            while (root->viewParent_)
            {
                off += root->viewOffset_;
                root = root->viewParent_;
            }
            if (!root->allowsViews_)
            {
                throw Error(Status::InvalidArgument, "sub-buffer view over a buffer not allocated with allowSubBufferViews");
            }
            viewParent_ = root;
            viewOffset_ = off;

            VkBufferCreateInfo bi {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.size        = bytes;
            bi.usage       = kBaseBufferUsage;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VK_CHECK(vkCreateBuffer(ctx_.device(), &bi, nullptr, &buf_));

            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(ctx_.device(), buf_, &req);
            // The driver's own binding requirement, and the only alignment a view offset must clear
            // to exist: 4 bytes on the target GPUs. It says nothing about the width of the loads a
            // kernel will issue through the view — a binding declared as a vec4/f16vec4 array needs
            // the whole vector's byte alignment, which baseAlignedTo() answers and every vectorized
            // kernel pick consults before it may run.
            if (req.alignment != 0 && off % req.alignment != 0)
            {
                throw Error(Status::Unsupported, "sub-buffer view offset " + std::to_string(off) + " violates alignment " + std::to_string(req.alignment));
            }
            if (off + req.size > root->memSize_)
            {
                throw Error(Status::Unsupported, "sub-buffer view [" + std::to_string(off) + ", +" + std::to_string(req.size) + ") exceeds the root allocation of " + std::to_string(root->memSize_) + " B");
            }
            if (!(req.memoryTypeBits & (1u << root->memTypeIndex_)))
            {
                throw Error(Status::Unsupported, "sub-buffer view memory-type requirements exclude the root's memory type");
            }
            VK_CHECK(vkBindBufferMemory(ctx_.device(), buf_, root->mem_, off));
            if (root->mapped_)
            {
                mapped_ = static_cast<char *>(root->mapped_) + off;
            }
            // No account(): a view allocates no memory, so it must not inflate the live/peak byte and
            // vkAllocateMemory-count totals those trackers exist to watch.
        } catch (...)
        {
            destroy();
            throw;
        }
    }

    Buffer::~Buffer() {
        destroy();
    }

    void Buffer::upload(const void *src, size_t n, size_t offset) {
        if (!mapped_)
        {
            throw Error(Status::Unsupported, "upload to non-host-visible buffer");
        }
        if (offset > bytes_ || n > bytes_ - offset)
        {
            throw Error(Status::InvalidArgument, "upload of " + std::to_string(n) + " B at offset " + std::to_string(offset) + " exceeds buffer size " + std::to_string(bytes_));
        }
        std::memcpy(static_cast<char *>(mapped_) + offset, src, n);
    }

    void Buffer::download(void *dst, size_t n, size_t offset) {
        if (!mapped_)
        {
            throw Error(Status::Unsupported, "download from non-host-visible buffer");
        }
        if (offset > bytes_ || n > bytes_ - offset)
        {
            throw Error(Status::InvalidArgument, "download of " + std::to_string(n) + " B at offset " + std::to_string(offset) + " exceeds buffer size " + std::to_string(bytes_));
        }
        std::memcpy(dst, static_cast<const char *>(mapped_) + offset, n);
    }

    std::unique_ptr<Buffer> Buffer::importDmaBufFd(VulkanContext &ctx, int fd, size_t bytes, VkBufferUsageFlags extraUsage) noexcept {
        if (!ctx.caps().externalMemoryFd || !ctx.caps().externalMemoryDmaBuf)
        {
            return nullptr;
        }
        try
        {
            std::unique_ptr<Buffer> b(new Buffer(ctx));
            b->bytes_    = bytes;
            b->imported_ = true;

            VkExternalMemoryBufferCreateInfo ext {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
            ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
            VkBufferCreateInfo bi {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.pNext       = &ext;
            bi.size        = bytes;
            bi.usage       = kBaseBufferUsage | extraUsage;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VK_CHECK(vkCreateBuffer(ctx.device(), &bi, nullptr, &b->buf_));

            // Which memory types accept this dma-buf fd is dictated by the fd, not by the buffer alone.
            VkMemoryFdPropertiesKHR fdProps {VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
            auto     pfnGetFdProps = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(vkGetDeviceProcAddr(ctx.device(), "vkGetMemoryFdPropertiesKHR"));
            uint32_t typeBits      = kAnyMemoryType;
            if (pfnGetFdProps)
            {
                // The query and the import each consume an fd reference, and the driver takes ownership
                // of the fd handed to vkAllocateMemory — so dup for the query and close it afterwards.
                int dupForQuery = ::dup(fd);
                if (pfnGetFdProps(ctx.device(), VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, dupForQuery, &fdProps) == VK_SUCCESS)
                {
                    typeBits = fdProps.memoryTypeBits;
                }
                ::close(dupForQuery);
            }
            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(ctx.device(), b->buf_, &req);
            typeBits &= req.memoryTypeBits;
            if (typeBits == 0)
            {
                typeBits = req.memoryTypeBits; // tolerate a driver that reports 0 from the fd query
            }

            // Prefer a host-visible type (so the imported buffer can also be CPU-mapped); otherwise
            // accept any type the fd allows rather than over-constraining the import.
            uint32_t typeIdx;
            try
            { typeIdx = b->findMemoryType(typeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT); } catch (const Error &)
            {
                const auto &mp   = ctx.memProps();
                int         pick = -1;
                for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
                {
                    if (typeBits & (1u << i))
                    {
                        pick = static_cast<int>(i);
                        break;
                    }
                }
                if (pick < 0)
                {
                    throw Error(Status::Unsupported, "no compatible memory type for dma-buf");
                }
                typeIdx = static_cast<uint32_t>(pick);
            }

            // Dup the caller's fd for the import. vkAllocateMemory takes ownership of this handle only
            // on success (VK_KHR_external_memory_fd); on failure ownership stays with us, so the dup is
            // closed explicitly below. The caller's original fd is never consumed either way.
            const int               importFd = ::dup(fd);
            VkImportMemoryFdInfoKHR importInfo {VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
            importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
            importInfo.fd         = importFd;
            VkMemoryDedicatedAllocateInfo dedicated {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
            dedicated.buffer = b->buf_;
            importInfo.pNext = &dedicated;
            VkMemoryAllocateInfo ai {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.pNext           = &importInfo;
            ai.allocationSize  = req.size;
            ai.memoryTypeIndex = typeIdx;
            VkResult ar        = vkAllocateMemory(ctx.device(), &ai, nullptr, &b->mem_);
            if (ar != VK_SUCCESS)
            {
                ::close(importFd); // reclaim the dup the driver did not take
                throw Error(Status::RuntimeError, std::string("dma-buf vkAllocateMemory -> ") + vkResultStr(ar));
            }
            VK_CHECK(vkBindBufferMemory(ctx.device(), b->buf_, b->mem_, 0));
            if (b->isHostVisible(typeIdx))
            {
                // Mapping an imported buffer is best-effort: failure just means GPU-only access.
                void *p = nullptr;
                if (vkMapMemory(ctx.device(), b->mem_, 0, VK_WHOLE_SIZE, 0, &p) == VK_SUCCESS)
                {
                    b->mapped_ = p;
                }
            }
            b->account();
            return b;
        } catch (const std::exception &e)
        {
            // The unique_ptr's destructor has already reclaimed any partial state.
            VKNN_WARN << "dma-buf import failed: " << e.what();
            return nullptr;
        } catch (...)
        { return nullptr; }
    }

}} // namespace vknn::vk
