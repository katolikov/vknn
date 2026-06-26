#include "vk_buffer.h"
#include <unistd.h>

namespace vknn { namespace vk {

    static const VkBufferUsageFlags kBaseUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    uint32_t Buffer::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
        const auto &mp = ctx_.memProps();
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        {
            if (!(typeBits & (1u << i)))
            {
                continue;
            }
            auto f = mp.memoryTypes[i].propertyFlags;
            if ((f & want) == want && (f & avoid) == 0)
            {
                return i;
            }
        }
        // relax avoid
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
        throw Error(Status::kRuntimeError, "no compatible memory type");
    }

    Buffer::Buffer(VulkanContext &ctx, size_t bytes, MemPref pref, VkBufferUsageFlags extraUsage): ctx_(ctx), bytes_(bytes) {
        VkBufferCreateInfo bi {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size        = bytes;
        bi.usage       = kBaseUsage | extraUsage;
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
                avoid = VK_MEMORY_PROPERTY_HOST_CACHED_BIT; // prefer write-combined for upload
                break;
            case MemPref::kReadback:
                want |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
                break;
            case MemPref::kDeviceOnly:
                break;
        }
        uint32_t typeIdx;
        try
        { typeIdx = findMemoryType(req.memoryTypeBits, want, avoid); } catch (...)
        {
            // fall back to any host-visible (non-UMA case)
            typeIdx = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }

        VkMemoryAllocateInfo ai {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = typeIdx;
        // Dedicated allocation for large/import buffers (helps the driver).
        VkMemoryDedicatedAllocateInfo dedicated {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
        if (ctx_.caps().dedicatedAllocation)
        {
            dedicated.buffer = buf_;
            ai.pNext         = &dedicated;
        }
        VK_CHECK(vkAllocateMemory(ctx_.device(), &ai, nullptr, &mem_));
        VK_CHECK(vkBindBufferMemory(ctx_.device(), buf_, mem_, 0));

        if (ctx_.memProps().memoryTypes[typeIdx].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        {
            VK_CHECK(vkMapMemory(ctx_.device(), mem_, 0, VK_WHOLE_SIZE, 0, &mapped_));
        }
    }

    Buffer::~Buffer() {
        if (mapped_ && !imported_)
        {
            vkUnmapMemory(ctx_.device(), mem_);
        }
        if (buf_)
        {
            vkDestroyBuffer(ctx_.device(), buf_, nullptr);
        }
        if (mem_)
        {
            vkFreeMemory(ctx_.device(), mem_, nullptr);
        }
    }

    void Buffer::upload(const void *src, size_t n, size_t offset) {
        if (!mapped_)
        {
            throw Error(Status::kUnsupported, "upload to non-host-visible buffer");
        }
        std::memcpy((char *) mapped_ + offset, src, n);
    }
    void Buffer::download(void *dst, size_t n, size_t offset) {
        if (!mapped_)
        {
            throw Error(Status::kUnsupported, "download from non-host-visible buffer");
        }
        std::memcpy(dst, (char *) mapped_ + offset, n);
    }

    Buffer *Buffer::importDmaBufFd(VulkanContext &ctx, int fd, size_t bytes, VkBufferUsageFlags extraUsage) {
        if (!ctx.caps().externalMemoryFd || !ctx.caps().externalMemoryDmaBuf)
        {
            return nullptr;
        }
        auto *b      = new Buffer(ctx);
        b->bytes_    = bytes;
        b->imported_ = true;
        try
        {
            VkExternalMemoryBufferCreateInfo ext {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
            ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
            VkBufferCreateInfo bi {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.pNext       = &ext;
            bi.size        = bytes;
            bi.usage       = kBaseUsage | extraUsage;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VK_CHECK(vkCreateBuffer(ctx.device(), &bi, nullptr, &b->buf_));

            // Query allowed memory types for this dma-buf fd.
            VkMemoryFdPropertiesKHR fdProps {VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
            auto                    pfnGetFdProps = (PFN_vkGetMemoryFdPropertiesKHR) vkGetDeviceProcAddr(ctx.device(), "vkGetMemoryFdPropertiesKHR");
            uint32_t                typeBits      = 0xffffffffu;
            if (pfnGetFdProps)
            {
                // Dup the fd: the query and the import each consume a reference; the
                // driver takes ownership of the fd passed to vkAllocateMemory.
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
                typeBits = req.memoryTypeBits; // tolerate driver returning 0 from the query
            }

            // For an imported dma-buf the allowed memory types are dictated by the fd; do not
            // over-constrain. Prefer host-visible (so we can also CPU-map it), else any allowed type.
            uint32_t typeIdx;
            try
            { typeIdx = b->findMemoryType(typeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT); } catch (...)
            {
                const auto &mp   = ctx.memProps();
                int         pick = -1;
                for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
                {
                    if (typeBits & (1u << i))
                    {
                        pick = (int) i;
                        break;
                    }
                }
                if (pick < 0)
                {
                    throw Error(Status::kUnsupported, "no compatible memory type for dma-buf");
                }
                typeIdx = (uint32_t) pick;
            }

            VkImportMemoryFdInfoKHR importInfo {VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
            importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
            importInfo.fd         = ::dup(fd); // driver takes ownership of this dup
            VkMemoryDedicatedAllocateInfo dedicated {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
            dedicated.buffer = b->buf_;
            importInfo.pNext = &dedicated;
            VkMemoryAllocateInfo ai {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.pNext           = &importInfo;
            ai.allocationSize  = req.size;
            ai.memoryTypeIndex = typeIdx;
            VK_CHECK(vkAllocateMemory(ctx.device(), &ai, nullptr, &b->mem_));
            VK_CHECK(vkBindBufferMemory(ctx.device(), b->buf_, b->mem_, 0));
            if (ctx.memProps().memoryTypes[typeIdx].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            {
                vkMapMemory(ctx.device(), b->mem_, 0, VK_WHOLE_SIZE, 0, &b->mapped_);
            }
            return b;
        } catch (const std::exception &e)
        {
            VKNN_WARN << "dma-buf import failed: " << e.what();
            delete b;
            return nullptr;
        }
    }

}} // namespace vknn::vk
