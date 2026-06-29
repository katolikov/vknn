// Vulkan buffer + memory. Exploits UMA (device-local + host-visible) on the target GPU
// so uploads/downloads are direct memcpy with no staging copy.
#pragma once
#include "vk_context.h"
#include <cstring>

namespace vknn { namespace vk {

    enum class MemPref {
        kAuto,       // device-local, host-visible if available (UMA fast path)
        kReadback,   // host-cached preferred (efficient CPU reads of outputs)
        kDeviceOnly, // device-local only (no host access)
    };

    /// A GPU buffer backed by a dedicated allocation. On UMA the memory is persistently
    /// mapped, so host()/upload()/download() are plain memcpy.
    class Buffer {
      public:
        // zeroInit clears host-visible memory on allocation. Set it for activation buffers (the liveness
        // planner aliases/reuses them, and a kernel that reads a lane its producer never wrote would
        // otherwise pick up uninitialized device memory); weight buffers are fully overwritten by upload
        // so they leave it off to skip the memset.
        Buffer(VulkanContext &ctx, size_t bytes, MemPref pref = MemPref::kAuto, VkBufferUsageFlags extraUsage = 0, bool zeroInit = false);
        ~Buffer();
        Buffer(const Buffer &)            = delete;
        Buffer &operator=(const Buffer &) = delete;

        VkBuffer handle() const {
            return buf_;
        }
        size_t bytes() const {
            return bytes_;
        }
        bool hostVisible() const {
            return mapped_ != nullptr;
        }
        void *host() {
            return mapped_;
        }

        void upload(const void *src, size_t n, size_t offset = 0);
        void download(void *dst, size_t n, size_t offset = 0);

        // Import an external dma-buf fd as the backing memory (ION zero-copy). Returns false
        // if the import path is unsupported; caller falls back to a staged copy.
        static Buffer *importDmaBufFd(VulkanContext &ctx, int fd, size_t bytes, VkBufferUsageFlags extraUsage = 0);

      private:
        Buffer(VulkanContext &ctx): ctx_(ctx) {
        }
        uint32_t       findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid = 0);
        VulkanContext &ctx_;
        VkBuffer       buf_      = VK_NULL_HANDLE;
        VkDeviceMemory mem_      = VK_NULL_HANDLE;
        size_t         bytes_    = 0;
        void          *mapped_   = nullptr;
        bool           imported_ = false;
    };

}} // namespace vknn::vk
