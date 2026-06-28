// DMA-BUF wrapper for zero-copy model I/O. See docs/adr/0005.
//
// vknn never allocates these — the caller owns the buffer (a camera/ION/gralloc dma-buf) and passes
// its fd. An IonBuffer wraps that fd with a CPU mmap so vknn can read an input straight from it, or
// write an output straight into it, with no vknn-side host buffer. The same fd can also be imported
// into Vulkan (VK_EXT_external_memory_dma_buf) via vk::Buffer::importDmaBufFd.
#pragma once
#include <cstddef>
#include <memory>

namespace vknn {

    class IonBuffer {
      public:
        ~IonBuffer();
        IonBuffer(const IonBuffer &)            = delete;
        IonBuffer &operator=(const IonBuffer &) = delete;

        // Wrap an existing dma-buf fd (`bytes` long) and mmap it for CPU access. By default does NOT
        // take ownership (the caller keeps/closes the fd). Returns null if the mmap fails.
        static std::unique_ptr<IonBuffer> wrapFd(int fd, size_t bytes, bool takeOwnership = false);

        int fd() const {
            return fd_;
        }
        size_t size() const {
            return size_;
        }
        void *data() const {
            return map_;
        } // CPU-mapped pointer (may be null if mmap failed)

      private:
        IonBuffer()  = default;
        int    fd_   = -1;
        size_t size_ = 0;
        void  *map_  = nullptr;
        bool   owns_ = false;
    };

} // namespace vknn
