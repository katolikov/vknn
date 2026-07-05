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

    /// CPU-mapped view over a caller-owned dma-buf, used for zero-copy model I/O.
    ///
    /// Not copyable: an instance is owned through the unique_ptr returned by wrapFd() and holds a
    /// single mmap (and optionally the fd) that is released in the destructor.
    class IonBuffer {
      public:
        ~IonBuffer();
        IonBuffer(const IonBuffer &)            = delete;
        IonBuffer &operator=(const IonBuffer &) = delete;

        /// Wrap an existing dma-buf fd and mmap it for CPU access.
        /// @param fd            Caller-owned dma-buf file descriptor to map.
        /// @param bytes         Length of the mapping in bytes.
        /// @param takeOwnership When true the IonBuffer closes `fd` on destruction; otherwise the
        ///                      caller retains ownership of the fd (the default).
        /// @returns An owning IonBuffer, or nullptr if the mmap fails.
        static std::unique_ptr<IonBuffer> wrapFd(int fd, size_t bytes, bool takeOwnership = false);

        /// The wrapped dma-buf file descriptor.
        int fd() const noexcept {
            return fd_;
        }
        /// Length of the mapping in bytes.
        size_t size() const noexcept {
            return size_;
        }
        /// CPU-mapped pointer, or nullptr if the mmap failed.
        void *data() const noexcept {
            return map_;
        }

      private:
        IonBuffer()  = default;
        int    fd_   = -1;
        size_t size_ = 0;
        void  *map_  = nullptr;
        bool   owns_ = false; ///< The destructor closes fd_ (fd ownership was transferred in).
    };

} // namespace vknn
