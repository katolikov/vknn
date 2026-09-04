// Page-aligned host memory the engine binds to the GPU directly. On a device whose driver imports
// host memory (VK_EXT_external_memory_host; the unified-memory phones the engine targets) a run
// reads an input from the block and writes an output into it with no copy on either side: the
// boundary convert between the caller's fp32 NCHW bytes and the device-native blocked fp16 runs on
// the GPU against the block itself. Without the extension, or when a block cannot be imported, the
// block behaves like ordinary host memory (one copy per direction, as an IOTensor's `data`).
//
// Allocate through alloc(); hand the block to the engine inside an IOTensor (`pinned`) or a Tensor
// (Tensor::pinned / Tensor::toPinned). A block may be reused across runs and sessions, and is the
// way to reuse it: the GPU binding is created on the first run that sees the block and kept while
// the block lives (creating it costs a few ms of page pinning). The block is freed with its last
// shared_ptr; the engine holds only a weak reference and drops its binding once the block is gone.
// An output block holds the run's result until the next run that writes it.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

namespace vknn {
    /// The block alignment and size granule: a page, the host-pointer import granularity of every
    /// driver the engine has run on (a driver asking for more falls back to the copy path).
    constexpr size_t kPinnedHostAlignment = 4096;

    class PinnedHostMemory {
      public:
        /// A zeroed block of at least `bytes` bytes (a zero request allocates one granule). Returns
        /// null when the allocation fails.
        static std::shared_ptr<PinnedHostMemory> alloc(size_t bytes);
        ~PinnedHostMemory();
        PinnedHostMemory(const PinnedHostMemory &)            = delete;
        PinnedHostMemory &operator=(const PinnedHostMemory &) = delete;

        uint8_t *data() noexcept {
            return static_cast<uint8_t *>(ptr_);
        }
        const uint8_t *data() const noexcept {
            return static_cast<const uint8_t *>(ptr_);
        }
        /// The payload size the caller asked for.
        size_t bytes() const noexcept {
            return bytes_;
        }
        /// The allocated size, `bytes()` rounded up to kPinnedHostAlignment: the range the engine imports.
        size_t capacity() const noexcept {
            return capacity_;
        }

      private:
        PinnedHostMemory(void *ptr, size_t bytes, size_t capacity) noexcept: ptr_(ptr), bytes_(bytes), capacity_(capacity) {
        }
        void  *ptr_;
        size_t bytes_;
        size_t capacity_;
    };
} // namespace vknn
