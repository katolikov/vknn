// Vulkan buffer + dedicated device-memory allocation.
//
// The target SoCs are unified-memory (UMA): a single physical pool is exposed as memory types that
// are simultaneously DEVICE_LOCAL and HOST_VISIBLE. When such a type exists a Buffer is persistently
// mapped at allocation time, so upload()/download()/host() are plain memcpy with no staging buffer or
// queue copy. On a discrete GPU (no host-visible device-local type) the allocator still succeeds by
// falling back to a host-visible type; the mapping then crosses PCIe but the API is unchanged.
#pragma once
#include "vk_context.h"
#include <cstddef>
#include <cstring>
#include <memory>

namespace vknn { namespace vk {

    /// Preferred memory-type traits for an allocation. The allocator treats these as a wish list:
    /// it first tries an exact match, then relaxes, and finally accepts any host-visible type so a
    /// buffer never fails to allocate merely because the ideal type is unavailable.
    enum class MemPref {
        kAuto,       ///< DEVICE_LOCAL + HOST_VISIBLE, avoiding HOST_CACHED (write-combined is the fast upload path on UMA).
        kReadback,   ///< DEVICE_LOCAL + HOST_VISIBLE + HOST_CACHED — efficient CPU reads of GPU-written outputs.
        kDeviceOnly, ///< DEVICE_LOCAL only; no host mapping (host()/upload()/download() are unavailable).
    };

    /// A GPU buffer that owns exactly one VkBuffer and one dedicated VkDeviceMemory (RAII). On a
    /// host-visible allocation the memory is persistently mapped for the object's lifetime, making
    /// host access a direct memcpy.
    ///
    /// Not copyable and not movable: instances are owned through the handle (stack scope or a
    /// shared_ptr) and never relocated. All Vulkan handles are released in the destructor; a
    /// constructor that fails partway releases whatever it already created before rethrowing.
    class Buffer {
      public:
        /// Allocate and (when host-visible) map a buffer.
        /// @param ctx        Owning context; must outlive the buffer.
        /// @param bytes      Logical size in bytes. The driver may pad the underlying allocation;
        ///                   handle()/bytes() report the logical size.
        /// @param pref       Memory-type preference (see MemPref).
        /// @param extraUsage Usage bits OR-ed onto the storage + transfer-src/dst baseline.
        /// @param zeroInit   Zero the mapped host memory after allocation. Set for pooled activation
        ///                   buffers: the liveness planner aliases and reuses them, so a kernel that
        ///                   reads a lane its producer never wrote would otherwise observe stale device
        ///                   memory. Weight buffers are fully overwritten by upload() and skip the memset.
        /// @throws Error if no compatible memory type exists or a Vulkan call fails.
        Buffer(VulkanContext &ctx, size_t bytes, MemPref pref = MemPref::kAuto, VkBufferUsageFlags extraUsage = 0, bool zeroInit = false);
        ~Buffer();
        Buffer(const Buffer &)            = delete;
        Buffer &operator=(const Buffer &) = delete;
        Buffer(Buffer &&)                 = delete;
        Buffer &operator=(Buffer &&)      = delete;

        VkBuffer handle() const noexcept {
            return buf_;
        }
        size_t bytes() const noexcept {
            return bytes_;
        }
        /// True when the buffer is host-visible and therefore mapped.
        bool hostVisible() const noexcept {
            return mapped_ != nullptr;
        }
        /// Persistent host mapping, or nullptr for a device-only buffer.
        void *host() noexcept {
            return mapped_;
        }

        /// memcpy `n` bytes from `src` into the mapping at `offset`. Precondition: the buffer is
        /// host-visible and `offset + n <= bytes()`; both are checked and throw on violation.
        void upload(const void *src, size_t n, size_t offset = 0);
        /// memcpy `n` bytes from the mapping at `offset` into `dst`. Same preconditions as upload().
        void download(void *dst, size_t n, size_t offset = 0);

        /// Import an external dma-buf fd as the backing memory (ION zero-copy), building a Buffer that
        /// aliases the caller's allocation instead of allocating its own. The fd is duplicated for the
        /// driver, so the caller retains ownership of `fd`.
        /// @returns An owning Buffer, or nullptr if the device lacks dma-buf import or the import fails
        ///          (the caller then falls back to a staged copy). Never throws.
        static std::unique_ptr<Buffer> importDmaBufFd(VulkanContext &ctx, int fd, size_t bytes, VkBufferUsageFlags extraUsage = 0) noexcept;

        /// Process-wide allocation accounting. Each Buffer owns one vkAllocateMemory, so liveCount()
        /// also tracks consumption of the driver's maxMemoryAllocationCount budget, not just bytes.
        static size_t liveCount() noexcept;
        static size_t liveBytes() noexcept;
        static size_t peakCount() noexcept;
        static size_t peakBytes() noexcept;

      private:
        /// Bare shell used only by importDmaBufFd, which fills the handles in by hand.
        explicit Buffer(VulkanContext &ctx) noexcept: ctx_(ctx) {
        }

        /// Index of the first memory type in `typeBits` that has every `want` bit and no `avoid` bit;
        /// if none qualifies, `avoid` is dropped and the search repeats. Throws if nothing matches.
        uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid = 0) const;
        bool     isHostVisible(uint32_t typeIdx) const noexcept;
        /// Readable context for an allocation failure, enriched with the heap budget/usage when
        /// VK_EXT_memory_budget is available so an OOM is actionable rather than opaque.
        std::string allocFailureDetail(VkDeviceSize size, uint32_t typeIdx) const;
        /// Add this allocation to the process-wide live/peak totals (called once, last, on success).
        void account() noexcept;
        /// Release every owned handle. noexcept and idempotent so it is safe from both the destructor
        /// and a failing constructor.
        void destroy() noexcept;

        VulkanContext &ctx_;
        VkBuffer       buf_       = VK_NULL_HANDLE;
        VkDeviceMemory mem_       = VK_NULL_HANDLE;
        size_t         bytes_     = 0;
        void          *mapped_    = nullptr;
        bool           imported_  = false; ///< Backing memory came from an imported fd (not owned here).
        bool           accounted_ = false; ///< This allocation has been added to the live/peak totals.
    };

}} // namespace vknn::vk
