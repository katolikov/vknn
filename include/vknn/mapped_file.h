// Read-only memory mapping of a model file, shared by every initializer that views into it.
#pragma once
#include <cstddef>
#include <memory>
#include <string>

namespace vknn {

    /// A whole file mapped read-only, kept alive by every HostBuffer viewing into it.
    ///
    /// Model weights already exist verbatim in the ".vxm"; copying them into heap memory at load makes
    /// the process hold two copies of every weight and, on a phone, pushes gigabytes of anonymous pages
    /// into swap, which gets a foreground app killed. Mapped file pages are clean and evictable: they
    /// never enter swap, and buckets that reference the same weight blob share the one mapping instead
    /// of each materializing their own copy.
    ///
    /// Mapping is best-effort. `open` returns nullptr when the platform has no mmap or the call fails,
    /// and the loader then falls back to reading bytes into owned storage.
    class MappedFile {
      public:
        /// Map `path` read-only in its entirety. Returns nullptr on any failure (caller must fall back).
        static std::shared_ptr<const MappedFile> open(const std::string &path);
        ~MappedFile();
        MappedFile(const MappedFile &)            = delete;
        MappedFile &operator=(const MappedFile &) = delete;

        const unsigned char *data() const noexcept {
            return base_;
        }
        size_t size() const noexcept {
            return size_;
        }

      private:
        MappedFile(const unsigned char *base, size_t size): base_(base), size_(size) {}
        const unsigned char *base_ = nullptr;
        size_t               size_ = 0;
    };

} // namespace vknn
