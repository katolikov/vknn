// Host-side raw byte storage for tensors (initializers, I/O, CPU compute results).
#pragma once
#include "vknn/dtype.h"
#include "vknn/mapped_file.h"
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace vknn {

    /// Raw bytes that are either OWNED (a heap vector) or a read-only VIEW into a mapped model file.
    ///
    /// A view costs no memory: the ".vxm" already holds the weight bytes, so a loaded model points at
    /// them instead of copying. Views are read-only; the first mutation materializes an owned copy
    /// (copy-on-write), which keeps every writer correct without knowing where the bytes came from.
    ///
    /// Blob offsets inside a ".vxm" are not 4- or 8-byte aligned, so a viewed payload must never be
    /// reinterpreted as a typed array — only `data()`/`size()` (memcpy, hashing, byte copies) are valid
    /// on it. HostBuffer's typed accessors materialize first for exactly that reason.
    class ByteStorage {
      public:
        ByteStorage() = default;

        // --- std::vector<uint8_t> subset used across the engine (keeps call sites unchanged) ---
        size_t size() const noexcept {
            return viewSize_ ? viewSize_ : owned_.size();
        }
        bool empty() const noexcept {
            return size() == 0;
        }
        const uint8_t *data() const noexcept {
            return viewSize_ ? viewData_ : owned_.data();
        }
        /// Mutable access materializes: a mapped file is read-only and shared.
        uint8_t *data() {
            materialize();
            return owned_.data();
        }
        void resize(size_t n) {
            materialize();
            owned_.resize(n);
        }
        void assign(size_t n, uint8_t value) {
            dropView();
            owned_.assign(n, value);
        }
        void clear() noexcept {
            dropView();
            owned_.clear();
        }
        void shrink_to_fit() {
            owned_.shrink_to_fit();
        }
        const uint8_t *begin() const noexcept {
            return data();
        }
        const uint8_t *end() const noexcept {
            return data() + size();
        }

        ByteStorage &operator=(std::vector<uint8_t> &&v) {
            dropView();
            owned_ = std::move(v);
            return *this;
        }
        ByteStorage &operator=(const std::vector<uint8_t> &v) {
            dropView();
            owned_ = v;
            return *this;
        }

        /// Point at `count` bytes inside `mapping`; the mapping stays alive while any view references it.
        void setView(std::shared_ptr<const MappedFile> mapping, const uint8_t *at, size_t count) {
            owned_.clear();
            owned_.shrink_to_fit();
            mapping_  = std::move(mapping);
            viewData_ = at;
            viewSize_ = count;
        }
        bool viewed() const noexcept {
            return viewSize_ != 0;
        }
        /// Copy the bytes out (used by the writer, which serializes owned and viewed buffers alike).
        std::vector<uint8_t> toVector() const {
            return viewSize_ ? std::vector<uint8_t>(viewData_, viewData_ + viewSize_) : owned_;
        }
        /// Give up the owned bytes (empties the buffer). A view materializes first.
        std::vector<uint8_t> release() {
            materialize();
            std::vector<uint8_t> out = std::move(owned_);
            owned_.clear();
            return out;
        }

      private:
        void dropView() noexcept {
            mapping_.reset();
            viewData_ = nullptr;
            viewSize_ = 0;
        }
        void materialize() {
            if (!viewSize_)
            {
                return;
            }
            std::vector<uint8_t> copy(viewData_, viewData_ + viewSize_);
            dropView();
            owned_ = std::move(copy);
        }

        std::vector<uint8_t>              owned_;
        std::shared_ptr<const MappedFile> mapping_;  // keeps the view's pages alive
        const uint8_t                    *viewData_ = nullptr;
        size_t                            viewSize_ = 0;
    };

    /// Host-side raw bytes (initializers, I/O, CPU compute results). Logical layout = NCHW.
    ///
    /// The typed accessors reinterpret the byte storage as a contiguous array of the requested element
    /// type; the caller is responsible for matching the accessor to the buffer's actual DType. A
    /// file-backed buffer materializes on typed access, since mapped payloads carry no alignment
    /// guarantee.
    struct HostBuffer {
        /// Raw element bytes, tightly packed in the buffer's element type (no per-element padding).
        ByteStorage bytes;
        /// Resize the storage to hold exactly `n` elements of type `dt`.
        ///
        /// A change in byte size reallocates and zero-fills the whole buffer, so a freshly constructed
        /// buffer -- and any buffer whose size moves -- reads back as all-zero and a caller may fill only
        /// the prefix it has data for. A resize to the byte size already held keeps both the bytes and the
        /// allocation, so a caller that overwrites every element on every run (the graph-input bind, the
        /// graph-output download) pays no redundant memset. A caller that fills only a prefix at the
        /// unchanged size zeroes its own tail.
        ///
        /// @param n  Element count (not bytes); the byte size is `n * dtypeSize(dt)`. A non-positive `n`
        ///           empties the buffer, matching the 0 numElements() reports for a rank-0 shape.
        /// @param dt Element type whose size sets the per-element stride.
        void resizeElems(int64_t n, DType dt) {
            const size_t want = n > 0 ? (size_t) n * dtypeSize(dt) : 0;
            if (bytes.size() != want)
            {
                bytes.assign(want, 0);
            }
        }
        /// View of the storage as fp32 elements. Valid only when the buffer holds Float32 data.
        float *f32() {
            return reinterpret_cast<float *>(bytes.data());
        }
        const float *f32() const {
            return reinterpret_cast<const float *>(alignedBytes());
        }
        /// View of the storage as int64 elements. Valid only when the buffer holds Int64 data.
        int64_t *i64() {
            return reinterpret_cast<int64_t *>(bytes.data());
        }
        const int64_t *i64() const {
            return reinterpret_cast<const int64_t *>(alignedBytes());
        }

      private:
        /// A typed read needs natural alignment, which a mapped blob does not provide; materializing the
        /// view moves the bytes into heap storage (suitably aligned) exactly once.
        const uint8_t *alignedBytes() const {
            if (bytes.viewed())
            {
                const_cast<ByteStorage &>(bytes).data(); // materialize
            }
            return bytes.data();
        }
    };

} // namespace vknn
