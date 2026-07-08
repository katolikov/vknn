// Host-side raw byte storage for tensors (initializers, I/O, CPU compute results).
#pragma once
#include "vknn/dtype.h"
#include <cstdint>
#include <vector>

namespace vknn {

    /// Host-side raw bytes (initializers, I/O, CPU compute results). Logical layout = NCHW.
    ///
    /// The typed accessors reinterpret the byte storage as a contiguous array of the requested element
    /// type; the caller is responsible for matching the accessor to the buffer's actual DType.
    struct HostBuffer {
        /// Raw element bytes, tightly packed in the buffer's element type (no per-element padding).
        std::vector<uint8_t> bytes;
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
            return reinterpret_cast<const float *>(bytes.data());
        }
        /// View of the storage as int64 elements. Valid only when the buffer holds Int64 data.
        int64_t *i64() {
            return reinterpret_cast<int64_t *>(bytes.data());
        }
        const int64_t *i64() const {
            return reinterpret_cast<const int64_t *>(bytes.data());
        }
    };

} // namespace vknn
