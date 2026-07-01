// Host-side raw byte storage for tensors (initializers, I/O, CPU compute results).
#pragma once
#include "vknn/dtype.h"
#include <cstdint>
#include <vector>

namespace vknn {

    /// Host-side raw bytes (initializers, I/O, CPU compute results). Logical layout = NCHW.
    struct HostBuffer {
        std::vector<uint8_t> bytes;
        void                 resizeElems(int64_t n, DType dt) {
            bytes.assign((size_t) n * dtypeSize(dt), 0);
        }
        float *f32() {
            return reinterpret_cast<float *>(bytes.data());
        }
        const float *f32() const {
            return reinterpret_cast<const float *>(bytes.data());
        }
        int64_t *i64() {
            return reinterpret_cast<int64_t *>(bytes.data());
        }
        const int64_t *i64() const {
            return reinterpret_cast<const int64_t *>(bytes.data());
        }
    };

} // namespace vknn
