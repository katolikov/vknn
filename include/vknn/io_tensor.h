// A named tensor handed in/out of the engine at the public API boundary.
#pragma once
#include "vknn/tensor.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace vknn {

    /// A named tensor handed in/out of the engine at the public API boundary (host side, NCHW canonical,
    /// fp32). Two mutually exclusive backing modes:
    ///  - Host mode (default): the payload lives in `data`; the engine copies it across the boundary.
    ///  - Zero-copy mode: set `dmaBufFd` >= 0 instead of filling `data`, and the engine imports the fd
    ///    as the boundary GPU buffer, reading an input from it / writing an output into it directly with
    ///    no host buffer. `dmaBufFormat` / `dmaBufDtype` declare the fd's layout + dtype; when they match
    ///    the device-native boundary (see IOInfo) the fd is bound directly, otherwise the GPU converts.
    ///    The engine never allocates the fd.
    struct IOTensor {
        /// Model input/output name this tensor binds to.
        std::string name;
        /// Logical shape, NCHW canonical.
        Shape shape;
        /// Element type of the host payload in `data`.
        DType dtype = DType::Float32;
        /// >= 0 selects zero-copy: this fd IS the GPU boundary buffer. -1 = host mode (use `data`).
        int dmaBufFd = -1;
        /// Layout of the dma-buf bytes (when dmaBufFd >= 0). Matching the device-native boundary binds
        /// the fd directly; otherwise the GPU converts. Auto = bytes are already device-native.
        TensorFormat dmaBufFormat = TensorFormat::NCHW;
        /// Dtype of the dma-buf bytes (when dmaBufFd >= 0). Paired with dmaBufFormat for the conversion.
        DType dmaBufDtype = DType::Float32;
        /// Host payload in host mode; empty in zero-copy mode. Raw bytes; reinterpret via f32().
        std::vector<uint8_t> data;
        /// Mutable typed view of `data` as fp32. Valid only when `dtype` is Float32 and `data` is sized
        /// to whole fp32 elements.
        float *f32() noexcept {
            return reinterpret_cast<float *>(data.data());
        }
        /// Read-only typed view of `data` as fp32. Same validity requirements as the mutable overload.
        const float *f32() const noexcept {
            return reinterpret_cast<const float *>(data.data());
        }
        /// Widen `data` to fp32 host values honoring the declared `dtype`: fp32 is a bit-exact copy;
        /// fp16 and integer dtypes are widened. The element count is derived from the payload length, so
        /// a non-fp32 output is neither over-read (as reinterpreting through f32() would 2x/4x-read it)
        /// nor dropped for a rank-0 scalar. Used at the public output boundary, where callers get fp32.
        std::vector<float> toFloat32() const {
            const size_t       es = dtypeSize(dtype);
            const int64_t      n  = es ? (int64_t) (data.size() / es) : 0;
            std::vector<float> out((size_t) n);
            switch (dtype)
            {
                case DType::Float16:
                    halfToFloatBulk(reinterpret_cast<const fp16_t *>(data.data()), out.data(), n);
                    break;
                case DType::Int64:
                {
                    const int64_t *v = reinterpret_cast<const int64_t *>(data.data());
                    for (int64_t i = 0; i < n; ++i)
                    {
                        out[(size_t) i] = (float) v[i];
                    }
                    break;
                }
                case DType::Int32:
                {
                    const int32_t *v = reinterpret_cast<const int32_t *>(data.data());
                    for (int64_t i = 0; i < n; ++i)
                    {
                        out[(size_t) i] = (float) v[i];
                    }
                    break;
                }
                case DType::UInt8:
                {
                    const uint8_t *v = data.data();
                    for (int64_t i = 0; i < n; ++i)
                    {
                        out[(size_t) i] = (float) v[i];
                    }
                    break;
                }
                case DType::Int8:
                {
                    const int8_t *v = reinterpret_cast<const int8_t *>(data.data());
                    for (int64_t i = 0; i < n; ++i)
                    {
                        out[(size_t) i] = (float) v[i];
                    }
                    break;
                }
                default: // Float32 (and any dtype materialized to fp32 bytes): bounded bit-exact copy.
                    if (n > 0)
                    {
                        std::memcpy(out.data(), data.data(), (size_t) n * sizeof(float));
                    }
                    break;
            }
            return out;
        }
    };

} // namespace vknn
