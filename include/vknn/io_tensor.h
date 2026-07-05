// A named tensor handed in/out of the engine at the public API boundary.
#pragma once
#include "vknn/tensor.h"
#include <cstdint>
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
    };

} // namespace vknn
