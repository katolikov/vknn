// A named tensor handed in/out of the engine at the public API boundary.
#pragma once
#include "vknn/tensor.h"
#include <cstdint>
#include <string>
#include <vector>

namespace vknn {

    /// A named tensor handed in/out of the engine (host side, NCHW canonical, fp32). For zero-copy I/O,
    /// set dmaBufFd >= 0 instead of filling `data`: the engine imports the fd as the boundary GPU buffer
    /// and reads the input from it / writes the output into it directly (no host buffer). dmaBufFormat /
    /// dmaBufDtype declare the fd's layout + dtype; when they match the device-native boundary (see
    /// IOInfo) the fd is bound directly, otherwise the GPU converts. The engine never allocates the fd.
    struct IOTensor {
        std::string name;
        Shape       shape;
        DType       dtype    = DType::Float32;
        int         dmaBufFd = -1; // >=0 => zero-copy: this fd IS the GPU boundary buffer
        // The layout + dtype of the dma-buf bytes (when dmaBufFd >= 0). Matching the device-native
        // boundary binds the fd directly; otherwise the GPU converts. Auto = bytes are device-native.
        TensorFormat         dmaBufFormat = TensorFormat::NCHW;
        DType                dmaBufDtype  = DType::Float32;
        std::vector<uint8_t> data;
        float               *f32() {
            return reinterpret_cast<float *>(data.data());
        }
        const float *f32() const {
            return reinterpret_cast<const float *>(data.data());
        }
    };

} // namespace vknn
