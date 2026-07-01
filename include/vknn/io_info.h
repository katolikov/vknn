// Describes a model input or output, resolved from the model at load time.
#pragma once
#include "vknn/tensor.h"
#include <cstdint>
#include <string>

namespace vknn {

    /// Describes a model input or output. Everything the caller would otherwise hand-specify, read
    /// straight from the model (concrete shapes resolved at load time, batch fixed to 1).
    struct IOInfo {
        std::string name;
        Shape       shape;
        DType       dtype = DType::Float32;
        int64_t     elems = 0; // product of shape = number of fp32 values expected/produced
        // For zero-copy (IOTensor::dmaBufFd): the size, layout and dtype of the device-native boundary
        // buffer. Declaring this exact (deviceFormat, deviceDtype) on fromDmaBuf/toDmaBuf — or Auto —
        // binds the fd directly; any other declared format is GPU-converted to/from it.
        int64_t      deviceBytes  = 0;
        TensorFormat deviceFormat = TensorFormat::NCHW;
        DType        deviceDtype  = DType::Float32;
    };

} // namespace vknn
