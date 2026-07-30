// Describes a model input or output, resolved from the model at load time.
#pragma once
#include "vknn/tensor.h"
#include <cstdint>
#include <string>

namespace vknn {

    /// Describes a model input or output. Everything the caller would otherwise hand-specify, read
    /// straight from the model (concrete shapes resolved at load time, batch fixed to 1).
    ///
    /// The first block (name/shape/dtype/elems) is the host-side canonical NCHW description a caller
    /// binds against by default. The second block (device*) describes the device-native boundary
    /// buffer used only for zero-copy I/O.
    struct IOInfo {
        std::string name;                   ///< Tensor name, matching the model graph.
        Shape       shape;                  ///< Concrete NCHW shape resolved at load time (batch fixed to 1).
        DType       dtype = DType::Float32; ///< Element type the model declares for this boundary tensor.
        int64_t     elems = 0;              ///< Element count (product of `shape`); independent of dtype.

        /// @name Device-native boundary (zero-copy)
        /// For zero-copy I/O (IOTensor::dmaBufFd): the size, layout and dtype of the device-native
        /// boundary buffer. Declaring this exact (deviceFormat, deviceDtype) on fromDmaBuf/toDmaBuf —
        /// or Auto — binds the fd directly; any other declared format is GPU-converted to/from it.
        /// @{
        int64_t      deviceBytes  = 0;                  ///< Byte size of the device-native buffer (includes NC4HW4 channel padding).
        TensorFormat deviceFormat = TensorFormat::NCHW; ///< Layout of the device-native buffer (NCHW for flat boundaries, else NC4HW4).
        DType        deviceDtype  = DType::Float32;     ///< Element type of the device-native buffer (fp16 at the compute precision, else fp32).
        /// @}
    };

} // namespace vknn
