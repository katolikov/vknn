// Runtime tensor: the live thing during a run (host data, device data, or both, with validity flags).
#pragma once
#include "vknn/common.h"
#include "vknn/device_storage.h"
#include "vknn/dtype.h"
#include "vknn/host_buffer.h"
#include "vknn/op.h"
#include "vknn/tensor_format.h"
#include <memory>

namespace vknn {

    /// Runtime tensor: may be resident on host and/or device. Tracks validity + the
    /// device layout/dtype so cross-backend handoff can convert correctly.
    struct RtTensor {
        TensorId id = kNoTensor;
        Shape    shape;
        DType    dtype = DType::Float32;

        // ---- host residency (canonical NCHW, fp32 for compute/IO) ----
        HostBuffer host;
        bool       hostValid = false;

        // ---- device residency (managed by a backend) ----
        std::shared_ptr<DeviceStorage> device; // null until a backend allocates it
        TensorFormat                   deviceFormat = TensorFormat::Unknown;
        DType                          deviceDtype  = DType::Float32;
        bool                           deviceValid  = false;
        // Zero-copy boundary: caller dma-buf fd to use directly as this tensor's GPU buffer (-1 = none).
        int dmaBufFd = -1;
        // The layout + dtype the caller declares the dma-buf holds. Matching the device-native boundary
        // binds the fd directly; otherwise the GPU converts between the fd and the boundary buffer.
        TensorFormat dmaBufFormat = TensorFormat::NCHW;
        DType        dmaBufDtype  = DType::Float32;

        int64_t elems() const {
            return numElements(shape);
        }
        void allocHost() {
            host.resizeElems(elems(), dtype);
            hostValid = true;
        }
    };

} // namespace vknn
