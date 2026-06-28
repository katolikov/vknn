// Two views of a tensor: TensorDesc is the static description in the graph, RtTensor is the
// live thing during a run (host data, device data, or both, with validity flags).
#pragma once
#include "vknn/common.h"
#include "vknn/dtype.h"
#include "vknn/op.h"
#include "vknn/tensor_format.h"
#include <cstring>
#include <memory>
#include <vector>

namespace vknn {

    /// Compile-time description of a tensor in the graph.
    struct TensorDesc {
        std::string  name;
        Shape        shape; // logical NCHW shape (may have dynamic dims as -1)
        DType        dtype         = DType::Float32;
        TensorFormat format        = TensorFormat::NCHW;
        bool         isInput       = false;
        bool         isOutput      = false;
        bool         isInitializer = false;
        // Vulkan only: store this tensor as a flat row-major buffer (set by the layout-convert pass for
        // the generic head ops) instead of the default NC4HW4 packing. Ignored by the CPU backend.
        bool gpuFlat = false;
    };

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

    /// Forward declaration: the Vulkan backend attaches its device storage here as an
    /// opaque handle, keeping the core free of Vulkan types.
    struct DeviceStorage;

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
