// Runtime tensor: the live thing during a run (host data, device data, or both, with validity flags).
#pragma once
#include "vknn/common.h"
#include "vknn/device_storage.h"
#include "vknn/dtype.h"
#include "vknn/host_buffer.h"
#include "vknn/op.h"
#include "vknn/tensor_format.h"
#include <cstring>
#include <memory>

namespace vknn {

    /// Runtime tensor: the live tensor during a run. May be resident on host and/or device
    /// simultaneously; the two residencies carry independent validity flags. The device-side
    /// layout and dtype are tracked separately from the host side so a cross-backend handoff can
    /// convert correctly rather than assuming both copies share a format.
    struct RtTensor {
        TensorId id = kNoTensor;         ///< Graph-unique tensor id, or kNoTensor when unbound.
        Shape    shape;                  ///< Logical dimensions in NCHW order.
        DType    dtype = DType::Float32; ///< Element type of the host residency (canonical NCHW).

        // ---- host residency (canonical NCHW, fp32 for compute/IO) ----
        HostBuffer host;              ///< Host-side elements in canonical NCHW layout.
        bool       hostValid = false; ///< True when `host` holds the current values.

        // ---- device residency (managed by a backend) ----
        std::shared_ptr<DeviceStorage> device;                               ///< Backend-owned device storage; null until a backend allocates it.
        TensorFormat                   deviceFormat = TensorFormat::Unknown; ///< Layout of the device copy (the Vulkan backend packs to NC4HW4).
        DType                          deviceDtype  = DType::Float32;        ///< Element type of the device copy.
        bool                           deviceValid  = false;                 ///< True when `device` holds the current values.
        /// Zero-copy boundary: caller dma-buf fd to use directly as this tensor's GPU buffer, or -1 for none.
        int dmaBufFd = -1;
        /// Layout the caller declares the dma-buf holds. Matching the device-native boundary binds the
        /// fd directly; otherwise the GPU converts between the fd and the boundary buffer.
        TensorFormat dmaBufFormat = TensorFormat::NCHW;
        /// Element type the caller declares the dma-buf holds (paired with `dmaBufFormat`).
        DType dmaBufDtype = DType::Float32;

        /// Caller-owned input bytes for the CURRENT run, lent instead of copied. A graph input whose
        /// only consumer is a GPU staging convert does not need a host mirror: Session points these
        /// at the caller's IOTensor::data and the backend copies straight to the staging buffer, one
        /// copy from the caller instead of two. Valid only for the duration of run(), which is why
        /// Session clears them when it returns; any path that needs owned bytes calls
        /// materializeHostBorrow() first.
        const uint8_t *hostBorrow      = nullptr;
        size_t         hostBorrowBytes = 0;

        /// Copy borrowed caller bytes into owned `host` storage and drop the borrow. A no-op when
        /// nothing is borrowed, so a caller can invoke it unconditionally before reading `host`.
        void materializeHostBorrow() {
            if (hostBorrow == nullptr)
            {
                return;
            }
            host.bytes.resize(hostBorrowBytes);
            std::memcpy(host.bytes.data(), hostBorrow, hostBorrowBytes);
            hostBorrow      = nullptr;
            hostBorrowBytes = 0;
        }

        /// @returns The number of logical elements implied by `shape` (0 for an empty shape).
        int64_t elems() const {
            return numElements(shape);
        }
        /// Allocate `host` for `elems()` elements of `dtype` and mark the host residency valid.
        void allocHost() {
            host.resizeElems(elems(), dtype);
            hostValid = true;
        }
    };

} // namespace vknn
