// Tensor layouts. The IR is always NCHW; the Vulkan backend packs to NC4HW4 internally.
#pragma once
#include <cstdint>

namespace vknn {

    /// Memory layout of a tensor at an I/O boundary or on the device. The IR always reasons in NCHW;
    /// these values name the physical layout a buffer's bytes are actually stored in.
    enum class TensorFormat : uint8_t {
        NCHW    = 0,   ///< Canonical dense layout (ONNX/Caffe): N, C, H, W in row-major order.
        NHWC    = 1,   ///< Channel-last dense layout, a common external I/O layout.
        NC4HW4  = 2,   ///< Internal Vulkan packed layout: channels grouped into vec4 blocks.
        Auto    = 3,   ///< Declared-boundary sentinel: bytes are already device-native, so bind the fd directly.
        Unknown = 255, ///< Layout not yet determined.
    };

    /// Short display name for a layout, for logs and diagnostics.
    /// @param f Layout to name.
    /// @returns A static string literal ("NCHW", "NHWC", "NC4HW4", "Auto"), or "?" for Unknown or any
    ///          unrecognized value. Never null and never owned by the caller.
    inline const char *formatStr(TensorFormat f) noexcept {
        switch (f)
        {
            case TensorFormat::NCHW:
                return "NCHW";
            case TensorFormat::NHWC:
                return "NHWC";
            case TensorFormat::NC4HW4:
                return "NC4HW4";
            case TensorFormat::Auto:
                return "Auto";
            default:
                return "?";
        }
    }

} // namespace vknn
