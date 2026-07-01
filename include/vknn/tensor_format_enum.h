// Tensor layouts. The IR is always NCHW; the Vulkan backend packs to NC4HW4 internally.
#pragma once
#include <cstdint>

namespace vknn {

    enum class TensorFormat : uint8_t {
        NCHW    = 0, // canonical (ONNX/Caffe). N, C, H, W.
        NHWC    = 1, // channel-last (common I/O layout).
        NC4HW4  = 2, // internal Vulkan packed layout: channels in vec4 blocks.
        Auto    = 3, // declared-boundary sentinel: bytes are already device-native, bind the fd directly.
        Unknown = 255,
    };

    inline const char *formatStr(TensorFormat f) {
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
