// The fused-activation codes applied after an op.
#pragma once
#include <cstdint>

namespace vknn {

    /// Fused activation applied to an op's output before it is stored. The integer values are a wire
    /// format: they are serialized into the model file and passed to the compute shaders as an `act`
    /// code, so they must stay in sync with the `ACT_*` codes and `vx_act()` in shaders/common.glsl.
    enum class ActType : int32_t {
        None      = 0, ///< Identity; the output is stored unchanged.
        Relu      = 1, ///< max(x, 0).
        Relu6     = 2, ///< clamp(x, 0, 6).
        Clip      = 3, ///< clamp(x, lo, hi) with the bounds supplied alongside the activation.
        HardSwish = 4, ///< x * clamp(x + 3, 0, 6) / 6.
        SiLU      = 5, ///< x / (1 + exp(-x)); also known as Swish.
    };

} // namespace vknn
