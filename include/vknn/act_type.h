// The fused-activation codes applied after an op.
#pragma once
#include <cstdint>

namespace vknn {

    /// Fused activation applied after an op (kept in sync with shaders/common.glsl).
    enum class ActType : int32_t { None = 0, Relu = 1, Relu6 = 2, Clip = 3, HardSwish = 4, SiLU = 5 };

} // namespace vknn
