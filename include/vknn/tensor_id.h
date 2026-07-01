// Tensor id typedef and the sentinel for "no tensor".
#pragma once
#include <cstdint>

namespace vknn {

    using TensorId               = int32_t;
    constexpr TensorId kNoTensor = -1;

} // namespace vknn
