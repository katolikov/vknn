// A single ONNX attribute value.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace vknn {

    /// A single attribute value (subset needed for CNNs).
    struct Attr {
        enum Kind { None, Int, Float, Ints, Floats, String } kind = None;
        int64_t              i                                    = 0;
        float                f                                    = 0;
        std::vector<int64_t> ints;
        std::vector<float>   floats;
        std::string          str;
        // For tensor-valued attributes (a Constant node's `value`): the tensor's dims, so the op can
        // emit the right shape instead of a flat 1-D vector (multi-dim constants like anchor grids).
        std::vector<int64_t> shape;
    };

} // namespace vknn
