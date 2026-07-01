#pragma once
#include "vknn/dtype.h"
#include <string>
#include <vector>

namespace vknn {

    /// Describes one model input or output (read from the model — you never set these yourself).
    struct TensorInfo {
        std::string          name;
        std::vector<int64_t> shape;
        DType                dtype = DType::Float32; // the tensor's element type (values cross the API as fp32)
        int64_t              count = 0;              // number of elements
        std::string          shapeString() const;    // e.g. "1x3x224x224"
    };

} // namespace vknn
