#pragma once
#include "vknn/dtype.h"
#include <string>
#include <vector>

namespace vknn {

    /// Describes one model input or output. Populated when the model is loaded and reported through
    /// the query API; callers read these fields and never assign them.
    struct TensorInfo {
        std::string          name;                   ///< Tensor name as declared in the model graph.
        std::vector<int64_t> shape;                  ///< Dimensions, outermost first; may include a symbolic/dynamic axis.
        DType                dtype = DType::Float32; ///< Declared element type. Values still cross the public API as fp32.
        int64_t              count = 0;              ///< Total element count (product of `shape`).
        /// The shape formatted as an `x`-joined string, e.g. "1x3x224x224" ("scalar" when rank 0).
        std::string shapeString() const;
    };

} // namespace vknn
