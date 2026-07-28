// A single ONNX attribute value.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace vknn {

    /// A single ONNX node attribute value (the subset of AttributeProto types CNN import needs).
    ///
    /// This is a tagged union by convention rather than a real union: `kind` selects which one of the
    /// scalar/vector members below carries the value, and the other members are left default. Read the
    /// member that matches `kind`; the rest are meaningless for that `kind`.
    struct Attr {
        /// Discriminant identifying which value member is populated.
        enum Kind {
            None,   ///< No value present; every value member is default.
            Int,    ///< Scalar int64, in `i`.
            Float,  ///< Scalar float32, in `f`.
            Ints,   ///< int64 list, in `ints`.
            Floats, ///< float32 list, in `floats` (also the form tensor-valued constants take; see `shape`).
            String  ///< UTF-8/bytes string, in `str`.
        } kind                 = None;
        int64_t              i = 0;  ///< Value for kind == Int.
        float                f = 0;  ///< Value for kind == Float.
        std::vector<int64_t> ints;   ///< Value for kind == Ints.
        std::vector<float>   floats; ///< Value for kind == Floats.
        std::string          str;    ///< Value for kind == String.
        /// Original tensor dims for a tensor-valued attribute (a Constant node's `value`), stored
        /// alongside the flattened `floats`. Lets the op emit the constant's true shape instead of a
        /// flat 1-D vector (multi-dim constants such as anchor grids). Empty for non-tensor attributes.
        std::vector<int64_t> shape;
    };

} // namespace vknn
