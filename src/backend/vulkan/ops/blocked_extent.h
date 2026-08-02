// How many elements an op that walks a buffer ELEMENT BY ELEMENT has to cover.
//
// A per-element map -- a dtype conversion, a clamp, a comparison -- reads element i and writes
// element i, so it is indifferent to how the elements are arranged: it computes the same answer in
// any layout as long as its input and output share one. What it is NOT indifferent to is how many
// elements the buffer holds. NC4HW4 pads the channel axis up to a multiple of kNC4Block, so a
// two-channel tensor occupies twice the elements its shape names, and walking the logical count
// covers a prefix and leaves the rest of the destination undefined.
//
// tests/test_blocked_extent.cpp pins this.
#pragma once
#include "vknn/nchw.h"
#include "vknn/tensor_format_enum.h"
#include <cstdint>

namespace vknn {

    /// Elements physically stored for a tensor of logical shape `shape` in the layout it carries.
    /// @param shape   Logical tensor shape.
    /// @param gpuFlat True when the tensor stores dense row-major, false for NC4HW4.
    /// @returns The stored element count an element-by-element walk must cover.
    inline int64_t storedElemCount(const Shape &shape, bool gpuFlat) {
        return formatElems(gpuFlat ? TensorFormat::NCHW : TensorFormat::NC4HW4, NCHW::from(shape));
    }

} // namespace vknn
