// ConvertDtype host-side rule: how many elements the conversion has to walk.
//
// The op changes only the element WIDTH. markFp32 builds the bridge tensor from the producer's
// descriptor and keeps its gpuFlat, so both sides are stored in the same layout and the conversion
// is element-for-element in that layout -- which is why it can be recorded as a flat identity map.
//
// The count is therefore the tensor's STORED footprint, not its logical element count. Those differ
// exactly when the tensor is blocked: NC4HW4 pads the channel axis up to a multiple of kNC4Block, so
// a two-channel tensor occupies twice the elements its shape names. Converting the logical count
// leaves the rest of the buffer holding source-width bytes that every consumer then reads at the
// destination width.
//
// tests/test_convert_dtype_rule.cpp pins this.
#pragma once
#include "vknn/nchw.h"
#include "vknn/tensor_format_enum.h"
#include <cstdint>

namespace vknn {

    /// Elements a ConvertDtype must convert for a tensor of logical shape `shape` stored flat or
    /// blocked.
    /// @param shape   Logical tensor shape (both sides of the convert share it).
    /// @param gpuFlat True when the tensor stores dense row-major, false for NC4HW4.
    /// @returns The stored element count to walk.
    inline int64_t convertDtypeElemCount(const Shape &shape, bool gpuFlat) {
        return formatElems(gpuFlat ? TensorFormat::NCHW : TensorFormat::NC4HW4, NCHW::from(shape));
    }

} // namespace vknn
