// uploadInit host-side rule: how many elements a flat initializer upload binds, and how many payload
// bytes that count needs.
//
// A graph initializer keeps its payload at the stored dtype's native width: 2 bytes per Float16 lane,
// 1 per Int8/UInt8 lane (a native quant or byte tensor is never widened in the graph), 8 per Int64
// lane, 4 for Float32 and for the Int32 label (INT32 materializes to fp32 lanes). The element count
// normally comes from the shape; a rank-0 scalar (numElements() 0) recovers it from the payload size
// divided by that native width, so a scalar Int64, Int8 or UInt8 constant binds its one value instead
// of a zero-length buffer or a spurious short-payload error.
//
// tests/test_binary_int_and_cast_fixes.cpp pins this.
#pragma once
#include "vknn/dtype.h"
#include "vknn/shape.h"
#include <cstddef>
#include <cstdint>

namespace vknn {

    /// Elements a flat upload of an initializer binds.
    /// @param shape        Shape the consumer binds the initializer at.
    /// @param storedDtype  The initializer's descriptor dtype, which fixes its payload lane width.
    /// @param payloadBytes Size of the stored payload in bytes.
    /// @returns numElements(shape), or for a rank-0 / zero-count shape the whole lanes the payload
    ///          holds at the stored width (0 for an unrecognized dtype).
    inline int64_t uploadInitElemCount(const Shape &shape, DType storedDtype, size_t payloadBytes) {
        const int64_t shapeCount = numElements(shape);
        if (shapeCount > 0)
        {
            return shapeCount;
        }
        const size_t laneBytes = dtypeSize(storedDtype);
        return laneBytes > 0 ? (int64_t) (payloadBytes / laneBytes) : 0;
    }

    /// Payload bytes `elemCount` elements of `storedDtype` occupy. A payload shorter than this is not
    /// element-typed by its descriptor (a packed quantized weight keeps its logical Float16 shape over
    /// a smaller payload) and must not take the flat upload path.
    inline size_t uploadInitPayloadBytesNeeded(int64_t elemCount, DType storedDtype) {
        return elemCount > 0 ? (size_t) elemCount * dtypeSize(storedDtype) : 0;
    }

} // namespace vknn
