// ONNX ArgMax / ArgMin (opset 13) on the CPU backend: the selection rule, the per-slice scan and the
// runner the two op files share. The CPU kernel is the byte oracle shaders/arg_extreme.comp is
// diffed against, so both implement the one sequential rule below.
//
// Selection rule per axis slice x[0..extent-1] (the ONNX Runtime scan):
//     best = x[0]; bestAt = 0;
//     for j in 1..extent-1:
//         ArgMax: take = selectLast ? (x[j] >= best) : (x[j] > best)
//         ArgMin: take = selectLast ? (x[j] <= best) : (x[j] < best)
//         if take: best = x[j]; bestAt = j
// Consequences: every comparison with a NaN is false, so a NaN at index 0 stays selected (an all-NaN
// slice yields 0) and a NaN anywhere else is never taken; equal values keep the first index, or the
// last one with select_last_index; -0.0 and +0.0 compare equal and so tie.
#pragma once
#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/parallel.h"
#include "vknn/error.h"
#include "vknn/op.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace vknn { namespace cpu {

    /// ONNX attribute defaults for ArgMax / ArgMin.
    inline constexpr int64_t kArgExtremeDefaultAxis            = 0;
    inline constexpr int64_t kArgExtremeDefaultKeepDims        = 1;
    inline constexpr int64_t kArgExtremeDefaultSelectLastIndex = 0;

    /// True when `candidate` replaces the running `best` under the selection rule above.
    template <class T> inline bool argExtremeTakes(T candidate, T best, bool selectLargest, bool selectLast) noexcept {
        if (selectLargest)
        {
            return selectLast ? candidate >= best : candidate > best;
        }
        return selectLast ? candidate <= best : candidate < best;
    }

    /// Scan the output elements [first, last) of one outer block. `block` points at the block's first
    /// data element (the block holds `extent` planes of `inner` elements); `bestAt` points at the
    /// block's output element `first`; `bestValue` is caller scratch of at least `last - first`
    /// elements. The planes are visited in axis order and each plane is read front to back, so every
    /// output element sees exactly the comparison sequence of its own serial scan while the data is
    /// read sequentially.
    template <class T>
    inline void argExtremeScanBlock(const T *block, int64_t extent, int64_t inner, int64_t first, int64_t last, bool selectLargest, bool selectLast, T *bestValue, int64_t *bestAt) {
        const int64_t count = last - first;
        const T      *head  = block + first;
        for (int64_t k = 0; k < count; ++k)
        {
            bestValue[k] = head[k];
            bestAt[k]    = 0;
        }
        for (int64_t j = 1; j < extent; ++j)
        {
            const T *plane = block + j * inner + first;
            for (int64_t k = 0; k < count; ++k)
            {
                if (argExtremeTakes(plane[k], bestValue[k], selectLargest, selectLast))
                {
                    bestValue[k] = plane[k];
                    bestAt[k]    = j;
                }
            }
        }
    }

    /// Scan the flat output range [lo, hi) of an [outer, extent, inner] view, splitting it at outer
    /// block boundaries. Every output element is an independent scan, so any partition of the output
    /// range computes the same bytes.
    template <class T>
    inline void argExtremeScanRange(const T *data, int64_t extent, int64_t inner, int64_t lo, int64_t hi, bool selectLargest, bool selectLast, int64_t *indices) {
        std::vector<T> bestValue((size_t) std::min(hi - lo, inner));
        for (int64_t outputIndex = lo; outputIndex < hi;)
        {
            const int64_t blockIndex = outputIndex / inner;
            const int64_t blockStart = blockIndex * inner; // output index of the block's first element
            const int64_t segmentEnd = std::min(hi, blockStart + inner);
            argExtremeScanBlock(data + blockIndex * extent * inner, extent, inner, outputIndex - blockStart, segmentEnd - blockStart, selectLargest, selectLast,
                                bestValue.data(), indices + outputIndex);
            outputIndex = segmentEnd;
        }
    }

    /// Run ArgMax (`selectLargest`) or ArgMin over node.inputs[0] into an int64 index tensor. int64 data
    /// is compared exactly in int64; every other runtime dtype reads through f32(). A rank-0 input, an
    /// axis outside [-rank, rank-1] and an axis of extent 0 throw InvalidArgument naming the node.
    inline void runArgExtreme(const Node &node, ExecContext &ctx, bool selectLargest) {
        const std::string where = std::string(opTypeName(node.type)) + " '" + node.name + "': ";
        if (node.inputs.empty() || node.inputs[0] == kNoTensor || node.outputs.empty() || node.outputs[0] == kNoTensor)
        {
            throw Error(Status::InvalidArgument, where + "needs one data input and one indices output");
        }
        const RtTensor &X             = ctx.t(node.inputs[0]);
        const int64_t   rank          = (int64_t) X.shape.size();
        const int64_t   attributeAxis = node.attr.geti("axis", kArgExtremeDefaultAxis);
        if (rank == 0)
        {
            throw Error(Status::InvalidArgument, where + "rank-0 input has no axis to select along");
        }
        if (attributeAxis < -rank || attributeAxis >= rank)
        {
            throw Error(Status::InvalidArgument, where + "axis " + std::to_string(attributeAxis) + " is out of range for a rank-" + std::to_string(rank) + " input");
        }
        const int64_t axis   = attributeAxis < 0 ? attributeAxis + rank : attributeAxis;
        const int64_t extent = X.shape[(size_t) axis];
        if (extent == 0)
        {
            throw Error(Status::InvalidArgument, where + "axis " + std::to_string(attributeAxis) + " has extent 0, so there is no element to select");
        }
        const bool keepDims   = node.attr.geti("keepdims", kArgExtremeDefaultKeepDims) != 0;
        const bool selectLast = node.attr.geti("select_last_index", kArgExtremeDefaultSelectLastIndex) != 0;

        // Output: the input shape with the axis set to 1 (keepdims) or removed; an empty result becomes
        // {1} (the IR has no rank-0 activations, the Reduce/Det convention).
        int64_t outer = 1, inner = 1;
        Shape   outShape;
        for (int64_t d = 0; d < rank; ++d)
        {
            const int64_t dim = X.shape[(size_t) d];
            if (d == axis)
            {
                if (keepDims)
                {
                    outShape.push_back(1);
                }
                continue;
            }
            outShape.push_back(dim);
            if (d < axis)
            {
                outer *= dim;
            } else
            {
                inner *= dim;
            }
        }
        if (outShape.empty())
        {
            outShape.push_back(1);
        }

        // Typed data pointers are taken before the partition: the const accessors can materialize a
        // mapped buffer on first touch, which must not race inside the parallel body.
        const bool     int64Data = X.dtype == DType::Int64;
        const int64_t *intData   = int64Data ? X.host.i64() : nullptr;
        const float   *floatData = int64Data ? nullptr : X.host.f32();
        int64_t       *indices   = allocOutI64(ctx.t(node.outputs[0]), outShape);
        const int64_t  total     = outer * inner;
        parallelFor(threadCount(ctx.config), 0, total, minChunkForWork(extent), [&](int64_t lo, int64_t hi) {
            if (int64Data)
            {
                argExtremeScanRange(intData, extent, inner, lo, hi, selectLargest, selectLast, indices);
            } else
            {
                argExtremeScanRange(floatData, extent, inner, lo, hi, selectLargest, selectLast, indices);
            }
        });
    }

}} // namespace vknn::cpu
