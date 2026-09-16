// uploadInit host-side rules: how many elements a flat initializer upload binds, how many payload bytes
// that count needs, which device copy of an initializer a segment's memo returns, and which initializers
// keep their host payload after the first upload.
//
// A graph initializer keeps its payload at the stored dtype's native width: 2 bytes per Float16 lane,
// 1 per Int8/UInt8 lane (a native quant or byte tensor is never widened in the graph), 8 per Int64
// lane, 4 for Float32 and for the Int32 label (INT32 materializes to fp32 lanes). The element count
// normally comes from the shape; a rank-0 scalar (numElements() 0) recovers it from the payload size
// divided by that native width, so a scalar Int64, Int8 or UInt8 constant binds its one value instead
// of a zero-length buffer or a spurious short-payload error.
//
// A segment memoizes the device copy each initializer upload produces, so a consumer after the first one
// resolves the copy without re-reading the host payload the first upload may have released. Two readers
// of one constant can run at different storage precisions (an fp32-pinned integer region beside an fp16
// float reader), and an fp16 flat copy holds other bytes than the fp32 one, so the memo keys each copy by
// the store it holds as well as by the tensor. An initializer read at both precisions keeps its host
// payload, which the reader at the second precision uploads from.
//
// tests/test_binary_int_and_cast_fixes.cpp pins this.
#pragma once
#include "vknn/dtype.h"
#include "vknn/graph.h"
#include "vknn/shape.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <set>
#include <tuple>
#include <vector>

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

    /// The payload a device copy of an initializer holds.
    enum class InitializerDeviceStore : uint8_t {
        FlatFp16, ///< flat element lanes at fp16 (uploadInit at a half-precision node)
        FlatFp32, ///< flat element lanes at fp32 (uploadInit at a full-precision node)
        RawBytes, ///< the payload bytes unconverted (uploadInitRaw), the same at either precision
    };

    /// The store a flat upload at `fp16Storage` precision holds.
    inline InitializerDeviceStore flatInitializerStore(bool fp16Storage) noexcept {
        return fp16Storage ? InitializerDeviceStore::FlatFp16 : InitializerDeviceStore::FlatFp32;
    }

    /// Key of a segment's memo of initializer device copies: one copy per (initializer, store).
    struct InitializerDeviceCopyKey {
        TensorId               tensor = kNoTensor;
        InitializerDeviceStore store  = InitializerDeviceStore::FlatFp32;

        bool operator<(const InitializerDeviceCopyKey &other) const noexcept {
            return std::tie(tensor, store) < std::tie(other.tensor, other.store);
        }
    };

    /// Initializers the nodes `nodeIndices` of `g` read (an operand, the fused residual or the fused bias)
    /// at both storage precisions, where `nodeStoresFp16(node)` is the precision a node's kernel runs at.
    /// A reader at the second precision uploads its own copy from the host payload, so these must keep
    /// the payload after the first upload.
    inline std::set<TensorId> initializersReadAtBothPrecisions(const Graph &g, const std::vector<int> &nodeIndices, const std::function<bool(const Node &)> &nodeStoresFp16) {
        std::set<TensorId> readAtFp16, readAtFp32, both;
        for (int nodeIndex: nodeIndices)
        {
            const Node &node     = g.nodes[(size_t) nodeIndex];
            const bool  fp16Node = nodeStoresFp16(node);
            auto        noteRead = [&](TensorId tensor) {
                if (tensor == kNoTensor || !g.isInitializer(tensor))
                {
                    return;
                }
                (fp16Node ? readAtFp16 : readAtFp32).insert(tensor);
                if ((fp16Node ? readAtFp32 : readAtFp16).count(tensor))
                {
                    both.insert(tensor);
                }
            };
            for (TensorId operand: node.inputs)
            {
                noteRead(operand);
            }
            noteRead(node.fusedResidual);
            noteRead(node.fusedBias);
        }
        return both;
    }

} // namespace vknn
