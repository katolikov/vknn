// Boolean operand reading and the broadcast sweep shared by the CPU logical ops (Or, Xor, Not).
//
// ONNX logical operands are bool tensors. On the host they arrive as fp32 (a bool/uint8 graph input or
// initializer materialized on the canonical fp32 path, or the result of a compare/And/Or/Xor/Not) or as
// int64 (a Cast-to-BOOL result, a shape-arithmetic mask, an int64 initializer). A value is TRUE iff it is
// nonzero: int64 lanes are read exactly through i64(), every other dtype through f32(). NaN, infinities
// and subnormals are therefore true; only +0.0 and -0.0 are false. Results are the canonical fp32
// 1.0 / 0.0 the compare family emits, so they feed a downstream Where/And/Not over fp32 tensors.
//
// The flat GPU kernels (shaders/or.comp, xor.comp, not.comp) apply the identical rule through
// shaders/logical_truth.glsl.
#pragma once
#include "backend/cpu/broadcast.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/parallel.h"
#include "vknn/error.h"
#include "vknn/op.h"
#include <algorithm>
#include <string>
#include <vector>

namespace vknn { namespace cpu {

    /// Canonical fp32 encoding of a boolean result.
    inline constexpr float kLogicalTrue  = 1.0f;
    inline constexpr float kLogicalFalse = 0.0f;

    /// The canonical fp32 value of `truth`.
    inline float logicalValue(bool truth) noexcept {
        return truth ? kLogicalTrue : kLogicalFalse;
    }

    /// Read-only truth view of one bool operand.
    ///
    /// The typed pointer is resolved at construction: the const HostBuffer accessors materialize a
    /// mapped payload on first touch, so a view must be built on the calling thread before a
    /// parallelFor partitions the sweep, never inside a chunk.
    class LogicalOperand {
      public:
        explicit LogicalOperand(const RtTensor &tensor):
            isInt64_(tensor.dtype == DType::Int64), floats_(isInt64_ ? nullptr : tensor.host.f32()), ints_(isInt64_ ? tensor.host.i64() : nullptr) {
        }

        /// True iff element `index` is nonzero (exact for int64; NaN is nonzero).
        bool isTrue(int64_t index) const noexcept {
            return isInt64_ ? ints_[index] != 0 : floats_[index] != 0.0f;
        }

      private:
        bool           isInt64_;
        const float   *floats_;
        const int64_t *ints_;
    };

    /// Throws InvalidArgument naming `node` unless it carries `operandCount` operands and an output.
    inline void requireLogicalOperands(const Node &node, size_t operandCount) {
        bool complete = node.inputs.size() >= operandCount && !node.outputs.empty() && node.outputs[0] != kNoTensor;
        for (size_t k = 0; complete && k < operandCount; ++k)
        {
            complete = node.inputs[k] != kNoTensor;
        }
        if (!complete)
        {
            throw Error(Status::InvalidArgument, std::string(opTypeName(node.type)) + " '" + node.name + "': expects " + std::to_string(operandCount) + " operand(s) and one output");
        }
    }

    /// Operand count of the two-operand logical ops (Or, Xor).
    inline constexpr size_t kLogicalBinaryOperands = 2;

    /// Run a two-operand logical op with NumPy broadcasting: y = combine(aTrue, bTrue) as canonical fp32.
    ///
    /// Output extent per axis is the larger operand extent after right-aligning both shapes; a 0 extent
    /// on either side makes the output extent 0 (NumPy), never 1. A rank-0 result carries one element.
    /// Every output element depends only on its own operand elements, so the sweep partitions over
    /// parallelFor with bytes identical to the serial loop for any thread count.
    /// @throws Error(InvalidArgument) naming the node when an axis pairs two different extents neither of
    ///         which is 1 (the shapes do not broadcast, and a stride walk would read past an operand).
    template <class Combine> void runLogicalBinary(const Node &node, ExecContext &ctx, Combine combine) {
        requireLogicalOperands(node, kLogicalBinaryOperands);
        const RtTensor &A            = ctx.t(node.inputs[0]);
        const RtTensor &B            = ctx.t(node.inputs[1]);
        RtTensor       &Y            = ctx.t(node.outputs[0]);
        const Shape     aShape       = A.shape;
        const Shape     bShape       = B.shape;
        const size_t    rank         = std::max(aShape.size(), bShape.size());
        auto            extentOnAxis = [&](const Shape &shape, size_t axis) -> int64_t {
            const size_t leadingPad = rank - shape.size();
            return axis < leadingPad ? 1 : shape[axis - leadingPad];
        };
        Shape out(rank, 1);
        for (size_t axis = 0; axis < rank; ++axis)
        {
            const int64_t aExtent = extentOnAxis(aShape, axis);
            const int64_t bExtent = extentOnAxis(bShape, axis);
            if (aExtent != bExtent && aExtent != 1 && bExtent != 1)
            {
                throw Error(Status::InvalidArgument, std::string(opTypeName(node.type)) + " '" + node.name + "': operand shapes " + shapeStr(aShape) + " and " + shapeStr(bShape) + " do not broadcast");
            }
            out[axis] = (aExtent == 0 || bExtent == 0) ? 0 : std::max(aExtent, bExtent);
        }
        const int64_t count = elemCount(out);
        // Row-major operand strides, built back to front: 0 on an axis where the operand's extent is 1,
        // so every output coordinate along it re-reads the single source element.
        std::vector<int64_t> aStride(rank), bStride(rank);
        int64_t              aRunning = 1, bRunning = 1;
        for (size_t axis = rank; axis-- > 0;)
        {
            aStride[axis] = extentOnAxis(aShape, axis) == 1 ? 0 : aRunning;
            bStride[axis] = extentOnAxis(bShape, axis) == 1 ? 0 : bRunning;
            aRunning *= extentOnAxis(aShape, axis);
            bRunning *= extentOnAxis(bShape, axis);
        }
        const LogicalOperand a(A);
        const LogicalOperand b(B);
        float               *y = allocOut(Y, out);
        parallelFor(threadCount(ctx.config), 0, count, minChunkForWork(1), [&](int64_t chunkBegin, int64_t chunkEnd) {
            BroadcastWalk walk(out, {aStride.data(), bStride.data()});
            walk.seek(chunkBegin);
            for (int64_t index = chunkBegin; index < chunkEnd; ++index, walk.next())
            {
                y[index] = logicalValue(combine(a.isTrue(walk.offset(0)), b.isTrue(walk.offset(1))));
            }
        });
    }

}} // namespace vknn::cpu
