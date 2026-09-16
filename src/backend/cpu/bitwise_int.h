// Integer semantics of the bitwise ops (BitwiseAnd/Or/Xor, BitwiseNot, BitShift) on the CPU oracle, and
// the broadcast sweeps their kernels share.
//
// Operand values: an Int64 runtime tensor is read exactly; every other runtime dtype carries integers as
// fp32 lanes, read by integerFromFloat (NaN reads 0, saturated to the int64 range, truncated toward
// zero), so no conversion is undefined. All arithmetic is two's-complement int64. The result is stored as
// int64 when any operand's runtime dtype is Int64, otherwise as the fp32 value of the int64 result (the
// Binary rule). The GLSL kernels (shaders/bitwise_int.glsl and the three bitwise .comp files) compute the
// same integers on float lanes and store the same fp32 lanes: BitwiseNot and BitShift for every fp32
// operand, BitwiseAnd/Or/Xor for operands within the int32 range.
#pragma once
#include "backend/cpu/broadcast.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/parallel.h"
#include "core/bitwise_attrs.h"
#include "vknn/op.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace vknn { namespace cpu { namespace bitwise {

    using vknn::bitwise::IntegerWidth;
    using vknn::bitwise::kInt64Bits;
    using vknn::bitwise::ShiftDirection;

    /// 2^63 as fp32 (exact): the smallest fp32 value above INT64_MAX.
    inline constexpr float kInt64UpperBoundFloat = 9223372036854775808.0f;
    /// -2^63 as fp32 (exact): INT64_MIN.
    inline constexpr float kInt64LowerBoundFloat = -9223372036854775808.0f;

    /// Integer value of one fp32-carried operand element: NaN reads 0, a value outside the int64 range
    /// saturates to INT64_MIN / INT64_MAX, anything else truncates toward zero.
    inline int64_t integerFromFloat(float value) noexcept {
        if (std::isnan(value))
        {
            return 0;
        }
        if (value >= kInt64UpperBoundFloat)
        {
            return std::numeric_limits<int64_t>::max();
        }
        if (value < kInt64LowerBoundFloat)
        {
            return std::numeric_limits<int64_t>::min();
        }
        return (int64_t) value;
    }

    /// Two's-complement bit pattern of `value` as unsigned (conversion to unsigned is modulo 2^64).
    inline uint64_t unsignedBits(int64_t value) noexcept {
        return static_cast<uint64_t>(value);
    }

    /// Signed int64 whose two's-complement bit pattern is `bits`.
    inline int64_t signedFromBits(uint64_t bits) noexcept {
        int64_t value;
        std::memcpy(&value, &bits, sizeof value);
        return value;
    }

    /// The low `bits` bits set (bits in [1, 64]); widthMask(64) is all ones.
    inline uint64_t widthMask(int bits) noexcept {
        return bits >= kInt64Bits ? std::numeric_limits<uint64_t>::max() : ((uint64_t {1} << bits) - 1);
    }

    inline int64_t bitwiseAnd(int64_t a, int64_t b) noexcept {
        return a & b;
    }
    inline int64_t bitwiseOr(int64_t a, int64_t b) noexcept {
        return a | b;
    }
    inline int64_t bitwiseXor(int64_t a, int64_t b) noexcept {
        return a ^ b;
    }

    /// Complement at the operand width: a signed type or a 64-bit width is ~v (width-independent under sign
    /// extension); an unsigned type narrower than 64 bits keeps only its low `bits` bits of ~v, which is
    /// mask - v for every v in [0, mask].
    inline int64_t bitwiseNot(int64_t value, const IntegerWidth &width) noexcept {
        if (width.isSigned || width.bits >= kInt64Bits)
        {
            return ~value;
        }
        return signedFromBits(widthMask(width.bits) & ~unsignedBits(value));
    }

    /// Shift at the operand width. A shift count outside [0, bits) yields 0. The operand is first reduced
    /// to its low `bits` bits; LEFT keeps the low `bits` bits of the shifted pattern, RIGHT is a logical
    /// shift. The result is the int64 with that bit pattern.
    inline int64_t bitShift(int64_t value, int64_t shift, ShiftDirection direction, int bits) noexcept {
        if (shift < 0 || shift >= bits)
        {
            return 0;
        }
        const uint64_t mask    = widthMask(bits);
        const uint64_t operand = unsignedBits(value) & mask;
        const uint64_t shifted = direction == ShiftDirection::Left ? ((operand << shift) & mask) : (operand >> shift);
        return signedFromBits(shifted);
    }

    /// Typed read views of one operand, resolved once before a parallel sweep: the const host accessors
    /// materialize a mapped payload on first touch, which must not happen concurrently inside a chunk.
    /// The storage decision reads the tensor's dtype, never a data pointer: an empty tensor's accessor is
    /// null whatever its dtype.
    struct OperandView {
        bool           int64    = false;   ///< the runtime dtype is Int64
        const int64_t *integers = nullptr; ///< an Int64 tensor's elements
        const float   *floats   = nullptr; ///< every other runtime dtype's elements (fp32 lanes)

        explicit OperandView(const RtTensor &tensor): int64(tensor.dtype == DType::Int64) {
            if (int64)
            {
                integers = tensor.host.i64();
            } else
            {
                floats = tensor.host.f32();
            }
        }
        bool isInt64() const noexcept {
            return int64;
        }
        int64_t at(int64_t index) const noexcept {
            return int64 ? integers[index] : integerFromFloat(floats[index]);
        }
    };

    /// Evaluate `integerOp(a, b)` over node.inputs[0..1] with NumPy broadcasting into node.outputs[0]:
    /// int64 storage when either operand is Int64, else the fp32 value of each int64 result. Elements are
    /// independent, so the sweep partitions across threads with bit-identical output.
    template <class IntegerOp> void runBroadcastInteger(const Node &node, ExecContext &ctx, IntegerOp integerOp) {
        const RtTensor &operandA = ctx.t(node.inputs[0]);
        const RtTensor &operandB = ctx.t(node.inputs[1]);
        RtTensor       &result   = ctx.t(node.outputs[0]);
        const Shape    &shapeA = operandA.shape, &shapeB = operandB.shape;
        const size_t    rank = std::max(shapeA.size(), shapeB.size());
        // NumPy broadcasting right-aligns the shapes: a lower-rank operand reads extent 1 on its padded
        // leading axes.
        auto extentOf = [&](const Shape &shape, size_t axis) -> int64_t {
            const size_t padding = rank - shape.size();
            return axis < padding ? 1 : shape[axis - padding];
        };
        Shape out(rank, 1);
        for (size_t axis = 0; axis < rank; ++axis)
        {
            const int64_t extentA = extentOf(shapeA, axis), extentB = extentOf(shapeB, axis);
            out[axis] = (extentA == 0 || extentB == 0) ? 0 : std::max(extentA, extentB); // a 0 extent broadcasts to 0
        }
        const int64_t count = cpu::elemCount(out); // a rank-0 result carries its one element
        // Row-major source strides per operand, 0 on a broadcast axis so every output coordinate along it
        // re-reads the operand's single element.
        std::vector<int64_t> stridesA(rank), stridesB(rank);
        int64_t              strideA = 1, strideB = 1;
        for (int axis = (int) rank - 1; axis >= 0; --axis)
        {
            stridesA[axis] = extentOf(shapeA, axis) == 1 ? 0 : strideA;
            stridesB[axis] = extentOf(shapeB, axis) == 1 ? 0 : strideB;
            strideA *= extentOf(shapeA, axis);
            strideB *= extentOf(shapeB, axis);
        }
        const OperandView viewA(operandA), viewB(operandB);
        const bool        int64Result = viewA.isInt64() || viewB.isInt64();
        int64_t          *integerOut  = int64Result ? cpu::allocOutI64(result, out) : nullptr;
        float            *floatOut    = int64Result ? nullptr : cpu::allocOut(result, out);
        cpu::parallelFor(cpu::threadCount(ctx.config), 0, count, cpu::minChunkForWork(1), [&](int64_t first, int64_t end) {
            cpu::BroadcastWalk walk(out, {stridesA.data(), stridesB.data()});
            walk.seek(first);
            for (int64_t index = first; index < end; ++index, walk.next())
            {
                const int64_t value = integerOp(viewA.at(walk.offset(0)), viewB.at(walk.offset(1)));
                if (int64Result)
                {
                    integerOut[index] = value;
                } else
                {
                    floatOut[index] = (float) value;
                }
            }
        });
    }

    /// Evaluate `integerOp(v)` over node.inputs[0] into node.outputs[0] (same shape), with the storage
    /// rule of runBroadcastInteger.
    template <class IntegerOp> void runElementwiseInteger(const Node &node, ExecContext &ctx, IntegerOp integerOp) {
        const RtTensor   &operand = ctx.t(node.inputs[0]);
        RtTensor         &result  = ctx.t(node.outputs[0]);
        const Shape       shape   = operand.shape;
        const int64_t     count   = cpu::elemCount(shape);
        const OperandView view(operand);
        const bool        int64Result = view.isInt64();
        int64_t          *integerOut  = int64Result ? cpu::allocOutI64(result, shape) : nullptr;
        float            *floatOut    = int64Result ? nullptr : cpu::allocOut(result, shape);
        cpu::parallelFor(cpu::threadCount(ctx.config), 0, count, cpu::minChunkForWork(1), [&](int64_t first, int64_t end) {
            for (int64_t index = first; index < end; ++index)
            {
                const int64_t value = integerOp(view.at(index));
                if (int64Result)
                {
                    integerOut[index] = value;
                } else
                {
                    floatOut[index] = (float) value;
                }
            }
        });
    }

}}} // namespace vknn::cpu::bitwise
