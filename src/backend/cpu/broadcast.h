// Row-major broadcast index walker and output-shape rule shared by the CPU elementwise ops.
#pragma once
#include "vknn/error.h"
#include "vknn/op.h"
#include "vknn/shape.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace vknn { namespace cpu {

    /// Output shape of NumPy broadcasting two operand shapes. The shapes right-align (an operand of lower
    /// rank reads extent 1 on its missing leading axes), a size-1 axis stretches to the other operand's
    /// extent, and a 0 extent broadcasts to 0, never to 1. A rank-0 operand broadcasts over any shape.
    /// @param node the op evaluating the broadcast, named in the error.
    /// @throws Error(InvalidArgument) "<Op> '<name>': operand shapes A and B do not broadcast (axis k)" when
    ///         an axis pairs two different extents neither of which is 1: a stride walk over such shapes
    ///         would read past the smaller operand's buffer.
    inline Shape broadcastOutputShape(const Node &node, const Shape &shapeA, const Shape &shapeB) {
        const size_t rank     = std::max(shapeA.size(), shapeB.size());
        auto         extentOf = [&](const Shape &shape, size_t axis) -> int64_t {
            const size_t leadingPadding = rank - shape.size();
            return axis < leadingPadding ? 1 : shape[axis - leadingPadding];
        };
        Shape out(rank, 1);
        for (size_t axis = 0; axis < rank; ++axis)
        {
            const int64_t extentA = extentOf(shapeA, axis);
            const int64_t extentB = extentOf(shapeB, axis);
            if (extentA != extentB && extentA != 1 && extentB != 1)
            {
                throw Error(Status::InvalidArgument, std::string(opTypeName(node.type)) + " '" + node.name + "': operand shapes " + shapeStr(shapeA) + " and " + shapeStr(shapeB) + " do not broadcast (axis " + std::to_string(axis) + ")");
            }
            out[axis] = (extentA == 0 || extentB == 0) ? 0 : std::max(extentA, extentB);
        }
        return out;
    }

    /// Walks a row-major output shape one flat element at a time, carrying the source element
    /// offset of every broadcast operand.
    ///
    /// Each operand contributes a rank-length stride array: 0 on a broadcast axis (operand extent
    /// 1, so every output coordinate along that axis re-reads one source element) and the
    /// operand's own row-major stride elsewhere. Those are exactly the `oa`/`ob` arrays the
    /// elementwise ops build back-to-front, and BroadcastWalk borrows them — they must outlive it.
    ///
    /// seek() resolves an arbitrary flat index with one pass of divides. next() advances by an
    /// odometer carry: the fast axis costs one add per operand, and a carry into axis d costs one
    /// correction per operand. A full sweep is therefore O(1) amortized integer adds per element
    /// rather than the O(rank^2) multiplies plus O(rank) 64-bit divides and mods a per-element
    /// unravel spends. The offsets are the same integers the unravel produces, so a consumer's
    /// float expression and its evaluation order are unchanged.
    class BroadcastWalk {
      public:
        /// `out` is the broadcast output shape; `strides` holds one rank-length stride array per
        /// operand, indexed later by offset(k). A rank-0 (scalar) output yields all-zero offsets.
        BroadcastWalk(const Shape &out, std::vector<const int64_t *> strides):
            dims_(out.begin(), out.end()), str_(std::move(strides)), off_(str_.size(), 0), coord_(dims_.size(), 0) {
        }

        /// Position the walker at flat output index `lin`, which must be in [0, elemCount(out)).
        void seek(int64_t lin) {
            std::fill(off_.begin(), off_.end(), 0);
            std::fill(coord_.begin(), coord_.end(), 0);
            for (size_t d = dims_.size(); d-- > 0;)
            {
                if (dims_[d] <= 0)
                {
                    continue; // a zero extent means the output is empty and no element is visited
                }
                int64_t id = lin % dims_[d];
                lin /= dims_[d];
                coord_[d] = id;
                for (size_t k = 0; k < str_.size(); ++k)
                {
                    off_[k] += id * str_[k][d];
                }
            }
        }

        /// Source element offset of operand `k` at the current position.
        int64_t offset(size_t k) const noexcept {
            return off_[k];
        }

        /// Advance to the next flat output index (row-major order).
        void next() {
            for (size_t d = dims_.size(); d-- > 0;)
            {
                for (size_t k = 0; k < str_.size(); ++k)
                {
                    off_[k] += str_[k][d];
                }
                if (++coord_[d] < dims_[d])
                {
                    return;
                }
                // Axis d wrapped: reset its coordinate and undo the whole extent's worth of stride
                // that the increments above accumulated, then carry into axis d-1.
                coord_[d] = 0;
                for (size_t k = 0; k < str_.size(); ++k)
                {
                    off_[k] -= str_[k][d] * dims_[d];
                }
            }
        }

      private:
        std::vector<int64_t>         dims_;
        std::vector<const int64_t *> str_;
        std::vector<int64_t>         off_;
        std::vector<int64_t>         coord_;
    };

}} // namespace vknn::cpu
