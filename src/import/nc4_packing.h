// When two logical shapes store their elements at the same NC4HW4 buffer positions: the condition
// under which a layout-agnostic op (a metadata reshape, a Cast, a channel shuffle) may keep NC4HW4, its
// output reading the input's buffer positions as its own.
#pragma once
#include "vknn/nchw.h"
#include "vknn/shape.h"
#include <cstdint>
#include <initializer_list>

namespace vknn {

    /// The canonical form of the map from a logical shape's row-major element index to its NC4HW4
    /// buffer position, together with the buffer's footprint.
    ///
    /// NC4HW4 stores the NCHW view of a shape (NCHW::from) as lane quads: channel c lives in block
    /// cb = c / kNC4Block at lane c % kNC4Block, the quad at (n, cb, h, w) has index
    /// ((n * cBlocks(C) + cb) * H + h) * W + w, and the element sits at quad * kNC4Block + lane; the
    /// lanes past C in the last block are padding. So the element at row-major index
    /// i = ((n * C + c) * H + h) * W + w lands at a position that depends on N, C and the plane size
    /// H * W only (h * W + w enumerates a plane in row-major order for every H/W split), and the
    /// footprint N * cBlocks(C) * kNC4Block * H * W depends on the same three numbers. Two reductions
    /// make the triple canonical, so equal keys hold exactly when the maps and footprints are equal:
    ///  - one channel: every element starts its own quad (position i * kNC4Block), so any split of
    ///    the elements between N and the plane stores like one plane of N * H * W elements;
    ///  - a channel count that is a multiple of kNC4Block (or a single batch): the block index
    ///    n * cBlocks(C) + cb equals (n * C + c) / kNC4Block and the lane equals (n * C + c) % kNC4Block,
    ///    so the batches store like one batch of N * C channels.
    /// A shape with a zero dim stores nothing; every such shape shares one key.
    struct Nc4PackingKey {
        int64_t batch    = 0;
        int64_t channels = 0;
        int64_t plane    = 0;

        bool operator==(const Nc4PackingKey &other) const noexcept {
            return batch == other.batch && channels == other.channels && plane == other.plane;
        }
    };

    /// The canonical NC4HW4 packing key of a shape with resolved (non-negative) dims.
    /// @param shape Logical tensor shape.
    /// @returns The key two shapes share exactly when their NC4HW4 storage maps and footprints agree.
    inline Nc4PackingKey nc4PackingKey(const Shape &shape) {
        const NCHW    view  = NCHW::from(shape);
        const int64_t plane = view.h * view.w;
        if (view.elems() == 0)
        {
            return {};
        }
        constexpr int64_t singleChannel = 1;
        constexpr int64_t singleBatch   = 1;
        if (view.c == singleChannel)
        {
            return {singleBatch, singleChannel, view.n * plane};
        }
        if (view.n == singleBatch || view.c % kNC4Block == 0)
        {
            return {singleBatch, view.n * view.c, plane};
        }
        return {view.n, view.c, plane};
    }

    /// Whether a buffer holding `from` in NC4HW4 already holds `to` in NC4HW4, element for element
    /// (padding lanes included): true for identical shapes, and for resolved shapes with equal
    /// packing keys. A shape with an unresolved (negative) dim proves nothing beyond identity.
    /// @param from Shape the buffer was written for.
    /// @param to   Shape the buffer is read as.
    /// @returns True when an NC4HW4 byte copy (or alias) of `from` is a valid NC4HW4 `to`.
    inline bool nc4PackingIdentical(const Shape &from, const Shape &to) {
        if (from == to)
        {
            return true;
        }
        for (const Shape *shape: {&from, &to})
        {
            for (int64_t dim: *shape)
            {
                if (dim < 0)
                {
                    return false;
                }
            }
        }
        return nc4PackingKey(from) == nc4PackingKey(to);
    }

} // namespace vknn
