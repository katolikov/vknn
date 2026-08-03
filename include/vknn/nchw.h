#pragma once
#include "vknn/common.h"
#include "vknn/tensor_format_enum.h"
#include <cstdint>

namespace vknn {

    /// A shape interpreted as NCHW. Ranks above 4 fold their leading dims into N (the last three dims
    /// are C, H, W), so a multi-view input like [1,8,3,224,224] becomes N=8, C=3, H=224, W=224 and
    /// N*C*H*W stays equal to the element count.
    struct NCHW {
        int64_t n = 1, c = 1, h = 1, w = 1;
        /// Interpret a logical shape as NCHW, folding rank>4 into N and right-aligning C, H, W.
        /// Lower ranks map dims left-to-right (rank 3 -> N,C,H with W=1; rank 2 -> N,C; rank 1 -> C),
        /// leaving the missing trailing dims at their default of 1.
        /// @param s Logical tensor shape.
        /// @returns The NCHW view of `s`.
        static NCHW from(const Shape &s) {
            NCHW r;
            if (s.size() >= 4)
            {
                size_t k = s.size();
                r.w      = s[k - 1];
                r.h      = s[k - 2];
                r.c      = s[k - 3];
                r.n      = 1;
                for (size_t i = 0; i + 3 < k; ++i)
                {
                    r.n *= s[i];
                }
            } else if (s.size() == 3)
            {
                // e.g. a reshaped detection map [N,C,L]; treat the trailing dim as spatial so the NC4HW4
                // element count (cBlocks(C)*4*H*W) matches and 3D tensors pack/unpack across GPU<->CPU.
                r.n = s[0];
                r.c = s[1];
                r.h = s[2];
                r.w = 1;
            } else if (s.size() == 2)
            {
                r.n = s[0];
                r.c = s[1];
                r.h = 1;
                r.w = 1;
            } else if (s.size() == 1)
            { r.c = s[0]; }
            return r;
        }
        /// Dense (unpadded) element count N*C*H*W of this shape.
        int64_t elems() const noexcept {
            return n * c * h * w;
        }
    };

    /// Channel-block width of the NC4HW4 boundary layout: channels are packed in groups of four.
    inline constexpr int64_t kNC4Block = 4;

    /// Rank of the canonical NCHW view: the rank a shape right-aligns into for broadcast
    /// classification and NC4HW4 packing (leading missing axes read as 1).
    inline constexpr size_t kNchwRank = 4;

    /// Number of channel blocks (of kNC4Block channels each) needed to hold `c` channels in NC4HW4,
    /// rounding up so a partial final block is counted.
    /// @param c Logical channel count.
    /// @returns ceil(c / kNC4Block).
    inline int64_t cBlocks(int64_t c) {
        return (c + kNC4Block - 1) / kNC4Block; // ceil-div: the +block-1 bias rounds a partial block up
    }

    /// Stored element count for a logical NCHW shape in a boundary layout. NCHW and NHWC are dense
    /// (N*C*H*W); NC4HW4 pads channels up to a multiple of kNC4Block. (Auto/Unknown have no own count.)
    /// @param fmt Boundary layout the tensor is stored in.
    /// @param x   Logical NCHW shape.
    /// @returns The number of elements physically stored for `x` under `fmt`.
    inline int64_t formatElems(TensorFormat fmt, const NCHW &x) {
        return fmt == TensorFormat::NC4HW4 ? x.n * cBlocks(x.c) * kNC4Block * x.h * x.w : x.elems();
    }

    /// Whether a caller-bound input shape is compatible with the planned shape it replaces. Two
    /// contracts, selected by @p frozenPlan:
    ///
    ///  - frozenPlan=false (input consumed only by CPU segments): the dynamic-reshape contract — CPU
    ///    ops recompute geometry from the runtime shape, so any rebinding whose stored footprint
    ///    (flat dense, or NC4HW4-padded) equals the planned allocation is valid.
    ///
    ///  - frozenPlan=true (input consumed by a GPU segment): push constants and dispatch geometry
    ///    are frozen from the planned shape while the boundary pack follows the runtime shape, so
    ///    the shapes must produce IDENTICAL stored bytes. Flat tensors store dense row-major (any
    ///    equal element count reinterprets losslessly); NC4HW4 packing is a function of N, C, and
    ///    the spatial product h*w (block index = (n*cBlocks(C) + cb)*h*w + spatial), so those must
    ///    match — equal PADDED footprints alone are not enough: [1,1,H,W] bound against a planned
    ///    [1,2,H,W] pads to the same block count but interleaves channels differently, and the
    ///    kernels then silently misread every channel (a wrong-values class, not an overrun).
    ///
    /// @param got        The caller's bound shape (non-empty, already known != planned).
    /// @param planned    The bucket's planned shape.
    /// @param flatStore  True when the tensor stores flat/dense (TensorDesc::gpuFlat), false = NC4HW4.
    /// @param frozenPlan True when a GPU segment consumes the input (frozen pack/dispatch geometry).
    /// @returns True when the rebinding is safe under the applicable contract.
    inline bool boundShapeCompatible(const Shape &got, const Shape &planned, bool flatStore, bool frozenPlan) {
        const NCHW a = NCHW::from(got), b = NCHW::from(planned);
        if (frozenPlan)
        {
            return flatStore ? a.elems() == b.elems() : (a.n == b.n && a.c == b.c && a.h * a.w == b.h * b.w);
        }
        const TensorFormat fmt = flatStore ? TensorFormat::NCHW : TensorFormat::NC4HW4;
        return formatElems(fmt, a) == formatElems(fmt, b);
    }

} // namespace vknn
