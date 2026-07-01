#pragma once
#include "vknn/common.h"
#include "vknn/tensor_format_enum.h"
#include <cstdint>

namespace vknn {

    // A shape interpreted as NCHW. Ranks above 4 fold their leading dims into N (the last three dims are
    // C, H, W), so a multi-view input like [1,8,3,224,224] becomes N=8, C=3, H=224, W=224 and N*C*H*W
    // stays equal to the element count.
    struct NCHW {
        int64_t     n = 1, c = 1, h = 1, w = 1;
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
        int64_t elems() const {
            return n * c * h * w;
        }
    };

    // Number of channel blocks of 4 (for NC4HW4).
    inline int64_t cBlocks(int64_t c) {
        return (c + 3) / 4;
    }

    // Stored element count for a logical NCHW shape in a boundary layout. NCHW and NHWC are dense
    // (N*C*H*W); NC4HW4 pads channels to a multiple of 4. (Auto/Unknown have no own count.)
    inline int64_t formatElems(TensorFormat fmt, const NCHW &x) {
        return fmt == TensorFormat::NC4HW4 ? x.n * cBlocks(x.c) * 4 * x.h * x.w : x.elems();
    }

} // namespace vknn
