#pragma once
#include "vknn/op.h"
#include <string>
#include <vector>

namespace vknn {

    /// Resolved 2D ConvTranspose output geometry. padH/padW are the BEGIN pads only: the gather
    /// kernels offset the input window by the begin pad, while the end pad affects only the output
    /// extent (already folded into outH/outW).
    struct ConvTransposeGeom {
        int64_t outH, outW, padH, padW;
    };

    /// ONNX ConvTranspose output size and begin-pads from the node attributes, honoring auto_pad
    /// (SAME_UPPER / SAME_LOWER / VALID / NOTSET) and the output_shape attr.
    ///   out_natural = stride*(in-1) + output_padding + ((k-1)*dilation+1).
    ///   total_pad   = out_natural - target, where target = output_shape if given else in*stride.
    /// The total is split with the larger half at the begin for SAME_LOWER and for output_shape
    /// without auto_pad, and at the end for SAME_UPPER. Each pad is clamped to >= 0 (ORT never pads
    /// up past out_natural), and for SAME the output is recomputed from the clamped pads, so a
    /// stride>kernel SAME lands on out_natural rather than in*stride.
    inline ConvTransposeGeom convTransposeGeom(int64_t inH, int64_t inW, int64_t kh, int64_t kw, const Attributes &attr) {
        auto ints = [&](const char *k, std::vector<int64_t> d) {
            const auto &v = attr.getints(k);
            return v.empty() ? d : v;
        };
        auto              st     = ints("strides", {1, 1});
        auto              pads   = ints("pads", {0, 0, 0, 0});
        auto              dil    = ints("dilations", {1, 1});
        auto              outpad = ints("output_padding", {0, 0});
        const std::string ap     = attr.gets("auto_pad", "NOTSET");
        const auto       &osh    = attr.getints("output_shape");

        int64_t           natH = (inH - 1) * st[0] + dil[0] * (kh - 1) + 1 + outpad[0];
        int64_t           natW = (inW - 1) * st[1] + dil[1] * (kw - 1) + 1 + outpad[1];
        ConvTransposeGeom g;
        g.outH = natH - pads[0] - pads[2];
        g.outW = natW - pads[1] - pads[3];
        g.padH = pads[0];
        g.padW = pads[1];

        const bool same = (ap == "SAME_UPPER" || ap == "SAME_LOWER");
        if (same || osh.size() == 2)
        {
            int64_t    tgtH  = (osh.size() == 2) ? osh[0] : inH * st[0];
            int64_t    tgtW  = (osh.size() == 2) ? osh[1] : inW * st[1];
            int64_t    totH  = natH - tgtH;
            int64_t    totW  = natW - tgtW;
            const bool upper = (ap == "SAME_UPPER"); // begin gets the smaller half
            int64_t    pbH = upper ? totH / 2 : totH - totH / 2, peH = totH - pbH;
            int64_t    pbW = upper ? totW / 2 : totW - totW / 2, peW = totW - pbW;
            pbH = pbH < 0 ? 0 : pbH, peH = peH < 0 ? 0 : peH;
            pbW = pbW < 0 ? 0 : pbW, peW = peW < 0 ? 0 : peW;
            g.padH = pbH;
            g.padW = pbW;
            g.outH = (osh.size() == 2) ? tgtH : natH - pbH - peH;
            g.outW = (osh.size() == 2) ? tgtW : natW - pbW - peW;
        }
        return g;
    }

} // namespace vknn
