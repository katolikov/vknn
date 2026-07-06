#pragma once
#include "vknn/op.h"
#include <string>
#include <vector>

namespace vknn {

    /// Resolved forward Conv/pool geometry: begin/end pads per spatial axis plus the output extent.
    /// pads() re-serializes to the ONNX order [top, left, bottom, right].
    struct ConvGeom {
        int64_t outH, outW;
        int64_t padT, padL; // begin pads (top, left): the input-window origin offset
        int64_t padB, padR; // end pads (bottom, right): folded into outH/outW only
        std::vector<int64_t> pads() const {
            return {padT, padL, padB, padR};
        }
    };

    /// Forward Conv/pool geometry from explicit parameters, honoring ONNX auto_pad. The single
    /// source of truth for every consumer (shape inference, the CPU ops, the Vulkan prepares):
    /// nothing bakes resolved pads into the graph, so a re-run on a new input shape re-resolves.
    ///   NOTSET       pads as given; out = floor((in + begin + end - span)/stride) + 1 with
    ///                span = (k-1)*dilation + 1 (the pool arms pass dilation 1).
    ///   SAME_UPPER / out = ceil(in/stride); total = max(0, (out-1)*stride + span - in) (the ORT
    ///   SAME_LOWER   clamp), split begin/end with the odd unit at the end for SAME_UPPER and at
    ///                the begin for SAME_LOWER.
    ///   VALID        zero pads (an accompanying pads attr is ignored, per the ONNX spec).
    inline ConvGeom convGeomEx(int64_t inH, int64_t inW, int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t dh, int64_t dw, const std::vector<int64_t> &pads, const std::string &autoPad) {
        int64_t  spanH = dh * (kh - 1) + 1, spanW = dw * (kw - 1) + 1;
        ConvGeom g {};
        if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER")
        {
            g.outH        = (inH + sh - 1) / sh;
            g.outW        = (inW + sw - 1) / sw;
            int64_t totH  = (g.outH - 1) * sh + spanH - inH;
            int64_t totW  = (g.outW - 1) * sw + spanW - inW;
            totH          = totH < 0 ? 0 : totH;
            totW          = totW < 0 ? 0 : totW;
            const bool up = (autoPad == "SAME_UPPER"); // begin gets the smaller half
            g.padT        = up ? totH / 2 : totH - totH / 2;
            g.padL        = up ? totW / 2 : totW - totW / 2;
            g.padB        = totH - g.padT;
            g.padR        = totW - g.padL;
        } else
        {
            if (autoPad != "VALID") // NOTSET (or absent): the explicit pads attr
            {
                g.padT = pads[0];
                g.padL = pads[1];
                g.padB = pads[2];
                g.padR = pads[3];
            }
            g.outH = (inH + g.padT + g.padB - spanH) / sh + 1;
            g.outW = (inW + g.padL + g.padR - spanW) / sw + 1;
        }
        return g;
    }

    /// Forward-Conv geometry from the node attributes (strides / dilations / pads / auto_pad);
    /// the kernel extent comes from the caller (the ONNX weight shape [M, C/g, kH, kW]).
    inline ConvGeom convGeom(int64_t inH, int64_t inW, int64_t kh, int64_t kw, const Attributes &attr) {
        auto ints = [&](const char *k, std::vector<int64_t> d) {
            const auto &v = attr.getints(k);
            return v.empty() ? d : v;
        };
        auto st = ints("strides", {1, 1}), pads = ints("pads", {0, 0, 0, 0}), dil = ints("dilations", {1, 1});
        return convGeomEx(inH, inW, kh, kw, st[0], st[1], dil[0], dil[1], pads, attr.gets("auto_pad", "NOTSET"));
    }

    /// Pool geometry from the node attributes (kernel_shape / strides / pads / auto_pad). Dilation
    /// is fixed at 1: the pool kernels do not dilate, so the extent math matches what they compute.
    inline ConvGeom poolGeom(int64_t inH, int64_t inW, const Attributes &attr) {
        auto ints = [&](const char *k, std::vector<int64_t> d) {
            const auto &v = attr.getints(k);
            return v.empty() ? d : v;
        };
        auto ks = ints("kernel_shape", {1, 1}), st = ints("strides", {1, 1}), pads = ints("pads", {0, 0, 0, 0});
        return convGeomEx(inH, inW, ks[0], ks[1], st[0], st[1], 1, 1, pads, attr.gets("auto_pad", "NOTSET"));
    }

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
