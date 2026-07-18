// See conv_geom.h. Bodies of the shared plan-time Conv / pool / ConvTranspose geometry resolvers.
#include "core/conv_geom.h"

namespace vknn {

    ConvGeom convGeomEx(int64_t inH, int64_t inW, int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t dh, int64_t dw, const std::vector<int64_t> &pads, const std::string &autoPad) {
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

    ConvGeom convGeom(int64_t inH, int64_t inW, int64_t kh, int64_t kw, const Attributes &attr) {
        auto ints = [&](const char *k, std::vector<int64_t> d) {
            const auto &v = attr.getints(k);
            return v.empty() ? d : v;
        };
        auto st = ints("strides", {1, 1}), pads = ints("pads", {0, 0, 0, 0}), dil = ints("dilations", {1, 1});
        return convGeomEx(inH, inW, kh, kw, st[0], st[1], dil[0], dil[1], pads, attr.gets("auto_pad", "NOTSET"));
    }

    ConvGeom poolGeom(int64_t inH, int64_t inW, const Attributes &attr) {
        auto ints = [&](const char *k, std::vector<int64_t> d) {
            const auto &v = attr.getints(k);
            return v.empty() ? d : v;
        };
        auto ks = ints("kernel_shape", {1, 1}), st = ints("strides", {1, 1}), pads = ints("pads", {0, 0, 0, 0});
        return convGeomEx(inH, inW, ks[0], ks[1], st[0], st[1], 1, 1, pads, attr.gets("auto_pad", "NOTSET"));
    }

    ConvTransposeGeom convTransposeGeom(int64_t inH, int64_t inW, int64_t kh, int64_t kw, const Attributes &attr) {
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
