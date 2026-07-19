#include "core/conv_geom.h"
#include "passes_internal.h"

namespace vknn {

    /// Rewrite a ConvTranspose (stride s, kernel k with k % s == 0) as a regular stride-1 Conv producing
    /// Cout*s*s channels followed by DepthToSpace(s, CRD). ConvTranspose is a true convolution with a
    /// stride-expanded, spatially-flipped kernel; grouping its output pixels by their s*s sub-pixel phase
    /// turns it into that Conv + pixel-shuffle. The gather deconv kernel is memory-bound (one thread per
    /// output pixel, scattered reads, no reuse); the Conv path tiles and reuses its input window.
    ///
    /// Eligible only for a group-1, dilation-1, square-stride>=2 ConvTranspose whose weight is an fp32
    /// initializer and whose padded output extents are stride-multiples that fit the clean sub-pixel form
    /// (every other shape keeps the deconv kernel). A fused residual disqualifies it: the residual carries
    /// the full output shape and cannot ride the coarse Conv.
    ///
    /// Weight rearrangement (verified exact in fp64): output sub-pixel (a,b) of channel oc reads the deconv
    /// taps ky = ay + s*(Cy - ty), kx = ax + s*(Cx - tx), where ay = (a+padH) % s, Cy = floor((a+padH)/s)
    /// (likewise ax/Cx), and ty/tx are the Conv's spatial taps. Only the same products are summed, so the
    /// device result stays at the fp16 floor (the accumulation order shifts, exactly as Winograd does).
    ///
    /// Postcondition: each converted ConvTranspose node is replaced in place by its Conv; the matching
    /// DepthToSpace is appended and inherits the original output tensor id, so consumers are unchanged.
    /// A single topoSort after the loop reorders the appended shuffles ahead of their consumers.
    void subpixelConvTranspose(Graph &g) {
        int               converted = 0;
        std::vector<Node> appended;
        size_t            n0 = g.nodes.size();
        for (size_t ni = 0; ni < n0; ++ni)
        {
            Node ct = g.nodes[ni]; // by value: g.nodes[ni] is replaced with the Conv below
            if (ct.type != OpType::ConvTranspose || ct.inputs.size() < 2 || ct.outputs.empty() || ct.outputs[0] == kNoTensor)
            {
                continue;
            }
            TensorId xId = ct.inputs[0], wId = ct.inputs[1];
            if (xId == kNoTensor || wId == kNoTensor || !g.isInitializer(wId) || g.desc(wId).dtype != DType::Float32)
            {
                continue;
            }
            const Shape &wsh = g.desc(wId).shape; // ONNX ConvTranspose weight [Cin, Cout, kH, kW]
            const Shape &xsh = g.desc(xId).shape; // [N, Cin, H, W]
            if (wsh.size() != 4 || xsh.size() != 4 || ct.attr.geti("group", 1) != 1)
            {
                continue;
            }
            if (ct.fusedResidual != kNoTensor)
            {
                continue; // a fused residual has the full output shape; it cannot ride the coarse Conv
            }
            const auto &st  = ct.attr.getints("strides");
            const auto &dil = ct.attr.getints("dilations");
            int64_t     sh = st.size() == 2 ? st[0] : 1, sw = st.size() == 2 ? st[1] : 1;
            int64_t     dH = dil.size() == 2 ? dil[0] : 1, dW = dil.size() == 2 ? dil[1] : 1;
            if (dH != 1 || dW != 1 || sh < 2 || sh != sw)
            {
                continue; // dilation 1, square stride >= 2 (DepthToSpace carries one blocksize)
            }
            int64_t Cin = wsh[0], Cout = wsh[1], kH = wsh[2], kW = wsh[3], N = xsh[0], H = xsh[2], W = xsh[3];
            if (kH % sh != 0 || kW % sw != 0 || H <= 0 || W <= 0)
            {
                continue; // k not divisible by stride, or dynamic spatial dims
            }
            ConvTransposeGeom geom = convTransposeGeom(H, W, kH, kW, ct.attr);
            int64_t           padH = geom.padH, padW = geom.padW, outH = geom.outH, outW = geom.outW;
            if (outH % sh != 0 || outW % sw != 0)
            {
                continue; // the sub-pixel form emits stride-multiple extents
            }
            int64_t Hc = outH / sh, Wc = outW / sw;
            int     R = (int) (kH / sh), S = (int) (kW / sw);
            auto    CyOf = [&](int a) { return (int) ((a + padH) / sh); };
            auto    CxOf = [&](int b) { return (int) ((b + padW) / sw); };
            // Seed the tap-index min/max reductions past any real tap index (taps are small kernel
            // offsets), so the first CyOf/CxOf sample always replaces the sentinel.
            constexpr int kTapIndexSentinel = 1 << 30;
            int     tyMin = kTapIndexSentinel, tyMax = -kTapIndexSentinel, txMin = kTapIndexSentinel, txMax = -kTapIndexSentinel;
            for (int a = 0; a < (int) sh; ++a)
            {
                tyMin = std::min(tyMin, CyOf(a) - R + 1);
                tyMax = std::max(tyMax, CyOf(a));
            }
            for (int b = 0; b < (int) sw; ++b)
            {
                txMin = std::min(txMin, CxOf(b) - S + 1);
                txMax = std::max(txMax, CxOf(b));
            }
            int     KcY = tyMax - tyMin + 1, KcX = txMax - txMin + 1;
            int64_t padYb = -tyMin, padXb = -txMin;
            int64_t padYe = Hc - H + KcY - 1 - padYb, padXe = Wc - W + KcX - 1 - padXb;
            if (padYb < 0 || padXb < 0 || padYe < 0 || padXe < 0)
            {
                continue; // padding does not fit the clean sub-pixel form; keep the deconv
            }
            // Build the Conv weight K[Cout*s*s, Cin, KcY, KcX] (ONNX Conv layout), zero-filled.
            int64_t    Co = Cout * sh * sw;
            HostBuffer Kb;
            Kb.resizeElems(Co * Cin * KcY * KcX, DType::Float32);
            if (g.desc(wId).dtype == DType::Float64 || (ct.inputs.size() > 2 && ct.inputs[2] != kNoTensor && g.desc(ct.inputs[2]).dtype == DType::Float64))
            {
                continue; // fp64 deconv weight/bias: this fp32 rewrite would misread it; keep the runtime deconv
            }
            const float *Wp   = g.initializers[wId].f32();
            float       *Kp   = Kb.f32();
            auto         Widx = [&](int64_t ic, int64_t oc, int ky, int kx) { return ((ic * Cout + oc) * kH + ky) * kW + kx; };
            auto         Kidx = [&](int64_t co, int64_t ic, int ty, int tx) { return ((co * Cin + ic) * KcY + ty) * KcX + tx; };
            for (int64_t oc = 0; oc < Cout; ++oc)
            {
                for (int a = 0; a < (int) sh; ++a)
                {
                    for (int b = 0; b < (int) sw; ++b)
                    {
                        int64_t inC = oc * (sh * sw) + (a * sw + b); // CRD channel order (DepthToSpace mode CRD)
                        int     ay = (int) ((a + padH) % sh), Cy = CyOf(a);
                        int     ax = (int) ((b + padW) % sw), Cx = CxOf(b);
                        for (int kty = 0; kty < KcY; ++kty)
                        {
                            int ky = ay + (int) sh * (Cy - (kty + tyMin));
                            if (ky < 0 || ky >= kH)
                            {
                                continue;
                            }
                            for (int ktx = 0; ktx < KcX; ++ktx)
                            {
                                int kx = ax + (int) sw * (Cx - (ktx + txMin));
                                if (kx < 0 || kx >= kW)
                                {
                                    continue;
                                }
                                for (int64_t ic = 0; ic < Cin; ++ic)
                                {
                                    Kp[Kidx(inC, ic, kty, ktx)] = Wp[Widx(ic, oc, ky, kx)];
                                }
                            }
                        }
                    }
                }
            }
            TensorDesc kd;
            kd.name          = ct.name + "_spw";
            kd.shape         = {Co, Cin, (int64_t) KcY, (int64_t) KcX};
            kd.isInitializer = true;
            TensorId kId     = g.addTensor(kd);
            g.initializers[kId] = std::move(Kb);
            // Bias: the deconv bias is per output channel; it lands once per output pixel, i.e. once per
            // (oc, phase). Broadcast bias[oc] across the s*s phase channels; DepthToSpace then passes it
            // through so each output pixel is offset exactly once.
            TensorId cbId = kNoTensor;
            if (ct.inputs.size() > 2 && ct.inputs[2] != kNoTensor && g.isInitializer(ct.inputs[2]))
            {
                HostBuffer   Bb;
                Bb.resizeElems(Co, DType::Float32);
                const float *Bp = g.initializers[ct.inputs[2]].f32();
                for (int64_t oc = 0; oc < Cout; ++oc)
                {
                    for (int p = 0; p < (int) (sh * sw); ++p)
                    {
                        Bb.f32()[oc * (sh * sw) + p] = Bp[oc];
                    }
                }
                TensorDesc bd;
                bd.name          = ct.name + "_spb";
                bd.shape         = {Co};
                bd.isInitializer = true;
                cbId             = g.addTensor(bd);
                g.initializers[cbId] = std::move(Bb);
            }
            TensorDesc cd;
            cd.name      = ct.name + "_spc";
            cd.shape     = {N, Co, Hc, Wc};
            TensorId cOut = g.addTensor(cd);

            auto setInts = [&](Node &nd, const char *k, std::vector<int64_t> v) {
                Attr a;
                a.kind = Attr::Ints;
                a.ints = std::move(v);
                nd.attr.map[k] = a;
            };
            Node conv;
            conv.type   = OpType::Conv;
            conv.name   = ct.name + "_spconv";
            conv.inputs = {xId, kId};
            if (cbId != kNoTensor)
            {
                conv.inputs.push_back(cbId);
            }
            conv.outputs = {cOut};
            // A fused activation is per-element and commutes with the DepthToSpace shuffle, so carry it on
            // the coarse Conv (its output is later pixel-shuffled unchanged).
            conv.fusedAct = ct.fusedAct;
            conv.actLo    = ct.actLo;
            conv.actHi    = ct.actHi;
            setInts(conv, "kernel_shape", {(int64_t) KcY, (int64_t) KcX});
            setInts(conv, "strides", {1, 1});
            setInts(conv, "pads", {padYb, padXb, padYe, padXe});
            setInts(conv, "dilations", {1, 1});
            {
                Attr a;
                a.kind                = Attr::Int;
                a.i                   = 1;
                conv.attr.map["group"] = a;
            }
            Node d2s;
            d2s.type    = OpType::DepthToSpace;
            d2s.name    = ct.name + "_spd2s";
            d2s.inputs  = {cOut};
            d2s.outputs = {ct.outputs[0]};
            {
                Attr a;
                a.kind                     = Attr::Int;
                a.i                        = sh;
                d2s.attr.map["blocksize"] = a;
            }
            {
                Attr a;
                a.kind                = Attr::String;
                a.str                 = "CRD";
                d2s.attr.map["mode"] = a;
            }
            g.nodes[ni] = std::move(conv);
            appended.push_back(std::move(d2s));
            converted++;
        }
        if (converted)
        {
            for (auto &nd: appended)
            {
                g.nodes.push_back(std::move(nd));
            }
            g.topoSort();
            VKNN_INFO << "subpixelConvTranspose: rewrote " << converted << " ConvTranspose to Conv + DepthToSpace";
        }
    }

} // namespace vknn
