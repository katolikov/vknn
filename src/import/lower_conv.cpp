#include "passes_internal.h"
#include <cstring>

namespace vknn {

    // Vulkan's per-dimension compute-workgroup-count limit (maxComputeWorkGroupCount): a dispatch
    // whose group count on any axis exceeds this must be split, and the ConvGemm kernel's X/Z axes
    // cannot be split, so a shape that would overflow keeps the plain Conv (whose kernels split on X).
    static constexpr int64_t kMaxWorkGroupCount = 65535;
    // Pixels the ConvGemm kernel packs into one dispatch-Y workgroup at its narrowest spec-constant
    // tile; the eligibility guard rounds OH*OW up by this so every runtime tile choice fits the limit.
    static constexpr int64_t kConvGemmMinPixelTile = 16;

    /// Lower each eligible Conv to a ConvGemm node — one implicit-GEMM kernel over the receptive
    /// field — with the weights repacked to row-major [K][Cout] (K = Cin*KH*KW, k = (ky*KW+kx)*Cin+ic,
    /// channel-fastest so the kernel's A panel gathers whole NC4HW4 channel blocks per k quad)
    /// as a new initializer. The panel's physical row stride is Cout here; the runtime widens the
    /// weight load to STORE4 when that stride is 4-aligned and otherwise repacks the panel to a
    /// zero-padded stride at plan time (convGemmWVec4Route, core/conv_gemm_route.h), so the lowered
    /// initializer stays exactly [K][Cout] — the shape both the CPU reference and the split-K
    /// partial pass read Cout from. The repack is a pure permutation (no arithmetic), so it is exact for
    /// fp32 and fp16 weights alike; the runtime kernel's fp32 K reduction runs in a fixed chunked
    /// order, making the result fp16-floor equivalent to the direct Conv kernel (the accumulation
    /// order shifts, exactly as Winograd's does) and deterministic run to run.
    ///
    /// Targets the KxK shapes the direct kernel serves today: group == 1, KH*KW > 1, static shapes,
    /// constant weight, explicit padding. 1x1 convs keep their register-tiled/split-K kernels, and
    /// the Winograd-eligible shape (3x3, stride 1, pads 1, dilation 1, Cin >= 32, Cout >= 32) keeps
    /// Conv so tuneWino stays in charge there. A non-matching Conv is left untouched.
    void lowerConv(Graph &g) {
        int lowered = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &c = g.nodes[i];
            if (c.type != OpType::Conv || c.fusedResidual != kNoTensor)
            {
                continue;
            }
            if (c.attr.geti("group", 1) != 1)
            {
                continue;
            }
            std::string ap = c.attr.gets("auto_pad", "NOTSET");
            if (!ap.empty() && ap != "NOTSET")
            {
                continue;
            }
            TensorId xId = c.inputs.size() > 0 ? c.inputs[0] : kNoTensor;
            TensorId wId = c.inputs.size() > 1 ? c.inputs[1] : kNoTensor;
            if (xId == kNoTensor || wId == kNoTensor || !g.isInitializer(wId))
            {
                continue;
            }
            const TensorDesc &wd = g.desc(wId);
            if (wd.shape.size() != 4 || (wd.dtype != DType::Float32 && wd.dtype != DType::Float16))
            {
                continue;
            }
            const Shape &xs = g.desc(xId).shape;
            const Shape &os = g.desc(c.outputs[0]).shape;
            if (xs.size() != 4 || os.size() != 4)
            {
                continue; // unresolved shapes: leave the Conv for the runtime paths
            }
            int64_t Cout = wd.shape[0], Cin = wd.shape[1], KH = wd.shape[2], KW = wd.shape[3];
            if (Cin != xs[1] || KH * KW <= 1)
            {
                continue; // grouped-weight layout or a 1x1 (its tuned kernels stay)
            }
            auto ints = [&](const char *k, std::vector<int64_t> d) {
                const auto &v = c.attr.getints(k);
                return v.empty() ? d : v;
            };
            auto st = ints("strides", {1, 1}), pd = ints("pads", {0, 0, 0, 0}), dl = ints("dilations", {1, 1});
            bool wino = KH == 3 && KW == 3 && st[0] == 1 && st[1] == 1 && dl[0] == 1 && dl[1] == 1 && pd[0] == 1 && pd[1] == 1 && pd[2] == 1 && pd[3] == 1 && Cin >= 32 && Cout >= 32;
            if (wino)
            {
                continue; // tuneWino owns this shape
            }
            // The kernel tiles OH*OW on the dispatch Y axis (a specialization-constant tile,
            // kConvGemmMinPixelTile pixels per group at the narrowest) and batch on Z, neither of
            // which the runtime's 1-D split can rescue past the device group-count limit; an oversized
            // spatial output keeps the plain Conv (its kernels split on X). The guard uses the
            // narrowest tile so every runtime tile choice fits.
            if ((os[2] * os[3] + kConvGemmMinPixelTile - 1) / kConvGemmMinPixelTile > kMaxWorkGroupCount || os[0] > kMaxWorkGroupCount)
            {
                continue;
            }

            // Repack W[oc][ic][ky][kx] -> Wt[(ky*KW+kx)*Cin+ic][oc]: a byte-level permutation, exact
            // for any element dtype. wd's reference dies at the addTensor below (Graph::tensors may
            // reallocate), so everything it feeds is captured first.
            const HostBuffer &src = g.initializers.at(wId);
            DType             wdt = wd.dtype;
            size_t            es  = dtypeSize(wdt);
            int64_t           K   = Cin * KH * KW;
            TensorDesc        td;
            td.name          = g.desc(wId).name + "#gemmw";
            td.shape         = {K, Cout};
            td.dtype         = wdt;
            td.isInitializer = true;
            TensorId   wtId  = g.addTensor(td);
            HostBuffer wt;
            wt.resizeElems(K * Cout, wdt);
            const uint8_t *sp = src.bytes.data();
            uint8_t       *dp = wt.bytes.data();
            for (int64_t oc = 0; oc < Cout; ++oc)
            {
                for (int64_t ic = 0; ic < Cin; ++ic)
                {
                    for (int64_t t = 0; t < KH * KW; ++t) // t = ky*KW + kx: the source tap index
                    {
                        int64_t k = t * Cin + ic;
                        std::memcpy(dp + ((size_t) (k * Cout + oc)) * es, sp + ((size_t) ((oc * Cin + ic) * KH * KW + t)) * es, es);
                    }
                }
            }
            g.initializers[wtId] = std::move(wt);

            c.type      = OpType::ConvGemm;
            c.inputs[1] = wtId; // bias (inputs[2], if any) carries over unchanged
            {
                Attr a;
                a.kind                     = Attr::Ints;
                a.ints                     = {KH, KW}; // explicit: the ONNX attr is optional
                c.attr.map["kernel_shape"] = a;
            }
            lowered++;
        }
        if (lowered)
        {
            VKNN_INFO << "lowerConv: lowered " << lowered << " Conv(s) to ConvGemm";
        }
    }

} // namespace vknn
