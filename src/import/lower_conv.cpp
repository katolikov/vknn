#include "passes_internal.h"
#include <cstring>

namespace vknn {

    /// Lower each eligible Conv to a ConvGemm node — one implicit-GEMM kernel over the receptive
    /// field — with the weights repacked to row-major [K][Cout] (K = Cin*KH*KW, k = (ic*KH+ky)*KW+kx)
    /// as a new initializer. The repack is a pure permutation (no arithmetic), so it is exact for
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

            // Repack W[oc][ic][ky][kx] -> Wt[(ic*KH+ky)*KW+kx][oc]: a byte-level permutation, exact
            // for any element dtype.
            const HostBuffer &src = g.initializers.at(wId);
            size_t            es  = dtypeSize(wd.dtype);
            int64_t           K   = Cin * KH * KW;
            TensorDesc        td;
            td.name          = g.desc(wId).name + "#gemmw";
            td.shape         = {K, Cout};
            td.dtype         = wd.dtype;
            td.isInitializer = true;
            TensorId   wtId  = g.addTensor(td);
            HostBuffer wt;
            wt.resizeElems(K * Cout, wd.dtype);
            const uint8_t *sp = src.bytes.data();
            uint8_t       *dp = wt.bytes.data();
            for (int64_t oc = 0; oc < Cout; ++oc)
            {
                for (int64_t k = 0; k < K; ++k)
                {
                    std::memcpy(dp + ((size_t) (k * Cout + oc)) * es, sp + ((size_t) (oc * K + k)) * es, es);
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
