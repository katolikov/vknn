// Lower a general grouped Conv (1 < group < Cin) into `group` independent group-1 Convs over
// per-group input-channel slices, joined by a Concat. Each part is a plain dense Conv, so it runs on
// the proven direct/Winograd/ConvGemm GPU kernels; the GPU never needs a dedicated grouped kernel and
// the NC4HW4 packing stays clean (every Slice materialises a contiguous channel slice that repacks
// into whole vec4 blocks). The pure-depthwise case (group == Cin == Cout) already has its own GPU
// kernel and is left untouched here.
#include "passes_internal.h"
#include "vknn/logging.h"
#include <cstring>

namespace vknn {
    namespace {

        Attr intsAttr(std::vector<int64_t> v) {
            Attr a;
            a.kind = Attr::Ints;
            a.ints = std::move(v);
            return a;
        }
        Attr intAttr(int64_t v) {
            Attr a;
            a.kind = Attr::Int;
            a.i    = v;
            return a;
        }

    } // namespace

    void lowerGroupedConv(Graph &g) {
        int lowered = 0;
        // Collect the replacement nodes and append them at the end, then drop the originals: mutating
        // g.nodes while iterating by index would invalidate the loop.
        std::vector<Node> add;
        std::vector<bool> drop(g.nodes.size(), false);

        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &c = g.nodes[i];
            if (c.type != OpType::Conv || c.fusedResidual != kNoTensor)
            {
                continue;
            }
            int64_t group = c.attr.geti("group", 1);
            if (group <= 1)
            {
                continue; // dense conv already runs on the GPU
            }
            TensorId xId = c.inputs.size() > 0 ? c.inputs[0] : kNoTensor;
            TensorId wId = c.inputs.size() > 1 ? c.inputs[1] : kNoTensor;
            if (xId == kNoTensor || wId == kNoTensor || !g.isInitializer(wId))
            {
                continue; // runtime weight: leave the grouped Conv for the CPU op
            }
            // Copy the shapes by value: addTensor() below may reallocate g.tensors and invalidate any
            // reference into it, and these dims are read after the first addTensor.
            const Shape xs = g.desc(xId).shape;
            const Shape ws = g.desc(wId).shape;
            const Shape os = g.desc(c.outputs[0]).shape;
            if (xs.size() != 4 || ws.size() != 4 || os.size() != 4)
            {
                continue; // unresolved / non-2D-spatial shapes stay on the CPU op
            }
            int64_t Cin = xs[1], Cout = ws[0], inCg = ws[1], KH = ws[2], KW = ws[3];
            if (Cin != group * inCg || Cout % group != 0)
            {
                continue; // channels do not partition evenly: keep the exact CPU op
            }
            // Pure depthwise (group == Cin == Cout, inCg == 1) has its own GPU kernel — do not expand it.
            if (group == Cin && group == Cout && inCg == 1)
            {
                continue;
            }
            int64_t outCg = Cout / group;

            // Bias (optional): inputs[2] is a per-output-channel bias unless it is the fused-residual
            // tensor (handled by the fusedResidual guard above, so any inputs[2] here is a real bias).
            bool     hasBias = c.inputs.size() > 2 && c.inputs[2] != kNoTensor;
            TensorId bId     = hasBias ? c.inputs[2] : kNoTensor;
            if (hasBias && !g.isInitializer(bId))
            {
                continue; // runtime bias: leave it for the CPU op
            }

            DType  wdt = g.desc(wId).dtype;
            size_t wes = dtypeSize(wdt);
            DType  bdt = hasBias ? g.desc(bId).dtype : DType::Float32;
            size_t bes = dtypeSize(bdt);
            // Capture the source bytes before addTensor (Graph::tensors may reallocate, invalidating
            // any reference into g.initializers / g.desc).
            std::vector<uint8_t> wsrc      = g.initializers.at(wId).bytes.toVector();
            std::vector<uint8_t> bsrc      = hasBias ? g.initializers.at(bId).bytes.toVector() : std::vector<uint8_t> {};
            std::string          base      = g.desc(c.outputs[0]).name;
            DType                xdt       = g.desc(xId).dtype;
            DType                odt       = g.desc(c.outputs[0]).dtype;
            int64_t              wRowBytes = inCg * KH * KW * (int64_t) wes; // bytes per output channel

            std::vector<TensorId> partOuts;
            for (int64_t gi = 0; gi < group; ++gi)
            {
                // Per-group weight [outCg, inCg, KH, KW]: a contiguous slice of the [Cout, ...] weight.
                TensorDesc wd;
                wd.name          = base + "#g" + std::to_string(gi) + "_w";
                wd.shape         = {outCg, inCg, KH, KW};
                wd.dtype         = wdt;
                wd.isInitializer = true;
                TensorId   wgId  = g.addTensor(wd);
                HostBuffer wgb;
                wgb.resizeElems(outCg * inCg * KH * KW, wdt);
                std::memcpy(wgb.bytes.data(), wsrc.data() + (size_t) (gi * outCg) * wRowBytes, (size_t) outCg * wRowBytes);
                g.initializers[wgId] = std::move(wgb);

                // Per-group bias [outCg], the matching slice of the [Cout] bias.
                TensorId bgId = kNoTensor;
                if (hasBias)
                {
                    TensorDesc bd;
                    bd.name          = base + "#g" + std::to_string(gi) + "_b";
                    bd.shape         = {outCg};
                    bd.dtype         = bdt;
                    bd.isInitializer = true;
                    bgId             = g.addTensor(bd);
                    HostBuffer bgb;
                    bgb.resizeElems(outCg, bdt);
                    std::memcpy(bgb.bytes.data(), bsrc.data() + (size_t) (gi * outCg) * bes, (size_t) outCg * bes);
                    g.initializers[bgId] = std::move(bgb);
                }

                // Slice the input channels [gi*inCg, (gi+1)*inCg) on axis 1 (opset<10 attr form).
                TensorDesc sd;
                sd.name      = base + "#g" + std::to_string(gi) + "_x";
                sd.shape     = {xs[0], inCg, xs[2], xs[3]};
                sd.dtype     = xdt;
                TensorId sId = g.addTensor(sd);
                Node     sl;
                sl.type               = OpType::Slice;
                sl.name               = base + "_g" + std::to_string(gi) + "_slice";
                sl.inputs             = {xId};
                sl.outputs            = {sId};
                sl.attr.map["starts"] = intsAttr({gi * inCg});
                sl.attr.map["ends"]   = intsAttr({(gi + 1) * inCg});
                sl.attr.map["axes"]   = intsAttr({1});
                add.push_back(std::move(sl));

                // Dense (group-1) Conv for this group.
                TensorDesc od;
                od.name      = base + "#g" + std::to_string(gi) + "_y";
                od.shape     = {os[0], outCg, os[2], os[3]};
                od.dtype     = odt;
                TensorId oId = g.addTensor(od);
                Node     cv;
                cv.type              = OpType::Conv;
                cv.name              = base + "_g" + std::to_string(gi) + "_conv";
                cv.inputs            = hasBias ? std::vector<TensorId> {sId, wgId, bgId} : std::vector<TensorId> {sId, wgId};
                cv.outputs           = {oId};
                cv.attr              = c.attr;     // strides/pads/dilations/kernel_shape carry over
                cv.attr.map["group"] = intAttr(1); // each part is a plain dense conv
                // A fused activation is per-output-channel and the groups own disjoint output channels,
                // so applying it inside each part (then concatenating the activated outputs) matches the
                // grouped Conv exactly.
                cv.fusedAct = c.fusedAct;
                cv.actLo    = c.actLo;
                cv.actHi    = c.actHi;
                add.push_back(std::move(cv));
                partOuts.push_back(oId);
            }

            // Concat the per-group outputs along the channel axis back into the original output tensor.
            Node cat;
            cat.type             = OpType::Concat;
            cat.name             = base + "_concat";
            cat.inputs           = partOuts;
            cat.outputs          = {c.outputs[0]};
            cat.attr.map["axis"] = intAttr(1);
            add.push_back(std::move(cat));

            drop[i] = true;
            lowered++;
        }

        if (!lowered)
        {
            return;
        }
        // Rebuild g.nodes: keep the untouched originals (minus the lowered Convs), append the
        // Slice/Conv/Concat replacements, then restore dependency order.
        std::vector<Node> kept;
        kept.reserve(g.nodes.size() - lowered + add.size());
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            if (!drop[i])
            {
                kept.push_back(std::move(g.nodes[i]));
            }
        }
        for (auto &n: add)
        {
            kept.push_back(std::move(n));
        }
        g.nodes = std::move(kept);
        g.topoSort();
        VKNN_INFO << "lowerGroupedConv: lowered " << lowered << " grouped Conv(s) to group-1 Conv + Concat";
    }

} // namespace vknn
