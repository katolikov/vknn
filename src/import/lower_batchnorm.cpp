#include "passes_internal.h"
#include <cmath>

namespace vknn {

    /// Lower every inference-mode BatchNorm that survived foldBatchNorm (its data input is not a
    /// foldable Conv — DenseNet-style pre-activation BN, BN after Concat, BN on a shared Conv
    /// output) into the per-channel affine it is at inference: y = x * s + b with
    /// s = scale/sqrt(var+eps) and b = B - mean*s, computed on the host in double and stored as two
    /// [1,C,1..] fp32 initializers. The Mul/Add pair is per-element, so the pointwise fusion pass
    /// can fold it — together with a trailing activation — into the producer's epilogue or one
    /// fused kernel, where the opaque BatchNorm op never fused. Runs unconditionally: the fused ==
    /// unfused gate compares two compiles of the SAME lowered graph, so the lowering itself never
    /// enters that comparison.
    ///
    /// Eligible only when all four parameters are 1-D initializers of matching length C, the data
    /// input's shape is resolved with dim 1 == C, and only the first output is consumed (training
    /// outputs disqualify). A non-matching BatchNorm keeps its opaque op.
    void lowerBatchNorm(Graph &g) {
        int               lowered = 0;
        std::vector<Node> appended;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &bn = g.nodes[i];
            if (bn.type != OpType::BatchNorm || bn.inputs.size() < 5)
            {
                continue;
            }
            bool extraOutUsed = false;
            for (size_t o = 1; o < bn.outputs.size(); ++o)
            {
                if (bn.outputs[o] != kNoTensor)
                {
                    extraOutUsed = true;
                }
            }
            if (extraOutUsed)
            {
                continue;
            }
            TensorId x = bn.inputs[0], sc = bn.inputs[1], bi = bn.inputs[2], mn = bn.inputs[3], vr = bn.inputs[4];
            bool     paramsOk = true;
            for (TensorId p: {sc, bi, mn, vr})
            {
                if (!g.isInitializer(p) || g.desc(p).dtype != DType::Float32)
                {
                    paramsOk = false; // fp16-stored params keep the opaque op (same rule as subpixelConvTranspose)
                }
            }
            if (!paramsOk)
            {
                continue;
            }
            const Shape &xs = g.desc(x).shape;
            if (xs.size() < 2)
            {
                continue; // unresolved or channel-less data input
            }
            int64_t     C     = xs[1];
            const auto &scale = g.initializers[sc].f32();
            const auto &beta  = g.initializers[bi].f32();
            const auto &mean  = g.initializers[mn].f32();
            const auto &var   = g.initializers[vr].f32();
            if (numElements(g.desc(sc).shape) != C || numElements(g.desc(bi).shape) != C || numElements(g.desc(mn).shape) != C || numElements(g.desc(vr).shape) != C)
            {
                continue;
            }
            double eps = bn.attr.getf("epsilon", 1e-5f);

            std::vector<float> s((size_t) C), b((size_t) C);
            for (int64_t c = 0; c < C; ++c)
            {
                double a      = (double) scale[c] / std::sqrt((double) var[c] + eps);
                s[(size_t) c] = (float) a;
                b[(size_t) c] = (float) ((double) beta[c] - (double) mean[c] * a);
            }

            // [1,C,1,..] to the data input's rank: rank-4 data gets the NC4-friendly [1,C,1,1]
            // channel-broadcast class; other ranks broadcast by right-alignment the same way.
            Shape pshape(xs.size(), 1);
            pshape[1]     = C;
            auto addParam = [&](const char *suffix, const std::vector<float> &v) -> TensorId {
                TensorDesc d;
                d.name          = bn.name + suffix;
                d.shape         = pshape;
                d.dtype         = DType::Float32;
                d.isInitializer = true;
                TensorId   id   = g.addTensor(d);
                HostBuffer hb;
                hb.resizeElems(v.size(), DType::Float32);
                for (size_t k = 0; k < v.size(); ++k)
                {
                    hb.f32()[k] = v[k];
                }
                g.initializers[id] = std::move(hb);
                return id;
            };
            TensorId sId = addParam("#bn_scale", s);
            TensorId bId = addParam("#bn_shift", b);

            TensorDesc mi;
            mi.name      = bn.name + "#bn_mul";
            mi.shape     = g.desc(bn.outputs[0]).shape;
            mi.dtype     = g.desc(bn.outputs[0]).dtype;
            TensorId mid = g.addTensor(mi);

            Node add;
            add.type    = OpType::Add;
            add.name    = bn.name + "#bn_add";
            add.inputs  = {mid, bId};
            add.outputs = {bn.outputs[0]}; // consumers keep reading the BatchNorm's tensor id

            Node mul;
            mul.type    = OpType::Binary;
            mul.subOp   = (int) BinaryType::Mul;
            mul.name    = bn.name + "#bn_mulop";
            mul.inputs  = {x, sId};
            mul.outputs = {mid};

            g.nodes[i] = mul; // replace the BatchNorm in place; the Add is appended and topo-sorted below
            appended.push_back(add);
            lowered++;
        }
        if (lowered)
        {
            for (auto &n: appended)
            {
                g.nodes.push_back(n);
            }
            g.topoSort();
            VKNN_INFO << "lowerBatchNorm: lowered " << lowered << " BatchNorm(s) to per-channel Mul+Add";
        }
    }

} // namespace vknn
