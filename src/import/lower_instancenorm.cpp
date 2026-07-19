#include "passes_internal.h"

namespace vknn {

    /// Lower every ONNX InstanceNormalization into ops both backends already implement:
    /// y = scale * (x - mean) / sqrt(var + eps) + B, with mean/var taken per (n, c) over the
    /// spatial dims. The emitted sequence is spatial ReduceMean(keepdims) -> Sub -> Mul(squared
    /// diff) -> ReduceMean -> Add(eps) -> Sqrt -> Div -> per-channel Mul(scale) -> Add(B). On a
    /// rank-4 input the two ReduceMeans recover as GlobalAvgPool (lowerReduceToGap runs next), the
    /// centered Sub/Div broadcast [N,C,1,1] against [N,C,H,W] on the packed Binary kernel, and the
    /// pointwise tail is fodder for fusePointwiseChains — so no InstanceNorm kernel exists on any
    /// backend. scale/B are copied into [1,C,1,..] fp32 initializers of the input's rank, the same
    /// channel-broadcast class lowerBatchNorm emits.
    ///
    /// Eligible only when scale and B are 1-D fp32 initializers of length C and the data input's
    /// shape is resolved with rank >= 3 ([N,C,spatial...]). Anything else keeps the opaque op with
    /// a WARN and fails backend planning as unsupported — parameters are never fabricated. Needs
    /// resolved input shapes, so it runs after the const-fold/infer fixpoint (an
    /// InstanceNormalization behind an Upsample resolves only once the folded scales chain fixes
    /// the spatial dims).
    void lowerInstanceNorm(Graph &g) {
        int               lowered = 0;
        std::vector<Node> appended;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            if (g.nodes[i].type != OpType::InstanceNorm)
            {
                continue;
            }
            // Copies, not references: the node is overwritten and addTensor may reallocate descs.
            const std::string name = g.nodes[i].name;
            if (g.nodes[i].inputs.size() < 3 || g.nodes[i].outputs.empty())
            {
                VKNN_WARN << "lowerInstanceNorm: " << name << " keeps its opaque op (missing scale/B input); no backend implements it";
                continue;
            }
            TensorId x = g.nodes[i].inputs[0], sc = g.nodes[i].inputs[1], bi = g.nodes[i].inputs[2];
            TensorId out = g.nodes[i].outputs[0];
            bool     paramsOk = true;
            for (TensorId p: {sc, bi})
            {
                if (p == kNoTensor || !g.isInitializer(p) || g.desc(p).dtype != DType::Float32)
                {
                    paramsOk = false; // runtime or fp16-stored params keep the opaque op (same rule as lowerBatchNorm)
                }
            }
            if (!paramsOk)
            {
                VKNN_WARN << "lowerInstanceNorm: " << name << " keeps its opaque op (scale/B are not fp32 initializers); no backend implements it";
                continue;
            }
            const Shape xs = g.desc(x).shape;
            if (xs.size() < 3)
            {
                VKNN_WARN << "lowerInstanceNorm: " << name << " keeps its opaque op (input rank " << xs.size() << " unresolved or < 3); no backend implements it";
                continue;
            }
            int64_t C = xs[1];
            if (numElements(g.desc(sc).shape) != C || numElements(g.desc(bi).shape) != C)
            {
                VKNN_WARN << "lowerInstanceNorm: " << name << " keeps its opaque op (scale/B length != C); no backend implements it";
                continue;
            }
            size_t rank = xs.size();
            float  eps  = g.nodes[i].attr.getf("epsilon", 1e-5f);
            DType  odt  = g.desc(out).dtype;

            // Copy the parameter payloads before any addTensor mutates the graph. dtype-safe reads: a
            // native-fp64 scale/bias decodes to fp32 here (the lowered params are fp32 -- exotic case).
            std::vector<float> scaleV = initFloats(g, sc);
            std::vector<float> biasV  = initFloats(g, bi);

            // Stats shape: the input with every spatial dim reduced to 1 (keepdims), the
            // [N,C,1,..] channel-broadcast class the packed Binary kernel handles natively.
            Shape ms = xs;
            for (size_t d = 2; d < rank; ++d)
            {
                ms[d] = 1;
            }
            Shape pshape(rank, 1); // [1,C,1,..]: scale/B broadcast form, as lowerBatchNorm emits
            pshape[1] = C;

            auto addParam = [&](const char *suffix, const Shape &sh, const std::vector<float> &v) -> TensorId {
                TensorDesc d;
                d.name          = name + suffix;
                d.shape         = sh;
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
            TensorId sId = addParam("#in_scale", pshape, scaleV);
            TensorId bId = addParam("#in_bias", pshape, biasV);
            TensorId eId = addParam("#in_eps", Shape(rank, 1), {eps});

            // Intermediates carry their exact shapes: everything is derivable from `xs` here, and
            // stamping them keeps the sequence resolved without another inferShapes round.
            auto addAct = [&](const char *suffix, const Shape &sh) -> TensorId {
                TensorDesc d;
                d.name  = name + suffix;
                d.shape = sh;
                d.dtype = odt;
                return g.addTensor(d);
            };
            TensorId mean   = addAct("#in_mean", ms);
            TensorId xc     = addAct("#in_center", xs);
            TensorId sq     = addAct("#in_sq", xs);
            TensorId varT   = addAct("#in_var", ms);
            TensorId veps   = addAct("#in_vareps", ms);
            TensorId stdev  = addAct("#in_std", ms);
            TensorId norm   = addAct("#in_norm", xs);
            TensorId scaled = addAct("#in_scaled", xs);

            Attr axes;
            axes.kind = Attr::Ints;
            for (size_t d = 2; d < rank; ++d)
            {
                axes.ints.push_back((int64_t) d);
            }
            Attr keep;
            keep.kind = Attr::Int;
            keep.i    = 1;
            auto reduceMean = [&](const char *suffix, TensorId in, TensorId o) {
                Node r;
                r.type                 = OpType::Reduce;
                r.subOp                = (int) ReduceType::Mean;
                r.name                 = name + suffix;
                r.inputs               = {in};
                r.outputs              = {o};
                r.attr.map["axes"]     = axes;
                r.attr.map["keepdims"] = keep;
                return r;
            };
            auto binary = [&](const char *suffix, BinaryType bt, TensorId a, TensorId b, TensorId o) {
                Node n;
                n.type    = OpType::Binary;
                n.subOp   = (int) bt;
                n.name    = name + suffix;
                n.inputs  = {a, b};
                n.outputs = {o};
                return n;
            };
            auto addOp = [&](const char *suffix, TensorId a, TensorId b, TensorId o) {
                Node n;
                n.type    = OpType::Add;
                n.name    = name + suffix;
                n.inputs  = {a, b};
                n.outputs = {o};
                return n;
            };
            Node sqrtOp;
            sqrtOp.type    = OpType::Unary;
            sqrtOp.subOp   = (int) UnaryType::Sqrt;
            sqrtOp.name    = name + "#in_sqrt";
            sqrtOp.inputs  = {veps};
            sqrtOp.outputs = {stdev};

            // Replace the InstanceNorm in place with the first op; the rest are appended and
            // topo-sorted below. The bias Add writes the original output tensor, so consumers keep
            // reading the InstanceNormalization's tensor id.
            g.nodes[i] = reduceMean("#in_meanop", x, mean);
            appended.push_back(binary("#in_subop", BinaryType::Sub, x, mean, xc));
            appended.push_back(binary("#in_sqop", BinaryType::Mul, xc, xc, sq));
            appended.push_back(reduceMean("#in_varop", sq, varT));
            appended.push_back(addOp("#in_epsop", varT, eId, veps));
            appended.push_back(sqrtOp);
            appended.push_back(binary("#in_divop", BinaryType::Div, xc, stdev, norm));
            appended.push_back(binary("#in_scaleop", BinaryType::Mul, norm, sId, scaled));
            appended.push_back(addOp("#in_biasop", scaled, bId, out));
            lowered++;
        }
        if (lowered)
        {
            for (auto &n: appended)
            {
                g.nodes.push_back(n);
            }
            g.topoSort();
            VKNN_INFO << "lowerInstanceNorm: lowered " << lowered << " InstanceNormalization(s) to per-channel normalize ops";
        }
    }

} // namespace vknn
