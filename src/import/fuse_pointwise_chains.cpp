#include "passes_internal.h"

namespace vknn {

    // Is this tensor forced to fp32 storage by the selective-fp32 preset (Precision::Normal), or
    // already marked storeFp32 by an earlier pass run? A fused kernel stores its whole chain in one
    // dtype, so fusing across such a tensor would round an intermediate that must stay fp32 to fp16 —
    // breaking bit-exactness. Matches by name substring (mixedPrecisionFp32Tensors()), the same rule
    // markFp32 applies at load; a no-op for models without those names (e.g. CNNs).
    static bool pwTensorIsFp32(const Graph &g, TensorId t) {
        if (t == kNoTensor)
        {
            return false;
        }
        if (g.desc(t).storeFp32)
        {
            return true;
        }
        static const std::string marks = mixedPrecisionFp32Tensors();
        const std::string       &nm    = g.desc(t).name;
        if (nm.empty() || marks.empty())
        {
            return false;
        }
        size_t start = 0;
        while (start < marks.size())
        {
            size_t      comma = marks.find(',', start);
            std::string sub   = marks.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            if (!sub.empty() && nm.find(sub) != std::string::npos)
            {
                return true;
            }
            if (comma == std::string::npos)
            {
                break;
            }
            start = comma + 1;
        }
        return false;
    }

    // Per-element ops eligible to join a fused-pointwise chain.
    static bool pwEligible(const Node &n) {
        switch (n.type)
        {
            case OpType::Binary:
            case OpType::Add:
            case OpType::Unary:
            case OpType::Clip:
            case OpType::Relu:
                return true;
            default:
                return false;
        }
    }

    // Encode node `n` as one pw_steps entry, given the chain's running value flows in as `chainVal`.
    // Sets `operand` to the second (non-chain) tensor for a BINARY step, kNoTensor otherwise. Returns
    // false when `n` can't be encoded as a step from this chain position (e.g. a non-commutative binary
    // op with the chain value on the wrong side).
    static bool pwEncodeStep(const Graph &g, const Node &n, const Shape &run, TensorId chainVal, int &kind, int &code, int &bcast, float &p0, float &p1, TensorId &operand) {
        p0 = p1      = 0;
        operand      = kNoTensor;
        auto bcastOf = [&](const Shape &s) -> int {
            if (s == run)
            {
                return 0;
            }
            if (s.size() == 4 && run.size() == 4 && s[0] == run[0] && s[1] == run[1] && s[2] == 1 && s[3] == 1)
            {
                return 1;
            }
            return 2;
        };
        auto pickBinary = [&](int codeIn, bool commutative) -> bool {
            TensorId a = n.inputs[0], b = n.inputs[1];
            if (a == chainVal)
            {
                operand = b;
                code    = codeIn;
                kind    = 0;
                bcast   = bcastOf(g.desc(b).shape);
                return true;
            }
            if (b == chainVal && commutative)
            {
                operand = a;
                code    = codeIn;
                kind    = 0;
                bcast   = bcastOf(g.desc(a).shape);
                return true;
            }
            return false; // non-commutative with chainVal as inputs[1] -> not fusable at this position
        };
        switch (n.type)
        {
            case OpType::Add:
                return pickBinary((int) BinaryType::Add, true);
            case OpType::Binary: {
                BinaryType bt   = (BinaryType) n.subOp;
                bool       comm = (bt == BinaryType::Mul || bt == BinaryType::Max || bt == BinaryType::Min);
                return pickBinary((int) bt, comm);
            }
            case OpType::Unary:
                kind  = 1;
                code  = n.subOp;
                bcast = 0;
                p0    = n.actLo;
                p1    = n.actHi;
                return true;
            case OpType::Relu:
                kind  = 2;
                code  = (int) ActType::Relu;
                bcast = 0;
                return true;
            case OpType::Clip: {
                kind  = 2;
                code  = (int) ActType::Clip;
                bcast = 0;
                p0    = -3.4e38f;
                p1    = 3.4e38f;
                if (n.inputs.size() > 1 && n.inputs[1] != kNoTensor && g.isInitializer(n.inputs[1]))
                {
                    p0 = g.initializers.at(n.inputs[1]).f32()[0];
                }
                if (n.inputs.size() > 2 && n.inputs[2] != kNoTensor && g.isInitializer(n.inputs[2]))
                {
                    p1 = g.initializers.at(n.inputs[2]).f32()[0];
                }
                if (n.attr.has("min"))
                {
                    p0 = n.attr.getf("min", p0);
                }
                if (n.attr.has("max"))
                {
                    p1 = n.attr.getf("max", p1);
                }
                return true;
            }
            default:
                return false;
        }
    }

    // Merge a maximal single-consumer per-element chain (Binary/Add/Unary/Relu/Clip, same output shape,
    // same GPU layout, no fp32-forced intermediate) into one standalone FusedPointwise node: inputs[0]
    // is the chain's primary (head) input, inputs[1..] are the extra step operands in encounter order,
    // outputs[0] reuses the chain tail's tensor id. Chains shorter than 2 nodes are left alone (nothing
    // to gain by wrapping a single op). Producer-attach (folding a chain into a producer's own epilogue
    // instead of emitting a standalone node) is a separate, later pass.
    // The chain's primary is the full-size runtime stream the kernel reads element-for-element: a
    // non-initializer input whose shape equals the chain output. A constant/broadcast input can only be
    // a step operand (uploaded), never the primary (the GPU op reads the primary from an activation
    // buffer; a constant has none). Returns kNoTensor when no input qualifies (then this op is not a
    // fusable chain head).
    static TensorId pwHeadPrimary(const Graph &g, const Node &n, const Shape &run) {
        auto usable = [&](TensorId t) {
            return t != kNoTensor && !g.isInitializer(t) && g.desc(t).shape == run;
        };
        if (n.type == OpType::Unary || n.type == OpType::Relu || n.type == OpType::Clip)
        {
            return (!n.inputs.empty() && usable(n.inputs[0])) ? n.inputs[0] : kNoTensor;
        }
        if ((n.type == OpType::Binary || n.type == OpType::Add) && n.inputs.size() >= 2)
        {
            if (usable(n.inputs[0]))
            {
                return n.inputs[0];
            }
            if (usable(n.inputs[1]))
            {
                return n.inputs[1];
            }
        }
        return kNoTensor;
    }

    void fusePointwiseChains(Graph &g) {
        std::vector<int> producer(g.tensors.size(), -1), consumers(g.tensors.size(), 0), nextOf(g.tensors.size(), -1);
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            for (TensorId o: g.nodes[i].outputs)
            {
                if (o != kNoTensor)
                {
                    producer[o] = (int) i;
                }
            }
        }
        for (size_t j = 0; j < g.nodes.size(); ++j)
        {
            for (TensorId in: g.nodes[j].inputs)
            {
                if (in != kNoTensor && in < (TensorId) consumers.size())
                {
                    consumers[in]++;
                    nextOf[in] = (int) j;
                }
            }
        }
        for (TensorId go: g.outputs)
        {
            if (go != kNoTensor)
            {
                consumers[go]++;
            }
        }
        auto single = [&](TensorId t) {
            return t != kNoTensor && consumers[t] == 1 && !pwTensorIsFp32(g, t);
        };

        std::set<int> removed;
        int           fused = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            if (removed.count((int) i) || !pwEligible(g.nodes[i]) || g.nodes[i].inputs.empty())
            {
                continue;
            }
            if (g.nodes[i].outputs[0] == kNoTensor)
            {
                continue;
            }
            Shape    run  = g.desc(g.nodes[i].outputs[0]).shape;
            if (run.empty())
            {
                continue; // unresolved/dynamic shape: fusing would bake an empty shape (see runStandardPasses)
            }
            if (g.desc(g.nodes[i].outputs[0]).dtype == DType::Int64)
            {
                continue; // shape-arithmetic chain (int64), not a float activation stream
            }
            TensorId prim = pwHeadPrimary(g, g.nodes[i], run);
            if (prim == kNoTensor)
            {
                continue; // no full-size runtime input to stream through the kernel
            }
            int pp = (prim >= 0 && prim < (TensorId) producer.size()) ? producer[prim] : -1;
            if (pp >= 0 && pwEligible(g.nodes[pp]) && single(prim) && !removed.count(pp))
            {
                continue; // not the chain head: an earlier iteration starting at pp will extend through here
            }
            if (pwTensorIsFp32(g, prim))
            {
                continue;
            }
            bool wantFlat = gpuFlatNode(g, g.nodes[i]);
            if (wantFlat && (int) run.size() > kPwMaxRank)
            {
                continue; // the flat kernel only stores kPwMaxRank broadcast dims
            }

            std::vector<int64_t>  steps;
            std::vector<float>    params;
            std::vector<TensorId> inputs {prim};
            std::vector<int>      chain;
            int                   cur      = (int) i;
            TensorId              chainVal = prim;
            while (true)
            {
                Node &nd = g.nodes[cur];
                if (!pwEligible(nd) || removed.count(cur))
                {
                    break;
                }
                if (gpuFlatNode(g, nd) != wantFlat)
                {
                    break;
                }
                if (g.desc(nd.outputs[0]).shape != run)
                {
                    break;
                }
                if (pwTensorIsFp32(g, nd.outputs[0]))
                {
                    break; // fp32-intermediate guard
                }
                if ((int) steps.size() / 4 >= kPwMaxSteps)
                {
                    break;
                }
                int      kind, code, bcast;
                float    p0, p1;
                TensorId operand;
                if (!pwEncodeStep(g, nd, run, chainVal, kind, code, bcast, p0, p1, operand))
                {
                    break;
                }
                int oi = -1;
                if (operand != kNoTensor)
                {
                    if ((int) inputs.size() - 1 >= kPwMaxOperands)
                    {
                        break;
                    }
                    if (pwTensorIsFp32(g, operand))
                    {
                        break;
                    }
                    inputs.push_back(operand);
                    oi = (int) inputs.size() - 1;
                }
                steps.insert(steps.end(), {(int64_t) kind, (int64_t) code, (int64_t) oi, (int64_t) bcast});
                params.insert(params.end(), {p0, p1});

                // A Binary/Add node may itself carry a fused activation epilogue (folded in by
                // fuseActivations, which runs before this pass and only ever targets Conv/Gemm/Add
                // producers) -- that epilogue is invisible as a separate node, so encode it as one more
                // step here or it would be silently dropped when the node is replaced.
                if (nd.fusedAct != ActType::None)
                {
                    if ((int) steps.size() / 4 >= kPwMaxSteps)
                    {
                        break; // no room for the epilogue step: leave this node unfused (chain ends before it)
                    }
                    steps.insert(steps.end(), {2, (int64_t) nd.fusedAct, -1, 0});
                    params.insert(params.end(), {nd.actLo, nd.actHi});
                }
                chain.push_back(cur);

                TensorId outT = nd.outputs[0];
                if (!single(outT))
                {
                    break;
                }
                int nxt = nextOf[outT];
                if (nxt < 0 || removed.count(nxt) || g.nodes[nxt].inputs.empty())
                {
                    break;
                }
                chainVal = outT;
                cur      = nxt;
            }
            if (chain.size() < 2)
            {
                continue; // nothing to fuse: a lone op gains nothing from the wrapper node
            }
            int      tail    = chain.back();
            TensorId tailOut = g.nodes[tail].outputs[0];
            Node     fn;
            fn.type    = OpType::FusedPointwise;
            fn.name    = g.nodes[chain.front()].name + "#pwchain";
            fn.inputs  = inputs;
            fn.outputs = {tailOut};
            {
                Attr a;
                a.kind                  = Attr::Ints;
                a.ints                  = steps;
                fn.attr.map["pw_steps"] = a;
            }
            {
                Attr a;
                a.kind                   = Attr::Floats;
                a.floats                 = params;
                fn.attr.map["pw_params"] = a;
            }
            {
                Attr a;
                a.kind                 = Attr::Int;
                a.i                    = wantFlat ? 1 : 0;
                fn.attr.map["pw_flat"] = a;
            }
            g.nodes[chain.front()] = fn;
            for (size_t kk = 1; kk < chain.size(); ++kk)
            {
                removed.insert(chain[kk]);
            }
            fused++;
        }
        if (fused)
        {
            std::vector<Node> kept;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!removed.count((int) i))
                {
                    kept.push_back(g.nodes[i]);
                }
            }
            g.nodes = std::move(kept);
            VKNN_INFO << "fusePointwiseChains: fused " << fused << " chain(s)";
        }
    }

} // namespace vknn
