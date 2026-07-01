#include "passes_internal.h"

namespace vknn {

    // Lower the two batched-matmul Einsum equations to Unsqueeze + MatMul (+ Squeeze) so they run on
    // the validated flat MatMul GPU kernel instead of the CPU einsum. The remaining "i,j->ij" outer
    // product keeps its own GPU kernel. Run after shapes are resolved (the einsum operand shapes must
    // be known).
    void lowerEinsum(Graph &g) {
        auto axesAttr = [](std::vector<int64_t> ax) {
            Attr a;
            a.kind = Attr::Ints;
            a.ints = std::move(ax);
            return a;
        };
        std::vector<Node> added;
        std::vector<int>  remove;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &n = g.nodes[i];
            if (n.type != OpType::Einsum || n.inputs.size() < 2)
            {
                continue;
            }
            std::string eq;
            for (char c: n.attr.gets("equation", ""))
            {
                if (c != ' ' && c != '\t')
                {
                    eq += c;
                }
            }
            TensorId A = n.inputs[0], B = n.inputs[1], out = n.outputs[0];
            // Copy shapes/descs/names by value up front: g.addTensor() below reallocates g.tensors, which
            // would dangle any reference held into it.
            if (eq == "...ab,...b->...a")
            {
                // y[...,a] = sum_b A[...,a,b]*x[...,b]  ==  Squeeze(MatMul(A, Unsqueeze(x,-1)), -1)
                Shape xs = g.desc(B).shape;
                if (xs.empty())
                {
                    continue;
                }
                int64_t    xrank    = (int64_t) xs.size();
                Shape      outShape = g.desc(out).shape;
                int64_t    outRank  = (int64_t) outShape.size();
                TensorDesc dxp      = g.desc(B);
                dxp.name            = dxp.name + "#e_unsq";
                dxp.isInitializer = dxp.isInput = dxp.isOutput = false;
                dxp.shape                                      = xs;
                dxp.shape.push_back(1);
                TensorId   xp     = g.addTensor(dxp);
                TensorDesc dmm    = g.desc(out);
                dmm.name          = dmm.name + "#e_mm";
                dmm.isInitializer = dmm.isInput = dmm.isOutput = false;
                dmm.shape                                      = outShape;
                dmm.shape.push_back(1);
                TensorId mm = g.addTensor(dmm);
                Node     un;
                un.type             = OpType::Unsqueeze;
                un.name             = n.name + "#unsq";
                un.inputs           = {B};
                un.outputs          = {xp};
                un.attr.map["axes"] = axesAttr({xrank});
                Node mul;
                mul.type    = OpType::MatMul;
                mul.name    = n.name + "#mm";
                mul.inputs  = {A, xp};
                mul.outputs = {mm};
                Node sq;
                sq.type             = OpType::Squeeze;
                sq.name             = n.name + "#sq";
                sq.inputs           = {mm};
                sq.outputs          = {out};
                sq.attr.map["axes"] = axesAttr({outRank});
                added.push_back(un);
                added.push_back(mul);
                added.push_back(sq);
                remove.push_back((int) i);
            } else if (eq == "bij,bnjk->bnik")
            {
                // C[b,n,i,k] = sum_j A[b,i,j]*B[b,n,j,k]  ==  MatMul(Unsqueeze(A,1), B)  (A broadcasts over
                // n)
                Shape as = g.desc(A).shape;
                if (as.empty())
                {
                    continue;
                }
                TensorDesc dap    = g.desc(A);
                dap.name          = dap.name + "#e_unsq";
                dap.isInitializer = dap.isInput = dap.isOutput = false;
                dap.shape                                      = as;
                dap.shape.insert(dap.shape.begin() + 1, 1);
                TensorId ap = g.addTensor(dap);
                Node     un;
                un.type             = OpType::Unsqueeze;
                un.name             = n.name + "#unsq";
                un.inputs           = {A};
                un.outputs          = {ap};
                un.attr.map["axes"] = axesAttr({1});
                Node mul;
                mul.type    = OpType::MatMul;
                mul.name    = n.name + "#mm";
                mul.inputs  = {ap, B};
                mul.outputs = {out};
                added.push_back(un);
                added.push_back(mul);
                remove.push_back((int) i);
            }
        }
        if (!added.empty())
        {
            std::set<int>     rm(remove.begin(), remove.end());
            std::vector<Node> kept;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!rm.count((int) i))
                {
                    kept.push_back(std::move(g.nodes[i]));
                }
            }
            for (auto &a: added)
            {
                kept.push_back(std::move(a));
            }
            g.nodes = std::move(kept);
            g.topoSort();
            VKNN_INFO << "lowerEinsum: lowered " << remove.size() << " batched einsum(s) to MatMul";
        }
    }


} // namespace vknn
