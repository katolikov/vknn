#include "passes_internal.h"

namespace vknn {

    void fuseSwish(Graph &g) {
        // Fuse the elementwise self-gating activation Mul(x, HardSigmoid(x)) = HardSwish and
        // Mul(x, Sigmoid(x)) = SiLU/Swish. If x is produced by a Conv/Gemm (consumed only by the gate +
        // the Mul), fold it into that op's epilogue activation (removes 2 dispatches + the intermediate);
        // otherwise collapse the [gate, Mul] pair into a single unary op. MobileNetV3 has ~21 of these.
        std::vector<int> producer(g.tensors.size(), -1);
        std::vector<int> consumers(g.tensors.size(), 0);
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            for (TensorId o: g.nodes[i].outputs)
            {
                if (o != kNoTensor)
                {
                    producer[o] = (int) i;
                }
            }
            for (TensorId in: g.nodes[i].inputs)
            {
                if (in != kNoTensor)
                {
                    consumers[in]++;
                }
            }
        }
        std::set<int> remove;
        int           fusedC = 0, fusedU = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &M = g.nodes[i];
            if (M.type != OpType::Binary || (BinaryType) M.subOp != BinaryType::Mul || M.inputs.size() != 2)
            {
                continue;
            }
            int      sigIdx = -1;
            TensorId x      = kNoTensor;
            for (int e = 0; e < 2; ++e)
            {
                int pc = producer[M.inputs[e]];
                if (pc < 0 || g.nodes[pc].type != OpType::Unary)
                {
                    continue;
                }
                UnaryType su = (UnaryType) g.nodes[pc].subOp;
                if ((su == UnaryType::HardSigmoid || su == UnaryType::Sigmoid) && g.nodes[pc].inputs[0] == M.inputs[1 - e])
                {
                    sigIdx = pc;
                    x      = M.inputs[1 - e];
                    break;
                }
            }
            if (sigIdx < 0)
            {
                continue;
            }
            if (consumers[g.nodes[sigIdx].outputs[0]] != 1)
            {
                continue; // gate feeds only this Mul
            }
            ActType act = (UnaryType) g.nodes[sigIdx].subOp == UnaryType::HardSigmoid ? ActType::HardSwish : ActType::SiLU;
            int     px  = producer[x];
            bool fuseConv = px >= 0 && (g.nodes[px].type == OpType::Conv || g.nodes[px].type == OpType::Gemm) && g.nodes[px].fusedAct == ActType::None && consumers[x] == 2;
            if (fuseConv)
            {
                g.nodes[px].fusedAct = act;
                // Fold the Mul output onto the conv output. Must also patch any fusedResidual edge that
                // pointed at the Mul output (a residual already folded into a later conv by fuseResidualAdd),
                // else that conv reads a dead tensor -> crash.
                rewireTensor(g, M.outputs[0], x);
                remove.insert((int) i);
                remove.insert(sigIdx);
                fusedC++;
            } else
            {
                // collapse [gate, Mul] -> one unary
                M.type   = OpType::Unary;
                M.subOp  = (int) (act == ActType::HardSwish ? UnaryType::HardSwish : UnaryType::SiLU);
                M.inputs = {x};
                remove.insert(sigIdx);
                fusedU++;
            }
        }
        if (!remove.empty())
        {
            std::vector<Node> kept;
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!remove.count((int) i))
                {
                    kept.push_back(g.nodes[i]);
                }
            }
            g.nodes = std::move(kept);
            VKNN_INFO << "fuseSwish: fused " << fusedC << " into conv, collapsed " << fusedU << " to unary";
        }
    }


} // namespace vknn
