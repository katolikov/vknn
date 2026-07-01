#include "passes_internal.h"

namespace vknn {

    // Fuse  out = Add(MatMul(A, W), bias)  into the MatMul epilogue (out[...,n] = matmul + bias[n]), so
    // the Linear's bias is added in the fp32 accumulator and the result is stored once — removing the
    // Add's whole-tensor read + write and one fp16 requantization per Linear. The transformer's qkv /
    // proj / fc1 / fc2 layers are all this shape. Eligible only when: the MatMul output feeds only this
    // Add, the other Add operand is a rank-1 [N] (or [1,..,1,N]) initializer matching the output's last
    // dim, and the MatMul is the standard 2-D-operand case (the kernel indexes bias by output column).
    void fuseMatMulBias(Graph &g) {
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
        for (TensorId go: g.outputs)
        {
            if (go != kNoTensor)
            {
                consumers[go]++;
            }
        }
        std::set<int> remove;
        int           fused = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &add = g.nodes[i];
            if (add.type != OpType::Add || add.inputs.size() != 2 || add.fusedAct != ActType::None)
            {
                continue;
            }
            // One operand is a MatMul output (single-consumer), the other a rank-1 [N] bias initializer.
            int mi = -1, mmIdx = -1;
            for (int e = 0; e < 2; ++e)
            {
                int p = producer[add.inputs[e]];
                if (p >= 0 && g.nodes[p].type == OpType::MatMul && g.nodes[p].fusedBias == kNoTensor && consumers[add.inputs[e]] == 1)
                {
                    mmIdx = p;
                    mi    = e;
                    break;
                }
            }
            if (mmIdx < 0)
            {
                continue;
            }
            TensorId biasId = add.inputs[1 - mi];
            if (!g.isInitializer(biasId))
            {
                continue;
            }
            Node        &mm  = g.nodes[mmIdx];
            const Shape &os  = g.desc(mm.outputs[0]).shape;
            const Shape &bs  = g.desc(biasId).shape;
            const Shape &as0 = g.desc(mm.inputs[0]).shape;
            const Shape &as1 = g.desc(mm.inputs[1]).shape;
            // Standard 2-D-operand MatMul (no 1-D promotion); bias is a per-output-column vector [.,N].
            if (os.empty() || as0.size() < 2 || as1.size() < 2)
            {
                continue;
            }
            int64_t N = os.back();
            if (bs.empty() || bs.back() != N || numElements(bs) != N)
            {
                continue;
            }
            mm.fusedBias = biasId;
            mm.inputs.push_back(biasId);    // keep the bias live for DCE / buffer allocation / scheduling
            mm.outputs[0] = add.outputs[0]; // MatMul now produces the (biased) Add output, name intact
            remove.insert((int) i);
            ++fused;
        }
        if (fused)
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
            VKNN_INFO << "fuseMatMulBias: fused " << fused << " bias Add(s) into MatMul";
        }
    }


} // namespace vknn
