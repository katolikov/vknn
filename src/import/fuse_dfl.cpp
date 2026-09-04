#include "core/dfl.h"
#include "passes_internal.h"

namespace vknn {
    namespace {
        bool isTranspose(const Node &n, const std::vector<int64_t> &perm) {
            return n.type == OpType::Transpose && n.attr.getints("perm") == perm;
        }
        // The Softmax must run over the last axis of its rank-4 input (the bins after the first
        // transpose), whichever way the exporter spelled the axis.
        bool isLastAxisSoftmax(const Graph &g, const Node &n) {
            if (n.type != OpType::Softmax || n.inputs.empty())
            {
                return false;
            }
            const int64_t rank = (int64_t) g.desc(n.inputs[0]).shape.size();
            int64_t       axis = n.attr.geti("axis", -1);
            if (axis < 0)
            {
                axis += rank;
            }
            return rank == 4 && axis == rank - 1;
        }
        // The bin-weight conv: [1][B][1][1] weights, ungrouped, unpadded, unstrided, no bias.
        bool isBinConv(const Graph &g, const Node &conv, int64_t &bins) {
            if (conv.type != OpType::Conv || conv.inputs.size() < 2 || conv.inputs[1] == kNoTensor || !g.isInitializer(conv.inputs[1]))
            {
                return false;
            }
            if (conv.inputs.size() > 2 && conv.inputs[2] != kNoTensor)
            {
                return false;
            }
            const Shape &w = g.desc(conv.inputs[1]).shape;
            if (w.size() != 4 || w[0] != 1 || w[2] != 1 || w[3] != 1 || w[1] < kDflMinBins || conv.attr.geti("group", 1) != 1 || conv.fusedAct != ActType::None)
            {
                return false;
            }
            for (int64_t p: conv.attr.getints("pads"))
            {
                if (p != 0)
                {
                    return false;
                }
            }
            for (int64_t s: conv.attr.getints("strides"))
            {
                if (s != 1)
                {
                    return false;
                }
            }
            bins = w[1];
            return true;
        }
    } // namespace

    void fuseDfl(Graph &g) {
        // Collapse the detection head's distribution-focal decode (core/dfl.h)
        //     Reshape [N,S*B,L] -> [N,S,B,L] -> Transpose(0,3,1,2) -> Softmax(last axis)
        //     -> Transpose(0,3,2,1) -> Conv1x1(B -> 1, bin weights) -> Reshape [N,1,S,L] -> [N,S,L]
        // into ONE FusedDfl node over the [N,S*B,L] map. The match anchors on the bin conv and walks
        // producers upward, then requires the conv's sole reader to be the closing Reshape.
        //
        // Precondition: every intermediate is consumed only inside the chain (removing it orphans no
        // live tensor), the shapes are resolved (inferShapes has run) and agree with the [N,S,B,L]
        // view, and the conv has no bias. A chain failing any of these is left unfused.
        // Postcondition: the closing Reshape's node slot holds the FusedDfl node (preserving its
        // output id for the decode that follows); the five nodes above it are gone.
        std::vector<int> producer(g.tensors.size(), -1);
        std::vector<int> consumers(g.tensors.size(), 0);
        std::vector<int> soleReader(g.tensors.size(), -1);
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
                    soleReader[in] = (int) i;
                }
            }
        }
        std::set<TensorId> graphOutputs(g.outputs.begin(), g.outputs.end());
        auto               single = [&](TensorId t) {
            return t != kNoTensor && consumers[t] == 1 && !graphOutputs.count(t);
        };
        std::set<int> remove;
        int           fused = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            const Node &conv = g.nodes[i];
            int64_t     bins = 0;
            if (!isBinConv(g, conv, bins) || conv.outputs.empty() || !single(conv.outputs[0]))
            {
                continue;
            }
            const int t1 = producer[conv.inputs[0]];
            if (t1 < 0 || !isTranspose(g.nodes[t1], dflBinsFirstPerm()) || !single(conv.inputs[0]) || g.nodes[t1].inputs.empty())
            {
                continue;
            }
            const int sm = producer[g.nodes[t1].inputs[0]];
            if (sm < 0 || !isLastAxisSoftmax(g, g.nodes[sm]) || !single(g.nodes[t1].inputs[0]))
            {
                continue;
            }
            const int t0 = producer[g.nodes[sm].inputs[0]];
            if (t0 < 0 || !isTranspose(g.nodes[t0], dflBinsLastPerm()) || !single(g.nodes[sm].inputs[0]) || g.nodes[t0].inputs.empty())
            {
                continue;
            }
            const int rs = producer[g.nodes[t0].inputs[0]];
            if (rs < 0 || g.nodes[rs].type != OpType::Reshape || !single(g.nodes[t0].inputs[0]) || g.nodes[rs].inputs.empty())
            {
                continue;
            }
            const TensorId x      = g.nodes[rs].inputs[0];
            const Shape   &xShape = g.desc(x).shape;                 // [N, S*B, L]
            const Shape   &view   = g.desc(g.nodes[rs].outputs[0]).shape; // [N, S, B, L]
            if (xShape.size() != 3 || view.size() != 4 || view[0] != xShape[0] || view[2] != bins || view[3] != xShape[2] || view[1] * bins != xShape[1] || view[1] < 1)
            {
                continue;
            }
            const int close = soleReader[conv.outputs[0]];
            if (close < 0 || g.nodes[close].type != OpType::Reshape || g.nodes[close].inputs.empty() || g.nodes[close].inputs[0] != conv.outputs[0] || g.nodes[close].outputs.empty())
            {
                continue;
            }
            const Shape &out = g.desc(g.nodes[close].outputs[0]).shape; // [N, S, L]
            if (out.size() != 3 || out[0] != xShape[0] || out[1] != view[1] || out[2] != xShape[2])
            {
                continue;
            }
            const int chain[] = {rs, t0, sm, t1, (int) i, close};
            bool      overlap = false;
            for (int n: chain)
            {
                overlap = overlap || remove.count(n);
            }
            if (overlap)
            {
                continue;
            }
            Node dfl;
            dfl.type    = OpType::FusedDfl;
            dfl.name    = conv.name + "#dfl";
            dfl.inputs  = {x, conv.inputs[1]};
            dfl.outputs = {g.nodes[close].outputs[0]};
            Attr binsAttr;
            binsAttr.kind        = Attr::Int;
            binsAttr.i           = bins;
            dfl.attr.map["bins"] = binsAttr;
            g.nodes[close]       = dfl; // the closing Reshape's slot becomes the fused node
            for (int n: chain)
            {
                if (n != close)
                {
                    remove.insert(n);
                }
            }
            fused++;
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
            VKNN_INFO << "fuseDfl: fused " << fused << " distribution-focal decode chain(s)";
        }
    }
} // namespace vknn
