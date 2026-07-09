#include "passes_internal.h"

namespace vknn {

    namespace {

        // Is `t` a scalar (single-element) constant initializer? The scaled-flow multiplier is a
        // graph-wide constant, so only a materialized 1-element payload qualifies. A rank-0 scalar
        // carries numElements()==0 but one payload element, so it counts too.
        static bool isScalarConst(const Graph &g, TensorId t) {
            if (t == kNoTensor || !g.isInitializer(t) || g.initializers.at(t).bytes.empty())
            {
                return false;
            }
            const Shape &s = g.desc(t).shape;
            return numElements(s) == 1 || s.empty();
        }

        // A channels-last coordinate grid constant [.,H,W,2]: rank-4, trailing dim 2, a materialized
        // initializer payload. The op uploads it fp32 (coordinates ride full precision), so its own
        // storage precision is irrelevant here.
        static bool isBaseGridConst(const Graph &g, TensorId t) {
            if (t == kNoTensor || !g.isInitializer(t))
            {
                return false;
            }
            const Shape &s = g.desc(t).shape;
            return s.size() == 4 && s[3] == 2;
        }

        // An Add producing `grid` whose two inputs are a base-grid constant and a runtime coordinate
        // tensor `fnhwc`. Returns the pair (base, fnhwc) via out-params, or false when neither operand
        // ordering matches. ONNX Add imports as OpType::Add; a Binary(subOp==Add) is accepted too.
        static bool matchGridAdd(const Graph &g, const Node &add, TensorId &base, TensorId &fnhwc) {
            bool isAdd = add.type == OpType::Add || (add.type == OpType::Binary && add.subOp == (int) BinaryType::Add);
            if (!isAdd || add.inputs.size() != 2)
            {
                return false;
            }
            for (int b = 0; b < 2; ++b)
            {
                TensorId cand = add.inputs[b], other = add.inputs[1 - b];
                if (isBaseGridConst(g, cand) && other != kNoTensor && !g.isInitializer(other))
                {
                    base  = cand;
                    fnhwc = other;
                    return true;
                }
            }
            return false;
        }

        // A Mul producing `fsc` whose inputs are a full-size flow tensor [N,2,H,W] and a scalar
        // constant scale. Returns (flow, scale). ONNX Mul imports as Binary(subOp==Mul).
        static bool matchScaledFlow(const Graph &g, const Node &mul, TensorId &flow, TensorId &scale) {
            if (mul.type != OpType::Binary || mul.subOp != (int) BinaryType::Mul || mul.inputs.size() != 2)
            {
                return false;
            }
            for (int b = 0; b < 2; ++b)
            {
                TensorId cand = mul.inputs[b], other = mul.inputs[1 - b];
                if (isScalarConst(g, other) && cand != kNoTensor && !g.isInitializer(cand))
                {
                    const Shape &fs = g.desc(cand).shape;
                    if (fs.size() == 4 && fs[1] == 2)
                    {
                        flow  = cand;
                        scale = other;
                        return true;
                    }
                }
            }
            return false;
        }

    } // namespace

    /// Fuse the scaled-flow warp coordinate computation into GridSample so the full-resolution grid
    /// tensor and its NCHW->NHWC Transpose are never materialized. The recognized idiom is
    ///
    ///   Mul(flow, scale) -> Transpose(perm 0,2,3,1) -> Add(base_grid, .) -> GridSample(img, grid)
    ///
    /// where `flow` is a runtime [N,2,H,W] field, `scale` a scalar constant, and `base_grid` a
    /// constant [.,H,W,2] identity grid. The GridSample is rewritten to inputs [img, flow, base] with
    /// attrs warp=1 and warp_scale, and computes its own per-output sample coordinate
    /// `coord = base + scale*flow` inside the sampler's coordinate lookup (see gridsample_warp*.comp /
    /// the CPU oracle). The sampler math (mode / padding / align_corners) is unchanged, so the fused
    /// op is bit-exact with the materialized-grid path: the GPU fp16 variant reproduces the standalone
    /// Mul's fp16 store (an fp32 base add of an fp16-rounded product), and the fp32/CPU paths compute
    /// the coordinate in fp32 exactly as the split Mul+Add did.
    ///
    /// Opportunistic and convex: it fires only on this exact pattern with single-consumer, non-output
    /// intermediates (fsc/fnhwc/grid), leaving every other GridSample — a runtime arbitrary grid, the
    /// common case — untouched. A base grid or scale shared with other nodes stays live; only the
    /// three chain nodes are removed. Runs before fusePointwiseChains (on statically-shaped inputs) so
    /// the coordinate chain is claimed before the general pointwise fusion would host the Add on the
    /// Transpose.
    void fuseGridSampleWarp(Graph &g) {
        std::vector<int>  producer(g.tensors.size(), -1);
        std::vector<int>  consumerCount(g.tensors.size(), 0);
        std::vector<char> isGraphOut(g.tensors.size(), 0);
        for (int ni = 0; ni < (int) g.nodes.size(); ++ni)
        {
            for (TensorId o: g.nodes[ni].outputs)
            {
                if (o != kNoTensor)
                {
                    producer[(size_t) o] = ni;
                }
            }
        }
        for (const auto &nd: g.nodes)
        {
            for (TensorId in: nd.inputs)
            {
                if (in != kNoTensor && in < (TensorId) consumerCount.size())
                {
                    consumerCount[(size_t) in]++;
                }
            }
        }
        for (TensorId go: g.outputs)
        {
            if (go != kNoTensor && go < (TensorId) isGraphOut.size())
            {
                isGraphOut[(size_t) go] = 1;
            }
        }

        // A chain intermediate is fusable only when its producing node is the sole writer, this
        // GridSample is its sole reader, and it is not a graph output (removing its producer must not
        // strand another consumer or an output).
        auto soleInternal = [&](TensorId t) {
            return t != kNoTensor && producer[(size_t) t] >= 0 && consumerCount[(size_t) t] == 1 && !isGraphOut[(size_t) t];
        };

        std::set<int> removed;
        int           fused = 0;
        for (int gi = 0; gi < (int) g.nodes.size(); ++gi)
        {
            Node &gs = g.nodes[gi];
            if (gs.type != OpType::GridSample || gs.inputs.size() != 2 || gs.attr.has("warp"))
            {
                continue;
            }
            TensorId img = gs.inputs[0], grid = gs.inputs[1];
            if (!soleInternal(grid))
            {
                continue;
            }
            int ai = producer[(size_t) grid];
            if (removed.count(ai))
            {
                continue;
            }
            TensorId base = kNoTensor, fnhwc = kNoTensor;
            if (!matchGridAdd(g, g.nodes[ai], base, fnhwc) || !soleInternal(fnhwc))
            {
                continue;
            }
            int ti = producer[(size_t) fnhwc];
            if (removed.count(ti) || g.nodes[ti].type != OpType::Transpose || g.nodes[ti].inputs.size() != 1)
            {
                continue;
            }
            const auto &perm = g.nodes[ti].attr.getints("perm");
            if (perm != std::vector<int64_t> {0, 2, 3, 1})
            {
                continue; // only the NCHW->NHWC channels-last move feeds a flat grid
            }
            TensorId fsc = g.nodes[ti].inputs[0];
            if (!soleInternal(fsc))
            {
                continue;
            }
            int mi = producer[(size_t) fsc];
            if (removed.count(mi))
            {
                continue;
            }
            TensorId flow = kNoTensor, scale = kNoTensor;
            if (!matchScaledFlow(g, g.nodes[mi], flow, scale))
            {
                continue;
            }
            // Geometry: flow [N,2,OH,OW], base [.,OH,OW,2], img rank-4. The base grid's spatial dims
            // must match the flow's so one dispatch lane resolves both the coordinate and the sample.
            const Shape &fs = g.desc(flow).shape, &bs = g.desc(base).shape, &is = g.desc(img).shape;
            if (is.size() != 4 || fs.size() != 4 || bs[1] != fs[2] || bs[2] != fs[3])
            {
                continue;
            }

            float scaleVal = initFloats(g, scale).at(0);
            gs.inputs      = {img, flow, base};
            {
                Attr w;
                w.kind              = Attr::Int;
                w.i                 = 1;
                gs.attr.map["warp"] = w;
            }
            {
                Attr s;
                s.kind                    = Attr::Float;
                s.f                       = scaleVal;
                gs.attr.map["warp_scale"] = s;
            }
            removed.insert(mi);
            removed.insert(ti);
            removed.insert(ai);
            fused++;
        }

        if (fused)
        {
            std::vector<Node> kept;
            kept.reserve(g.nodes.size());
            for (int i = 0; i < (int) g.nodes.size(); ++i)
            {
                if (!removed.count(i))
                {
                    kept.push_back(std::move(g.nodes[i]));
                }
            }
            g.nodes = std::move(kept);
            VKNN_INFO << "fuseGridSampleWarp: fused " << fused << " scaled-flow warp coordinate chain(s) into GridSample";
        }
    }

} // namespace vknn
