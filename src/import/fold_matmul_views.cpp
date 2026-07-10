#include "core/matmul_tile.h"
#include "core/matmul_view.h"
#include "core/quant_int4.h"
#include "passes_internal.h"
#include <cstdint>
#include <deque>
#include <optional>

namespace vknn {
    namespace {

        // A strided view of a source buffer, one axis at a time. Each axis is an outer->inner list
        // of (size, stride) blocks; a multi-block axis is one the source cannot express with a
        // single stride (a GQA reshape merging a real head dim with a broadcast repeat dim keeps
        // both blocks, e.g. 14 heads = [(2, 65600), (7, 0)]). Stride 0 is a broadcast block.
        struct SubDim {
            int64_t size;
            int64_t stride;
        };
        using ViewAxis = std::vector<SubDim>;
        using View     = std::vector<ViewAxis>;

        // Merge adjacent blocks that are jointly dense (outer stride == inner size * inner stride,
        // which also collapses adjacent broadcast blocks) and drop size-1 blocks, so single-stride
        // axes end up as exactly one block. An axis of total size 1 canonicalizes to {1, 0}.
        ViewAxis canonicalAxis(const ViewAxis &axis) {
            ViewAxis out;
            for (const SubDim &s: axis)
            {
                if (s.size == 1)
                {
                    continue;
                }
                if (!out.empty() && out.back().stride == s.size * s.stride)
                {
                    out.back() = {out.back().size * s.size, s.stride};
                } else
                {
                    out.push_back(s);
                }
            }
            if (out.empty())
            {
                out.push_back({1, 0});
            }
            return out;
        }

        // Dense row-major view of a shape.
        View denseView(const Shape &shape) {
            View    v(shape.size());
            int64_t stride = 1;
            for (int i = (int) shape.size() - 1; i >= 0; --i)
            {
                v[i] = {{shape[i], shape[i] == 1 ? 0 : stride}};
                stride *= shape[i];
            }
            return v;
        }

        // Row-major reshape of a strided view to `target`. Blocks split at any divisor; a target
        // dim boundary that falls inside a block at a non-divisible point is not expressible and
        // yields nullopt. Merges never force a single stride — the multi-block axis carries them.
        std::optional<View> reshapeView(const View &view, const Shape &target) {
            std::deque<SubDim> flat;
            for (const ViewAxis &axis: view)
            {
                for (const SubDim &s: axis)
                {
                    if (s.size != 1)
                    {
                        flat.push_back(s);
                    }
                }
            }
            View out;
            for (int64_t t: target)
            {
                if (t <= 0)
                {
                    return std::nullopt;
                }
                ViewAxis axis;
                int64_t  need = t;
                while (need > 1)
                {
                    if (flat.empty())
                    {
                        return std::nullopt;
                    }
                    SubDim s = flat.front();
                    if (s.size <= need)
                    {
                        if (need % s.size != 0)
                        {
                            return std::nullopt;
                        }
                        axis.push_back(s);
                        need /= s.size;
                        flat.pop_front();
                    } else
                    {
                        if (s.size % need != 0)
                        {
                            return std::nullopt;
                        }
                        // Consume the outer `need` of the block; the inner remainder stays queued.
                        axis.push_back({need, s.stride * (s.size / need)});
                        flat.front() = {s.size / need, s.stride};
                        need         = 1;
                    }
                }
                out.push_back(canonicalAxis(axis));
            }
            return flat.empty() ? std::optional<View>(out) : std::nullopt;
        }

        // NumPy-broadcast expand of a view to `target` (rank may grow on the left). A broadcast
        // axis becomes a single zero-stride block; a non-1 axis must match the target extent.
        std::optional<View> expandView(const View &view, const Shape &target) {
            if (target.size() < view.size())
            {
                return std::nullopt;
            }
            View padded(target.size() - view.size(), ViewAxis {{1, 0}});
            padded.insert(padded.end(), view.begin(), view.end());
            for (size_t i = 0; i < target.size(); ++i)
            {
                int64_t cur = 1;
                for (const SubDim &s: padded[i])
                {
                    cur *= s.size;
                }
                if (cur == target[i])
                {
                    continue;
                }
                if (cur != 1)
                {
                    return std::nullopt;
                }
                padded[i] = {{target[i], 0}};
            }
            return padded;
        }

        // Apply one chain op to the view, using the node's OUTPUT shape as the target (static
        // post-bucket shapes make attr parsing unnecessary for Reshape/Unsqueeze/Squeeze/Expand).
        std::optional<View> applyChainOp(const Graph &g, const Node &n, const View &view) {
            const Shape &outShape = g.desc(n.outputs[0]).shape;
            switch (n.type)
            {
                case OpType::Transpose: {
                    std::vector<int64_t> perm = n.attr.getints("perm");
                    if (perm.empty())
                    {
                        for (int64_t i = (int64_t) view.size() - 1; i >= 0; --i)
                        {
                            perm.push_back(i);
                        }
                    }
                    if (perm.size() != view.size())
                    {
                        return std::nullopt;
                    }
                    View out(view.size());
                    for (size_t i = 0; i < perm.size(); ++i)
                    {
                        if (perm[i] < 0 || perm[i] >= (int64_t) view.size())
                        {
                            return std::nullopt;
                        }
                        out[i] = view[perm[i]];
                    }
                    return out;
                }
                case OpType::Reshape:
                case OpType::Unsqueeze:
                case OpType::Squeeze:
                    return reshapeView(view, outShape);
                case OpType::Expand:
                    return expandView(view, outShape);
                default:
                    return std::nullopt;
            }
        }

        // The ops a view can absorb. Reshape/Unsqueeze/Squeeze are index bookkeeping; Transpose and
        // Expand are the copies the fold exists to remove.
        bool chainOp(OpType t) {
            return t == OpType::Transpose || t == OpType::Reshape || t == OpType::Unsqueeze || t == OpType::Squeeze || t == OpType::Expand;
        }
        // A chain node must be PURE data movement. Transpose/Slice/Concat can host a fused pointwise
        // epilogue (pw_steps) or a fused activation, which computes values on the way through —
        // folding such a node away would drop that computation (a K-side attention pre-scale fused
        // onto the K^T transpose changes every score by the scale factor).
        bool bareChainNode(const Node &n) {
            return !n.attr.has("pw_steps") && n.fusedAct == ActType::None && n.fusedBias == kNoTensor && n.fusedResidual == kNoTensor && n.outputs.size() == 1;
        }
        bool moverOp(OpType t) {
            return t == OpType::Transpose || t == OpType::Expand;
        }

        struct OperandView {
            View     view;   // per semantic operand axis, over the source buffer
            TensorId source; // buffer the strides address (the operand itself when no chain)
            bool     folded; // true when a Transpose/Expand chain was absorbed
        };

        // Walk the producer chain of `operand` upward through view-expressible ops and compose the
        // strided view over the chain's source. A chain without a Transpose/Expand is not worth
        // claiming (the planner aliases pure reshapes already); an inexpressible step, an
        // ineligible producer, or a tensor the later markFp32 pass will pin (`fp32Pins`) keeps the
        // operand as-is with its dense view — removing a pinned tensor would remove the fp32 store
        // the pin exists for.
        OperandView resolveOperand(const Graph &g, const std::vector<int> &producer, TensorId operand, const std::string &fp32Pins) {
            std::vector<int> chain;
            TensorId         src   = operand;
            bool             mover = false;
            while (chain.size() < 16)
            {
                int p = src >= 0 && src < (int) producer.size() ? producer[src] : -1;
                if (p < 0 || !chainOp(g.nodes[p].type) || g.nodes[p].inputs.empty() || !bareChainNode(g.nodes[p]))
                {
                    break;
                }
                if (!fp32Pins.empty() && fp32NameMatch(g.desc(src).name, fp32Pins))
                {
                    break;
                }
                chain.push_back(p);
                mover = mover || moverOp(g.nodes[p].type);
                src   = g.nodes[p].inputs[0];
            }
            if (chain.empty() || !mover)
            {
                return {denseView(g.desc(operand).shape), operand, false};
            }
            View view = denseView(g.desc(src).shape);
            for (auto it = chain.rbegin(); it != chain.rend(); ++it)
            {
                auto next = applyChainOp(g, g.nodes[*it], view);
                if (!next)
                {
                    return {denseView(g.desc(operand).shape), operand, false};
                }
                view = std::move(*next);
            }
            return {std::move(view), src, true};
        }

        // Common refinement of two block partitions of the same axis extent: split both at the
        // union of their block boundaries. Returns (size, aStride, bStride) triples outer->inner,
        // or nullopt when a boundary falls at a non-divisible point of the other side's block.
        struct RefinedBlock {
            int64_t size, aStride, bStride;
        };
        std::optional<std::vector<RefinedBlock>> refineAxis(const ViewAxis &a, const ViewAxis &b) {
            std::deque<SubDim>        pa(a.begin(), a.end()), pb(b.begin(), b.end());
            std::vector<RefinedBlock> out;
            while (!pa.empty() && !pb.empty())
            {
                SubDim &x = pa.front(), &y = pb.front();
                if (x.size == 1 && y.size == 1)
                {
                    if (out.empty())
                    {
                        out.push_back({1, 0, 0});
                    }
                    pa.pop_front();
                    pb.pop_front();
                } else if (x.size == y.size)
                {
                    out.push_back({x.size, x.stride, y.stride});
                    pa.pop_front();
                    pb.pop_front();
                } else if (x.size > y.size)
                {
                    if (x.size % y.size != 0)
                    {
                        return std::nullopt;
                    }
                    out.push_back({y.size, x.stride * (x.size / y.size), y.stride});
                    x = {x.size / y.size, x.stride};
                    pb.pop_front();
                } else
                {
                    if (y.size % x.size != 0)
                    {
                        return std::nullopt;
                    }
                    out.push_back({x.size, x.stride, y.stride * (y.size / x.size)});
                    y = {y.size / x.size, y.stride};
                    pa.pop_front();
                }
            }
            if (!pa.empty() || !pb.empty())
            {
                return std::nullopt;
            }
            return out;
        }

        // A view axis usable as a matrix axis (m/n/k) must reduce to one stride.
        std::optional<int64_t> singleStride(const ViewAxis &axis) {
            ViewAxis c = canonicalAxis(axis);
            if (c.size() != 1)
            {
                return std::nullopt;
            }
            return c[0].stride;
        }

        constexpr int64_t kStrideMax = INT32_MAX;

    } // namespace

    // Fold Transpose/Expand/Reshape/Unsqueeze/Squeeze chains feeding a MatMul operand into
    // per-axis stride metadata on the MatMul node (core/matmul_view.h), rewiring the operand to the
    // chain's source so the chain's copies never run. A GQA repeat_kv (Expand broadcasting kv heads
    // to query heads, then transposes into QK/PV orientation) folds to a zero-stride block on the
    // split head axis, so decode attention reads the KV cache in place. Only shapes outside the
    // tiled-GEMM class fold (the tiled kernels assume dense row-major panels; decode attention is
    // mat-vec shaped, and prefill-sized matmuls keep their materialized chain). Values and the
    // ascending-k accumulation order are unchanged, so a folded graph's outputs are bit-identical.
    // Runs at session load only — never at compile — so .vxm bytes never carry view attrs.
    void foldMatMulViews(Graph &g, const std::string &fp32Pins) {
        std::vector<int> producer(g.tensors.size(), -1);
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            for (TensorId t: g.nodes[i].outputs)
            {
                if (t >= 0 && t < (int) producer.size())
                {
                    producer[t] = (int) i;
                }
            }
        }

        int folded = 0;
        for (Node &node: g.nodes)
        {
            if (node.type != OpType::MatMul || node.attr.has(kWq) || node.attr.has(kMmView))
            {
                continue;
            }
            if (node.inputs.size() < 2 || node.outputs.empty())
            {
                continue;
            }
            const Shape &sa  = g.desc(node.inputs[0]).shape;
            const Shape &sb  = g.desc(node.inputs[1]).shape;
            const Shape &out = g.desc(node.outputs[0]).shape;
            if (sa.size() < 2 || sb.size() < 2 || out.size() != std::max(sa.size(), sb.size()))
            {
                continue; // 1-D promotion cases keep the materialized chain
            }
            auto staticShape = [](const Shape &s) {
                for (int64_t d: s)
                {
                    if (d <= 0)
                    {
                        return false;
                    }
                }
                return !s.empty();
            };
            if (!staticShape(sa) || !staticShape(sb) || !staticShape(out))
            {
                continue;
            }
            const int64_t M = sa[sa.size() - 2], K = sa[sa.size() - 1], N = sb[sb.size() - 1];
            if (M >= kTiledMatMulMin && N >= kTiledMatMulMin && K >= kTiledMatMulMin)
            {
                continue; // tiled-GEMM class: dense panels required, chain stays
            }

            OperandView a = resolveOperand(g, producer, node.inputs[0], fp32Pins);
            OperandView b = resolveOperand(g, producer, node.inputs[1], fp32Pins);
            if (!a.folded && !b.folded)
            {
                continue;
            }

            // Left-pad each operand's view to the output rank, then map every axis onto the
            // output's coordinate space: batch axes align, A contributes (M, K) as its last two
            // axes, B contributes (K, N).
            const int rank = (int) out.size(), batchRank = rank - 2;
            auto      padTo = [&](const View &v) {
                View p((size_t) rank - v.size(), ViewAxis {{1, 0}});
                p.insert(p.end(), v.begin(), v.end());
                return p;
            };
            View va = padTo(a.view), vb = padTo(b.view);

            // Matrix-axis strides: each must be a single stride (the k step is one push constant;
            // m/n are one geometry entry each).
            auto aM = singleStride(va[rank - 2]), aK = singleStride(va[rank - 1]);
            auto bK = singleStride(vb[rank - 2]), bN = singleStride(vb[rank - 1]);
            if (!aM || !aK || !bK || !bN)
            {
                continue;
            }

            // Batch axes: an operand axis either matches the output extent or broadcasts (size 1).
            auto batchAxis = [&](const View &v, int i, int64_t want) -> std::optional<ViewAxis> {
                int64_t total = 1;
                for (const SubDim &s: v[i])
                {
                    total *= s.size;
                }
                if (total == want)
                {
                    return canonicalAxis(v[i]);
                }
                if (total == 1)
                {
                    return ViewAxis {{want, 0}};
                }
                return std::nullopt;
            };

            std::vector<int64_t> dims, aStride, bStride;
            bool                 ok = true;
            for (int i = 0; ok && i < batchRank; ++i)
            {
                auto axA = batchAxis(va, i, out[i]);
                auto axB = batchAxis(vb, i, out[i]);
                if (!axA || !axB)
                {
                    ok = false;
                    break;
                }
                auto refined = refineAxis(*axA, *axB);
                if (!refined)
                {
                    ok = false;
                    break;
                }
                for (const RefinedBlock &r: *refined)
                {
                    dims.push_back(r.size);
                    aStride.push_back(r.aStride);
                    bStride.push_back(r.bStride);
                }
            }
            if (!ok)
            {
                continue;
            }
            dims.push_back(M);
            aStride.push_back(*aM);
            bStride.push_back(0);
            dims.push_back(N);
            aStride.push_back(0);
            bStride.push_back(*bN);

            // Kernel-class parity: a mat-vec-range shape (core/matmul_tile.h gemv predicate) keeps
            // its materialized chain unless the view preserves gemv4's B contract (n contiguous,
            // every other B offset term kGemvVec-aligned). Without this, a device whose workgroup
            // limit admits gemv4 but not the scalar gemv would demote a folded matmul to the naive
            // kernel — a different accumulation grouping, so different bits than the fold-off run.
            if (M == 1 && K >= kGemvMinK && N < kGemvMaxN && N % kGemvVec == 0)
            {
                bool b4 = *bN == 1 && *bK % kGemvVec == 0;
                for (size_t i = 0; b4 && i + 2 < bStride.size(); ++i)
                {
                    b4 = bStride[i] % kGemvVec == 0;
                }
                if (!b4)
                {
                    continue;
                }
            }

            int64_t total = 1, want = 1;
            for (int64_t d: dims)
            {
                total *= d;
            }
            for (int64_t d: out)
            {
                want *= d;
            }
            auto fits = [](const std::vector<int64_t> &v) {
                for (int64_t x: v)
                {
                    if (x < 0 || x > kStrideMax)
                    {
                        return false;
                    }
                }
                return true;
            };
            if (total != want || !fits(dims) || !fits(aStride) || !fits(bStride) || *aK > kStrideMax || *bK > kStrideMax)
            {
                continue;
            }

            node.inputs[0] = a.source;
            node.inputs[1] = b.source;
            auto setInt    = [&](const char *k, int64_t v) {
                Attr at;
                at.kind          = Attr::Int;
                at.i             = v;
                node.attr.map[k] = at;
            };
            auto setInts = [&](const char *k, std::vector<int64_t> v) {
                Attr at;
                at.kind          = Attr::Ints;
                at.ints          = std::move(v);
                node.attr.map[k] = at;
            };
            setInt(kMmView, kMmViewVersion);
            setInts(kMmViewDims, std::move(dims));
            setInts(kMmViewAStride, std::move(aStride));
            setInts(kMmViewBStride, std::move(bStride));
            setInt(kMmViewAK, *aK);
            setInt(kMmViewBK, *bK);
            setInt(kMmViewM, M);
            setInt(kMmViewN, N);
            setInt(kMmViewK, K);
            ++folded;
        }

        if (folded)
        {
            eliminateDeadNodes(g);
            VKNN_INFO << "foldMatMulViews: folded operand chains into " << folded << " MatMul view(s)";
        }
    }

} // namespace vknn
