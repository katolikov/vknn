// fuseRope: collapse each rotate-half RoPE chain into ONE OpType::Rope node at session load.
//
// The pattern is the primitive expansion lowerOrtContribOps emits for a contrib RotaryEmbedding
// (expandRotary in lower_ort_contrib.cpp is the authoritative definition), as it survives the
// standard compile passes:
//
//   r4  = Reshape(x)                       [.., H, head]        (stays: the planner aliases it)
//   x1  = Slice(r4, last axis, [0, half))                       half = head / 2
//   x2  = Slice(r4, last axis, [half, head))
//   cosB = Unsqueeze(Gather(cos_table[P, half], pos, axis 0))   [.., 1, half]
//   sinB = Unsqueeze(Gather(sin_table[P, half], pos, axis 0))   [.., 1, half]
//   o1  = x1*cosB - x2*sinB      o2 = x1*sinB + x2*cosB         Binary nodes, or the two
//                                                               FusedPointwise units the pointwise
//                                                               fusion built from them
//   cat = Concat(o1, o2, last axis)                             [.., H, head]
//   y   = Reshape(cat)                                          (stays: aliased)
//
// The Concat node is rewritten in place to Rope(r4, pos, cos_table, sin_table) with a `half`
// attribute; the slices, gathers, unsqueezes and rotate products go dead and eliminateDeadNodes
// removes them (liveness follows fused edges, so an intermediate referenced by another node's
// fusedResidual/fusedBias stays). The outer reshapes are pure metadata the buffer planner aliases,
// so they cost no dispatch and are left untouched. Per site this replaces ~7 dispatches (2 slices,
// 2 gathers, 2 rotate units, 1 concat) with one.
//
// Recognition is structural only (op kinds, shapes, slice bounds, the rotate expression) — never by
// node or tensor name. The rotate products are matched through a small symbolic evaluation: each
// candidate value is expanded into a sum of degree<=2 monomials over its leaf tensors, walking bare
// Binary Mul/Sub/Add nodes and interpreting a FusedPointwise unit's pw_steps; o1 must equal exactly
// sA*rA - sB*rB and o2 exactly sA*rB + sB*rA over the same four leaves. Any node carrying work the
// fusion does not reproduce — a pw epilogue on a structural node, a fused activation, a fused
// bias/residual edge, extra outputs, or an unrecognized attribute — refuses the site (the
// bareChainNode discipline from fold_matmul_views.cpp, extended to a per-op attribute allowlist).
//
// Numerics: the fused kernel computes both output halves in fp32 from the stored inputs and rounds
// once at the store. The decomposed chain stores the sliced halves and gathered rows as fp16
// intermediates; removing those round-trips is the accepted numeric change for this op class (the
// table rows and slice inputs are already fp16-valued, so the difference is confined to dropping
// the rotate products' intermediate stores).
//
// Runs at load only (Hint::RopeFusion), before foldMatMulViews — the two passes claim disjoint node
// kinds (the view fold absorbs Transpose/Expand movers, never Slice/Gather/Concat), and a MatMul
// chain walk that previously stopped at the Concat output stops at the same tensor now produced by
// the Rope node. Never serialized: a .vxm's bytes are unchanged and old files fuse on load.
#include "passes_internal.h"
#include <map>

namespace vknn {
    namespace {

        // A value as a sum of monomials over leaf tensors: key = the (sorted) factor pair of a
        // degree-2 product or (t, kNoTensor) for a linear term, value = the integer coefficient.
        using Monomial = std::pair<TensorId, TensorId>;
        using Expr     = std::map<Monomial, int>;

        Monomial mono(TensorId a, TensorId b) {
            return a <= b ? Monomial {a, b} : Monomial {b, a};
        }

        Expr leafExpr(TensorId t) {
            return {{mono(t, kNoTensor), 1}};
        }

        bool addExpr(Expr &into, const Expr &e, int sign) {
            for (const auto &kv: e)
            {
                int &c = into[kv.first];
                c += sign * kv.second;
                if (c == 0)
                {
                    into.erase(kv.first);
                }
            }
            return into.size() <= 4; // the rotate expression never exceeds two products
        }

        // Product of two expressions; fails on any degree-3 term (a linear factor is (t, kNoTensor)).
        bool mulExpr(const Expr &a, const Expr &b, Expr &out) {
            out.clear();
            for (const auto &ka: a)
            {
                for (const auto &kb: b)
                {
                    if (ka.first.first != kNoTensor || kb.first.first != kNoTensor)
                    {
                        return false; // both factors must be linear
                    }
                    out[mono(ka.first.second, kb.first.second)] += ka.second * kb.second;
                }
            }
            return out.size() <= 4;
        }

        // A node is bare when it carries no fused work and no attribute outside `allowed`: the
        // fusion reproduces only the op's own arithmetic, so an epilogue, activation, fused edge,
        // extra output, or unknown attribute on a chain node refuses the site.
        bool bareNode(const Node &n, std::initializer_list<const char *> allowed) {
            if (n.fusedAct != ActType::None || n.fusedBias != kNoTensor || n.fusedResidual != kNoTensor || n.outputs.size() != 1)
            {
                return false;
            }
            for (const auto &kv: n.attr.map)
            {
                bool ok = false;
                for (const char *k: allowed)
                {
                    if (kv.first == k)
                    {
                        ok = true;
                        break;
                    }
                }
                if (!ok)
                {
                    return false;
                }
            }
            return true;
        }

        // Interpret a FusedPointwise unit's pw_steps symbolically over its input tensors. Only the
        // Mul/Sub/Add binary steps and Load steps the rotate units use are evaluated; any other
        // kind/code (an activation, a comparison, a select) fails. Register/accumulator semantics
        // mirror the CPU VM (backend/cpu/ops/fused_pointwise.cpp): srcA/srcB name the accumulator
        // (kPwRefAcc), the entry value (kPwRefEntry = inputs[0]), a register (kPwRefReg0 - r), or an
        // operand (kPwRefOp0 - i = inputs[i]); dst >= 0 additionally copies the result to that
        // register.
        bool evalPwUnit(const Node &n, Expr &out) {
            const std::vector<int64_t> &steps = n.attr.getints("pw_steps");
            if (steps.empty() || steps.size() % 8 != 0)
            {
                return false;
            }
            for (float p: n.attr.getfloats("pw_params"))
            {
                if (p != 0.0f)
                {
                    return false; // the rotate steps carry no parameters
                }
            }
            Expr acc = leafExpr(n.inputs[0]);
            Expr reg[kPwMaxRegs];
            bool regSet[kPwMaxRegs] = {};
            auto resolve = [&](int64_t ref, Expr &e) {
                if (ref == kPwRefAcc)
                {
                    e = acc;
                    return true;
                }
                if (ref == kPwRefEntry)
                {
                    e = leafExpr(n.inputs[0]);
                    return true;
                }
                if (ref <= kPwRefReg0 && ref > kPwRefReg0 - kPwMaxRegs)
                {
                    int r = (int) (kPwRefReg0 - ref);
                    if (!regSet[r])
                    {
                        return false;
                    }
                    e = reg[r];
                    return true;
                }
                if (ref <= kPwRefOp0)
                {
                    size_t i = (size_t) (kPwRefOp0 - ref);
                    if (i >= n.inputs.size() || n.inputs[i] == kNoTensor)
                    {
                        return false;
                    }
                    e = leafExpr(n.inputs[i]);
                    return true;
                }
                return false;
            };
            for (size_t s = 0; s < steps.size() / 8; ++s)
            {
                const int64_t kind = steps[s * 8], code = steps[s * 8 + 1];
                const int64_t srcA = steps[s * 8 + 2], srcB = steps[s * 8 + 3];
                const int64_t dst  = steps[s * 8 + 5];
                Expr          a, b, result;
                if (kind == kPwKindLoad)
                {
                    if (!resolve(srcA, result))
                    {
                        return false;
                    }
                } else if (kind == kPwKindBinary && (code == (int64_t) BinaryType::Mul || code == (int64_t) BinaryType::Sub || code == (int64_t) BinaryType::Add))
                {
                    if (!resolve(srcA, a) || !resolve(srcB, b))
                    {
                        return false;
                    }
                    if (code == (int64_t) BinaryType::Mul)
                    {
                        if (!mulExpr(a, b, result))
                        {
                            return false;
                        }
                    } else
                    {
                        result = a;
                        if (!addExpr(result, b, code == (int64_t) BinaryType::Sub ? -1 : 1))
                        {
                            return false;
                        }
                    }
                } else
                {
                    return false; // any other step kind computes something the Rope kernel does not
                }
                acc = result;
                if (dst >= 0 && dst < kPwMaxRegs)
                {
                    reg[dst]    = acc;
                    regSet[dst] = true;
                }
            }
            out = std::move(acc);
            return true;
        }

        // Expand `t` into an Expr by walking bare Binary Mul/Sub/Add producers and FusedPointwise
        // rotate units; anything else is a leaf. `visited` bounds the walk (the rotate expression is
        // three ops deep). Nodes consumed by the expansion are recorded in `absorbed` so the caller
        // can apply the fp32-pin refusal to every tensor whose store the fusion removes.
        bool exprOf(const Graph &g, const std::vector<int> &producer, TensorId t, Expr &out, std::vector<TensorId> &absorbed, int depth) {
            const int p = t >= 0 && t < (int) producer.size() ? producer[t] : -1;
            if (p < 0 || depth > 4)
            {
                out = leafExpr(t);
                return true;
            }
            const Node &nd = g.nodes[p];
            if (nd.type == OpType::FusedPointwise)
            {
                if (!bareNode(nd, {"pw_steps", "pw_params", "pw_flat", "pw_relax"}) || nd.inputs.empty() || !evalPwUnit(nd, out))
                {
                    out = leafExpr(t);
                    return true;
                }
                absorbed.push_back(t);
                return true;
            }
            const bool isAdd = nd.type == OpType::Add;
            const bool isBin = nd.type == OpType::Binary && (nd.subOp == (int32_t) BinaryType::Mul || nd.subOp == (int32_t) BinaryType::Sub || nd.subOp == (int32_t) BinaryType::Add);
            if ((!isAdd && !isBin) || nd.inputs.size() != 2 || !bareNode(nd, {}))
            {
                out = leafExpr(t);
                return true;
            }
            Expr a, b;
            if (!exprOf(g, producer, nd.inputs[0], a, absorbed, depth + 1) || !exprOf(g, producer, nd.inputs[1], b, absorbed, depth + 1))
            {
                return false;
            }
            const int32_t op = isAdd ? (int32_t) BinaryType::Add : nd.subOp;
            bool          ok;
            if (op == (int32_t) BinaryType::Mul)
            {
                ok = mulExpr(a, b, out);
            } else
            {
                out = a;
                ok  = addExpr(out, b, op == (int32_t) BinaryType::Sub ? -1 : 1);
            }
            if (!ok)
            {
                out = leafExpr(t); // not the rotate shape; treat the value as opaque
                return true;
            }
            absorbed.push_back(t);
            return true;
        }

        // Walk a broadcast operand up through bare metadata reshapes (Unsqueeze/Reshape/Squeeze —
        // flat element order is unchanged) to the Gather that reads the table row. Returns the
        // gather node index or -1; intermediate tensors are appended to `absorbed`.
        int gatherBehind(const Graph &g, const std::vector<int> &producer, TensorId t, std::vector<TensorId> &absorbed) {
            for (int hop = 0; hop < 8; ++hop)
            {
                const int p = t >= 0 && t < (int) producer.size() ? producer[t] : -1;
                if (p < 0)
                {
                    return -1;
                }
                const Node &nd = g.nodes[p];
                if (nd.type == OpType::Gather)
                {
                    if (!bareNode(nd, {"axis", "idx_scalar"}) || nd.inputs.size() < 2)
                    {
                        return -1;
                    }
                    absorbed.push_back(t);
                    return p;
                }
                const bool metadataReshape = nd.type == OpType::Unsqueeze || nd.type == OpType::Reshape || nd.type == OpType::Squeeze;
                if (!metadataReshape || !bareNode(nd, {"axes"}) || nd.inputs.empty())
                {
                    return -1;
                }
                absorbed.push_back(t);
                t = nd.inputs[0];
            }
            return -1;
        }

        // Static, fully-known shape.
        bool staticShape(const Shape &s) {
            if (s.empty())
            {
                return false;
            }
            for (int64_t d: s)
            {
                if (d <= 0)
                {
                    return false;
                }
            }
            return true;
        }

        // The Slice over the last axis of `r4` covering [lo, lo+half). Bounds come from attributes
        // or initializer inputs (readI64Param); the output shape is the authority for the width.
        bool sliceOfHalf(const Graph &g, const Node &nd, TensorId r4, int64_t lo, int64_t half) {
            if (nd.type != OpType::Slice || !bareNode(nd, {"starts", "ends", "axes", "steps"}) || nd.inputs.empty() || nd.inputs[0] != r4)
            {
                return false;
            }
            const Shape &in  = g.desc(nd.inputs[0]).shape;
            const Shape &out = g.desc(nd.outputs[0]).shape;
            if (!staticShape(in) || !staticShape(out) || in.size() != out.size() || out.empty())
            {
                return false;
            }
            const int rank = (int) in.size();
            for (int i = 0; i + 1 < rank; ++i)
            {
                if (in[i] != out[i])
                {
                    return false; // only the last axis may narrow
                }
            }
            if (out[rank - 1] != half || in[rank - 1] != 2 * half)
            {
                return false;
            }
            std::vector<int64_t> axes = readI64Param(g, nd, "axes", 3);
            if (axes.size() != 1)
            {
                return false;
            }
            int64_t axis = axes[0] < 0 ? axes[0] + rank : axes[0];
            if (axis != rank - 1)
            {
                return false;
            }
            std::vector<int64_t> starts = readI64Param(g, nd, "starts", 1);
            if (starts.size() != 1 || starts[0] != lo)
            {
                return false;
            }
            for (int64_t st: readI64Param(g, nd, "steps", 4))
            {
                if (st != 1)
                {
                    return false;
                }
            }
            return true;
        }

        struct RopeSite {
            TensorId x        = kNoTensor; // the sliced source r4 [.., H, head]
            TensorId pos      = kNoTensor; // gather index (positions)
            TensorId cosTable = kNoTensor; // [P, half]
            TensorId sinTable = kNoTensor; // [P, half]
            int64_t  half     = 0;
            std::vector<TensorId> internal; // tensors whose stores the fusion removes
        };

        // Match the rotate-half site anchored at Concat node `ci`. Fills `site` on success.
        bool matchSite(const Graph &g, const std::vector<int> &producer, size_t ci, RopeSite &site) {
            const Node &cat = g.nodes[ci];
            if (cat.type != OpType::Concat || !bareNode(cat, {"axis"}) || cat.inputs.size() != 2)
            {
                return false;
            }
            const Shape &outShape = g.desc(cat.outputs[0]).shape;
            const Shape &o1Shape  = g.desc(cat.inputs[0]).shape;
            const Shape &o2Shape  = g.desc(cat.inputs[1]).shape;
            if (!staticShape(outShape) || !staticShape(o1Shape) || o1Shape != o2Shape || outShape.size() < 3)
            {
                return false;
            }
            const int rank = (int) outShape.size();
            int64_t   axis = cat.attr.geti("axis", 0);
            if (axis < 0)
            {
                axis += rank;
            }
            const int64_t half = o1Shape[rank - 1];
            if (axis != rank - 1 || outShape[rank - 1] != 2 * half)
            {
                return false;
            }

            // The two rotate products as monomial sums over their leaves.
            std::vector<TensorId> absorbed;
            Expr                  e1, e2;
            if (!exprOf(g, producer, cat.inputs[0], e1, absorbed, 0) || !exprOf(g, producer, cat.inputs[1], e2, absorbed, 0))
            {
                return false;
            }
            if (e1.size() != 2 || e2.size() != 2)
            {
                return false;
            }

            // o1 = sA*rA - sB*rB: exactly one +1 and one -1 degree-2 monomial. The stream/row role
            // of each factor is settled by structure below (a stream is a Slice of the source, a
            // row reaches a table Gather), so a single-head site — where both shapes coincide —
            // still resolves: both orientations of each monomial are tried.
            Monomial plus {kNoTensor, kNoTensor}, minus {kNoTensor, kNoTensor};
            for (const auto &kv: e1)
            {
                if (kv.first.first == kNoTensor || (kv.second != 1 && kv.second != -1))
                {
                    return false; // linear term or non-unit coefficient
                }
                (kv.second == 1 ? plus : minus) = kv.first;
            }
            if (plus.first == kNoTensor || minus.first == kNoTensor)
            {
                return false;
            }

            // One orientation attempt: (sA, rA) from the +1 monomial, (sB, rB) from the -1 one.
            auto tryOrientation = [&](TensorId sA, TensorId rA, TensorId sB, TensorId rB) {
                if (sA == sB || rA == rB || sA == rA || sB == rB)
                {
                    return false;
                }
                // A row broadcast has extent 1 on the H axis (rank-2) and the product shape elsewhere.
                auto rowShaped = [&](TensorId t) {
                    const Shape &s = g.desc(t).shape;
                    if ((int) s.size() != rank || s[rank - 1] != half || s[rank - 2] != 1)
                    {
                        return false;
                    }
                    for (int i = 0; i + 2 < rank; ++i)
                    {
                        if (s[i] != o1Shape[i])
                        {
                            return false;
                        }
                    }
                    return true;
                };
                if (g.desc(sA).shape != o1Shape || g.desc(sB).shape != o1Shape || !rowShaped(rA) || !rowShaped(rB))
                {
                    return false;
                }
                // o2 must be exactly sA*rB + sB*rA over the same leaves.
                Expr want = {{mono(sA, rB), 1}, {mono(sB, rA), 1}};
                if (e2 != want)
                {
                    return false;
                }

                // sA/sB are the last-axis halves of one source: sA = [0, half), sB = [half, 2*half).
                const int pA = sA >= 0 && sA < (int) producer.size() ? producer[sA] : -1;
                const int pB = sB >= 0 && sB < (int) producer.size() ? producer[sB] : -1;
                if (pA < 0 || pB < 0)
                {
                    return false;
                }
                const TensorId r4 = g.nodes[pA].inputs.empty() ? kNoTensor : g.nodes[pA].inputs[0];
                if (r4 == kNoTensor || !sliceOfHalf(g, g.nodes[pA], r4, 0, half) || !sliceOfHalf(g, g.nodes[pB], r4, half, half))
                {
                    return false;
                }
                if (g.desc(r4).shape != outShape)
                {
                    return false; // Rope writes the concat output in the source's element order
                }
                std::vector<TensorId> chain = absorbed;
                chain.push_back(sA);
                chain.push_back(sB);

                // rA/rB reach a Gather of a [P, half] table row by the SAME position tensor.
                const int gA = gatherBehind(g, producer, rA, chain);
                const int gB = gatherBehind(g, producer, rB, chain);
                if (gA < 0 || gB < 0)
                {
                    return false;
                }
                const Node &gcos = g.nodes[gA], &gsin = g.nodes[gB];
                auto        tableOk = [&](const Node &gn) {
                    const Shape &ts   = g.desc(gn.inputs[0]).shape;
                    int64_t      axis = gn.attr.geti("axis", 0);
                    return axis == 0 && ts.size() == 2 && ts[1] == half && staticShape(ts);
                };
                // The kernel bounds positions by ONE table height (the wrap and the clamp use the
                // cos table's rows), so both tables must have identical shapes — a shorter sin
                // table would be read past its end for positions the cos table still covers.
                if (!tableOk(gcos) || !tableOk(gsin) || gcos.inputs[1] != gsin.inputs[1] || g.desc(gcos.inputs[0]).shape != g.desc(gsin.inputs[0]).shape)
                {
                    return false;
                }
                // Positions must cover the leading dims exactly (one row per (batch..., seq)
                // index); a broadcast position vector over a larger batch keeps the decomposed form.
                int64_t lead = 1;
                for (int i = 0; i + 2 < rank; ++i)
                {
                    lead *= outShape[i];
                }
                const Shape &posShape = g.desc(gcos.inputs[1]).shape;
                if (posShape.empty() || !staticShape(posShape))
                {
                    return false;
                }
                int64_t posElems = 1;
                for (int64_t d: posShape)
                {
                    posElems *= d;
                }
                if (posElems != lead)
                {
                    return false;
                }

                site.x        = r4;
                site.pos      = gcos.inputs[1];
                site.cosTable = gcos.inputs[0];
                site.sinTable = gsin.inputs[0];
                site.half     = half;
                site.internal = std::move(chain);
                return true;
            };
            return tryOrientation(plus.first, plus.second, minus.first, minus.second) || tryOrientation(plus.first, plus.second, minus.second, minus.first) || tryOrientation(plus.second, plus.first, minus.first, minus.second) || tryOrientation(plus.second, plus.first, minus.second, minus.first);
        }

    } // namespace

    int fuseRope(Graph &g, const std::string &fp32Pins) {
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

        int fused = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            RopeSite site;
            if (!matchSite(g, producer, i, site))
            {
                continue;
            }
            // A pinned internal tensor keeps its decomposed form: the fusion would remove the fp32
            // store the pin exists for (mirrors foldMatMulViews' refusal).
            bool pinned = false;
            for (TensorId t: site.internal)
            {
                if (!fp32Pins.empty() && fp32NameMatch(g.desc(t).name, fp32Pins))
                {
                    pinned = true;
                    break;
                }
            }
            if (pinned)
            {
                continue;
            }
            Node &nd = g.nodes[i];
            nd.type  = OpType::Rope;
            nd.inputs = {site.x, site.pos, site.cosTable, site.sinTable};
            nd.attr.map.clear();
            Attr h;
            h.kind             = Attr::Int;
            h.i                = site.half;
            nd.attr.map["half"] = h;
            ++fused;
        }

        if (fused)
        {
            eliminateDeadNodes(g);
            VKNN_INFO << "fuseRope: fused " << fused << " rotate-half chain(s) into Rope node(s)";
        }
        return fused;
    }

} // namespace vknn
