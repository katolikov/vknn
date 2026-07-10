#include "core/fused_attention.h"
#include "core/matmul_view.h"
#include "core/quant_int4.h"
#include "passes_internal.h"
#include <cstdint>
#include <optional>

namespace vknn {
    namespace {

        // A node whose only work is the value it computes into outputs[0]: no fused pointwise unit,
        // activation, bias, or residual riding on it, and exactly one output. Mirrors the
        // bareChainNode discipline of fold_matmul_views.cpp — fusing across a node that carries
        // hidden computation would silently drop that computation.
        bool bareNode(const Node &n) {
            return !n.attr.has("pw_steps") && n.fusedAct == ActType::None && n.fusedBias == kNoTensor && n.fusedResidual == kNoTensor && n.outputs.size() == 1;
        }

        // The scale/mask affine transform applied to the raw q.k^T dot before the softmax:
        //   score = dot * scale + mask[.] * maskScale
        // Composed step by step from the chain between the QK MatMul and the Softmax (a fused
        // pointwise epilogue on the MatMul, or standalone Mul/Add nodes): a scalar multiply scales
        // both terms accumulated so far, an additive operand becomes the mask. Any step outside
        // this family refuses the whole fusion.
        struct ScaleMask {
            float    scale     = 1.f;
            float    maskScale = 1.f;
            TensorId mask      = kNoTensor;
        };

        bool scalarInitValue(const Graph &g, TensorId t, float *out) {
            if (t == kNoTensor || !g.isInitializer(t))
            {
                return false;
            }
            const Shape &s = g.desc(t).shape;
            if (!s.empty() && numElements(s) != 1)
            {
                return false;
            }
            std::vector<float> v = initFloats(g, t);
            if (v.empty())
            {
                return false;
            }
            *out = v[0];
            return true;
        }

        // NumPy-broadcast compatibility of `operand` against `target` (left-padded ranks).
        bool broadcastsTo(const Shape &operand, const Shape &target) {
            if (operand.size() > target.size())
            {
                return false;
            }
            size_t pad = target.size() - operand.size();
            for (size_t i = 0; i < operand.size(); ++i)
            {
                if (operand[i] != 1 && operand[i] != target[pad + i])
                {
                    return false;
                }
            }
            return true;
        }

        // Apply one scale/mask step: a multiply by a scalar initializer, or an add of an operand
        // broadcastable over the scores shape. Returns false (refuse) for anything else.
        bool absorbStep(const Graph &g, ScaleMask &sm, int binCode, TensorId operand, const Shape &scores) {
            if (binCode == (int) BinaryType::Mul)
            {
                float v = 0.f;
                if (!scalarInitValue(g, operand, &v))
                {
                    return false;
                }
                sm.scale *= v;
                if (sm.mask != kNoTensor)
                {
                    sm.maskScale *= v;
                }
                return true;
            }
            if (binCode == (int) BinaryType::Add)
            {
                if (sm.mask != kNoTensor || operand == kNoTensor)
                {
                    return false;
                }
                if (!broadcastsTo(g.desc(operand).shape, scores))
                {
                    return false;
                }
                sm.mask      = operand;
                sm.maskScale = 1.f;
                return true;
            }
            return false;
        }

        // Decode a fused pointwise epilogue on the QK MatMul as a scale/mask transform. Accepted
        // steps are binary Mul-by-scalar-initializer / Add-of-mask chained through the accumulator
        // (the first step reads the entry value); a register destination, an extra exported stream
        // (pw_outs), or any other step kind/code/source refuses. Step record layout: 8 ints per
        // step — kind, code, srcA, srcB, srcC, dst, bcast, bcastSrc (see op_type.h).
        std::optional<ScaleMask> decodeEpilogue(const Graph &g, const Node &qk, const Shape &scores) {
            ScaleMask sm;
            if (!qk.attr.has("pw_steps"))
            {
                return sm;
            }
            if (qk.attr.has("pw_outs") && !qk.attr.getints("pw_outs").empty())
            {
                return std::nullopt;
            }
            const std::vector<int64_t> &steps = qk.attr.getints("pw_steps");
            if (steps.empty() || steps.size() % 8 != 0)
            {
                return std::nullopt;
            }
            const int64_t opBase = qk.attr.geti("pw_opbase", (int64_t) qk.inputs.size());
            for (size_t st = 0; st < steps.size(); st += 8)
            {
                const int64_t kind = steps[st], code = steps[st + 1];
                const int64_t srcA = steps[st + 2], srcB = steps[st + 3], dst = steps[st + 5];
                if (kind != kPwKindBinary || dst != kPwRefNone)
                {
                    return std::nullopt;
                }
                // The running value: the entry for step 0, the accumulator after; commutative
                // steps may carry it in either source slot.
                const int64_t chainRef = st == 0 ? (int64_t) kPwRefEntry : (int64_t) kPwRefAcc;
                int64_t       opRef;
                if (srcA == chainRef)
                {
                    opRef = srcB;
                } else if (srcB == chainRef)
                {
                    opRef = srcA;
                } else
                {
                    return std::nullopt;
                }
                if (opRef > kPwRefOp0)
                {
                    return std::nullopt; // not a tensor operand (acc/entry/register/none)
                }
                // Operand refs index node.inputs directly (kPwRefOp0 - i encodes input i); a ref
                // into the core operand range would alias a matmul input and is refused.
                const int64_t opIdx = kPwRefOp0 - opRef;
                if (opIdx < opBase || opIdx >= (int64_t) qk.inputs.size())
                {
                    return std::nullopt;
                }
                if (!absorbStep(g, sm, (int) code, qk.inputs[(size_t) opIdx], scores))
                {
                    return std::nullopt;
                }
            }
            return sm;
        }

        // Row-major dense strides of `dims`; a size-1 axis canonicalizes to stride 0 (the form the
        // view fold emits, so comparisons against folded stride arrays are direct).
        std::vector<int64_t> denseStrides(const std::vector<int64_t> &dims) {
            std::vector<int64_t> st(dims.size(), 0);
            int64_t              acc = 1;
            for (int i = (int) dims.size() - 1; i >= 0; --i)
            {
                st[(size_t) i] = dims[(size_t) i] == 1 ? 0 : acc;
                acc *= dims[(size_t) i];
            }
            return st;
        }

        // Per-refined-dim strides of the mask operand over the refined score dims. The refined dims
        // are a row-major split of the score axes (the operand-view contract), so each score axis's
        // (extent, mask stride) block splits across the dims it refines; a broadcast mask axis is a
        // zero-stride block at any split. nullopt when a dim boundary falls at a non-divisible point
        // (never for dims produced by the fold, kept as a guard).
        std::optional<std::vector<int64_t>> maskStridesOverDims(const Shape &scores, const Shape &mask, const std::vector<int64_t> &dims) {
            const size_t         pad = scores.size() - mask.size();
            std::vector<int64_t> axisStride(scores.size(), 0);
            std::vector<int64_t> maskDense = denseStrides(std::vector<int64_t>(mask.begin(), mask.end()));
            for (size_t i = 0; i < scores.size(); ++i)
            {
                if (i >= pad && mask[i - pad] == scores[i])
                {
                    axisStride[i] = maskDense[i - pad];
                }
                // A broadcast (size-1 or left-padded) mask axis keeps stride 0.
            }
            std::vector<int64_t> out(dims.size(), 0);
            size_t               axis = 0;
            int64_t              rem = scores.empty() ? 1 : scores[0], stride = axisStride.empty() ? 0 : axisStride[0];
            for (size_t j = 0; j < dims.size(); ++j)
            {
                while (rem == 1 && axis + 1 < scores.size())
                {
                    ++axis;
                    rem    = scores[axis];
                    stride = axisStride[axis];
                }
                const int64_t d = dims[j];
                if (d == 1)
                {
                    out[j] = 0;
                    continue;
                }
                if (rem % d != 0)
                {
                    return std::nullopt;
                }
                rem /= d;
                out[j] = stride * rem; // outer factor steps by the remaining inner extent
                if (rem == 1)
                {
                    stride = 0;
                }
            }
            if (rem != 1)
            {
                return std::nullopt;
            }
            return out;
        }

        // True when Transpose(perm) of a dense row-major tensor of `shape` leaves the flat element
        // order unchanged: the non-size-1 source axes must keep their relative order under perm.
        bool flatIdentityPerm(const Shape &shape, const std::vector<int64_t> &perm) {
            if (perm.size() != shape.size())
            {
                return false;
            }
            int64_t last = -1;
            for (int64_t p: perm)
            {
                if (p < 0 || p >= (int64_t) shape.size())
                {
                    return false;
                }
                if (shape[(size_t) p] == 1)
                {
                    continue;
                }
                if (p < last)
                {
                    return false;
                }
                last = p;
            }
            return true;
        }

        struct Consumers {
            std::vector<int>  count; // consumer node count per tensor
            std::vector<int>  sole;  // the single consumer node index (valid when count == 1)
            std::vector<char> isOut; // tensor is a graph output
        };

        Consumers buildConsumers(const Graph &g) {
            Consumers c;
            c.count.assign(g.tensors.size(), 0);
            c.sole.assign(g.tensors.size(), -1);
            c.isOut.assign(g.tensors.size(), 0);
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                for (TensorId t: g.nodes[i].inputs)
                {
                    if (t >= 0 && t < (TensorId) c.count.size())
                    {
                        c.count[t]++;
                        c.sole[t] = (int) i;
                    }
                }
                if (g.nodes[i].fusedResidual != kNoTensor)
                {
                    c.count[g.nodes[i].fusedResidual]++;
                }
            }
            for (TensorId t: g.outputs)
            {
                if (t >= 0 && t < (TensorId) c.isOut.size())
                {
                    c.isOut[t] = 1;
                }
            }
            return c;
        }

        // A tensor the fusion would erase must be strictly interior: one consumer, not a graph
        // output, and not pinned to fp32 storage by the later markFp32 pass (removing it would
        // remove the fp32 store the pin exists for).
        bool interior(const Graph &g, const Consumers &c, TensorId t, const std::string &fp32Pins) {
            if (t < 0 || t >= (TensorId) c.count.size() || c.count[t] != 1 || c.isOut[t])
            {
                return false;
            }
            return fp32Pins.empty() || !fp32NameMatch(g.desc(t).name, fp32Pins);
        }

    } // namespace

    // Fuse the single-query decode-attention chain
    //   MatMul(view) [-> scale/mask pointwise] -> Softmax -> MatMul(view) [-> Transpose -> Reshape]
    // into one FusedAttention node (core/fused_attention.h), so a decode step's attention core is
    // one dispatch per layer and the score/probability intermediates never round-trip through
    // memory. Consumes the operand-view stride attrs foldMatMulViews composed — the fused kernel
    // reads q/k/v through those same strides, which is what makes the fusion work on the GQA cache
    // layout without materialization — so it must run after that pass; a chain the fold left
    // materialized never matches here. Matches only the M == 1 (query length 1) decode form;
    // prefill and every non-attention graph are untouched. Numerics: dot products and the softmax
    // run in fp32 with no fp16 score store, so a fused graph is numerically finer than — not
    // byte-identical to — the decomposed chain. Runs at session load only (Hint::FusedAttention);
    // never serialized.
    void fuseDecodeAttention(Graph &g, const std::string &fp32Pins) {
        std::vector<int> producer(g.tensors.size(), -1);
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            for (TensorId t: g.nodes[i].outputs)
            {
                if (t >= 0 && t < (TensorId) producer.size())
                {
                    producer[t] = (int) i;
                }
            }
        }
        Consumers cons = buildConsumers(g);

        std::set<int>     remove;
        std::vector<Node> added;
        int               fused = 0;

        for (size_t si = 0; si < g.nodes.size(); ++si)
        {
            const Node &sm = g.nodes[si];
            if (sm.type != OpType::Softmax || !bareNode(sm) || remove.count((int) si))
            {
                continue;
            }
            // Softmax over the last axis of the scores.
            const Shape &scores = g.desc(sm.inputs[0]).shape;
            if (scores.empty())
            {
                continue;
            }
            int64_t axis = sm.attr.geti("axis", -1);
            if (axis < 0)
            {
                axis += (int64_t) scores.size();
            }
            if (axis != (int64_t) scores.size() - 1)
            {
                continue;
            }

            // --- the QK side: walk producers up through the scale/mask chain to a view MatMul ---
            struct ChainStep {
                int      node;
                TensorId operand;
                int      code;
            };
            std::vector<ChainStep> chainUp; // standalone scale/mask nodes between qk and softmax
            TensorId               cur     = sm.inputs[0];
            int                    qkIdx   = -1;
            bool                   refused = false;
            for (int hops = 0; hops < 4 && !refused; ++hops)
            {
                int p = cur >= 0 && cur < (TensorId) producer.size() ? producer[cur] : -1;
                if (p < 0 || remove.count(p))
                {
                    refused = true;
                    break;
                }
                const Node &pn = g.nodes[(size_t) p];
                if (pn.type == OpType::MatMul)
                {
                    qkIdx = p;
                    break;
                }
                // Standalone scale (Binary Mul by a scalar initializer) or mask (Add) node. The
                // composition is order-sensitive, so the chain is collected here and absorbed
                // top-down once the MatMul is found.
                const bool mulNode = pn.type == OpType::Binary && pn.subOp == (int32_t) BinaryType::Mul;
                const bool addNode = pn.type == OpType::Add || (pn.type == OpType::Binary && pn.subOp == (int32_t) BinaryType::Add);
                if ((!mulNode && !addNode) || !bareNode(pn) || pn.inputs.size() != 2 || g.desc(pn.outputs[0]).shape != scores)
                {
                    refused = true;
                    break;
                }
                // The chain-value input continues toward the MatMul; the other is the step operand.
                auto chainCandidate = [&](TensorId t) {
                    int q = t >= 0 && t < (TensorId) producer.size() ? producer[t] : -1;
                    if (q < 0)
                    {
                        return false;
                    }
                    OpType ty = g.nodes[(size_t) q].type;
                    return ty == OpType::MatMul || ty == OpType::Binary || ty == OpType::Add;
                };
                TensorId chainIn;
                if (chainCandidate(pn.inputs[0]))
                {
                    chainIn = pn.inputs[0];
                } else if (chainCandidate(pn.inputs[1]))
                {
                    chainIn = pn.inputs[1];
                } else
                {
                    refused = true;
                    break;
                }
                chainUp.push_back({p, chainIn == pn.inputs[0] ? pn.inputs[1] : pn.inputs[0], mulNode ? (int) BinaryType::Mul : (int) BinaryType::Add});
                cur = chainIn;
            }
            if (refused || qkIdx < 0)
            {
                continue;
            }
            const Node &qk = g.nodes[(size_t) qkIdx];
            if (!qk.attr.has(kMmView) || qk.attr.has(kWq) || qk.fusedAct != ActType::None || qk.fusedBias != kNoTensor || qk.fusedResidual != kNoTensor ||
                qk.outputs.size() != 1)
            {
                continue;
            }
            if (g.desc(qk.outputs[0]).shape != scores)
            {
                continue; // a broadcast-up between the MatMul and the softmax breaks the row map
            }
            // The epilogue (if any) applies first, then the standalone nodes compose top-down.
            ScaleMask smk;
            {
                auto epi = decodeEpilogue(g, qk, g.desc(qk.outputs[0]).shape);
                if (!epi)
                {
                    continue;
                }
                smk = *epi;
            }
            for (auto it = chainUp.rbegin(); it != chainUp.rend() && !refused; ++it)
            {
                if (!absorbStep(g, smk, it->code, it->operand, scores))
                {
                    refused = true;
                }
            }
            if (refused)
            {
                continue;
            }

            // Every tensor between the QK MatMul and the Softmax is erased by the fusion.
            if (!interior(g, cons, qk.outputs[0], fp32Pins) || !interior(g, cons, sm.outputs[0], fp32Pins))
            {
                continue;
            }
            for (const ChainStep &c: chainUp)
            {
                if (!interior(g, cons, g.nodes[(size_t) c.node].outputs[0], fp32Pins))
                {
                    refused = true;
                }
            }
            if (refused)
            {
                continue;
            }

            // --- the PV side: the softmax's sole consumer, a view MatMul reading it as A ---
            const int pvIdx = cons.sole[sm.outputs[0]];
            if (pvIdx < 0 || remove.count(pvIdx))
            {
                continue;
            }
            const Node &pv = g.nodes[(size_t) pvIdx];
            if (pv.type != OpType::MatMul || !bareNode(pv) || !pv.attr.has(kMmView) || pv.attr.has(kWq) || pv.inputs.size() < 2 || pv.inputs[0] != sm.outputs[0])
            {
                continue;
            }

            // --- geometry: both views must agree on the decode shape ---
            const std::vector<int64_t> &qkDims = qk.attr.getints(kMmViewDims);
            const std::vector<int64_t> &qkA    = qk.attr.getints(kMmViewAStride);
            const std::vector<int64_t> &qkB    = qk.attr.getints(kMmViewBStride);
            const std::vector<int64_t> &pvDims = pv.attr.getints(kMmViewDims);
            const std::vector<int64_t> &pvA    = pv.attr.getints(kMmViewAStride);
            const std::vector<int64_t> &pvB    = pv.attr.getints(kMmViewBStride);
            const size_t                rank   = qkDims.size();
            if (rank < 2 || pvDims.size() != rank || qkA.size() != rank || qkB.size() != rank || pvA.size() != rank || pvB.size() != rank)
            {
                continue;
            }
            const int64_t C  = qk.attr.geti(kMmViewN);
            const int64_t hd = pv.attr.geti(kMmViewN);
            if (qk.attr.geti(kMmViewM) != 1 || pv.attr.geti(kMmViewM) != 1)
            {
                continue; // prefill (M > 1) keeps the primitive path
            }
            if (pv.attr.geti(kMmViewK) != C || qk.attr.geti(kMmViewK) <= 0 || C <= 0 || C > kFaMaxContext || hd <= 0 || hd > kFaMaxHeadDim)
            {
                continue;
            }
            bool dimsMatch = true;
            for (size_t d = 0; d + 2 < rank; ++d)
            {
                dimsMatch = dimsMatch && qkDims[d] == pvDims[d];
            }
            if (!dimsMatch || qkDims[rank - 2] != 1 || pvDims[rank - 2] != 1 || qkDims[rank - 1] != C || pvDims[rank - 1] != hd)
            {
                continue;
            }
            // The PV A operand must read the scores dense row-major over the refined dims — the
            // exact layout the QK kernel writes — with the token axis as its k walk.
            {
                std::vector<int64_t> expect = denseStrides(qkDims);
                bool                 denseA = pv.attr.geti(kMmViewAK) == 1 && pvA[rank - 1] == 0;
                for (size_t d = 0; denseA && d + 2 < rank; ++d)
                {
                    denseA = pvDims[d] == 1 || pvA[d] == expect[d];
                }
                if (!denseA)
                {
                    continue;
                }
            }

            // --- the output tail: fold a flat-identity Transpose/Reshape chain when present ---
            TensorId         outT = pv.outputs[0];
            std::vector<int> tail;
            while (true)
            {
                if (cons.isOut[outT] || cons.count[outT] != 1)
                {
                    break;
                }
                const int   ci = cons.sole[outT];
                const Node &cn = g.nodes[(size_t) ci];
                if (remove.count(ci) || !bareNode(cn))
                {
                    break;
                }
                if (cn.type == OpType::Transpose)
                {
                    if (!flatIdentityPerm(g.desc(cn.inputs[0]).shape, cn.attr.getints("perm")))
                    {
                        break;
                    }
                } else if (cn.type != OpType::Reshape && cn.type != OpType::Squeeze && cn.type != OpType::Unsqueeze && cn.type != OpType::Flatten)
                { break; }
                if (numElements(g.desc(cn.outputs[0]).shape) != numElements(g.desc(outT).shape))
                {
                    break;
                }
                if (!fp32Pins.empty() && fp32NameMatch(g.desc(outT).name, fp32Pins))
                {
                    break;
                }
                tail.push_back(ci);
                outT = cn.outputs[0];
            }

            // --- mask addressing over the refined dims ---
            std::vector<int64_t> mStride;
            int64_t              mN = 0;
            if (smk.mask != kNoTensor)
            {
                auto ms = maskStridesOverDims(scores, g.desc(smk.mask).shape, qkDims);
                if (!ms)
                {
                    continue;
                }
                mN = (*ms)[rank - 1];
                mStride.assign(ms->begin(), ms->end() - 2);
            }

            // --- emit the fused node ---
            Node fa;
            fa.type   = OpType::FusedAttention;
            fa.name   = qk.name + "#fattn";
            fa.inputs = {qk.inputs[0], qk.inputs[1], pv.inputs[1]};
            if (smk.mask != kNoTensor)
            {
                fa.inputs.push_back(smk.mask);
            }
            fa.outputs  = {outT};
            auto setInt = [&](const char *k, int64_t v) {
                Attr a;
                a.kind         = Attr::Int;
                a.i            = v;
                fa.attr.map[k] = a;
            };
            auto setInts = [&](const char *k, std::vector<int64_t> v) {
                Attr a;
                a.kind         = Attr::Ints;
                a.ints         = std::move(v);
                fa.attr.map[k] = a;
            };
            auto setFloat = [&](const char *k, float v) {
                Attr a;
                a.kind         = Attr::Float;
                a.f            = v;
                fa.attr.map[k] = a;
            };
            std::vector<int64_t> rowDims(qkDims.begin(), qkDims.end() - 2);
            std::vector<int64_t> qStride(qkA.begin(), qkA.end() - 2);
            std::vector<int64_t> kStride(qkB.begin(), qkB.end() - 2);
            std::vector<int64_t> vStride(pvB.begin(), pvB.end() - 2);
            setInt(kFa, kFaVersion);
            setInts(kFaDims, std::move(rowDims));
            setInts(kFaQStride, std::move(qStride));
            setInts(kFaKStride, std::move(kStride));
            setInts(kFaVStride, std::move(vStride));
            setInt(kFaQK, qk.attr.geti(kMmViewAK));
            setInt(kFaKN, qkB[rank - 1]);
            setInt(kFaKK, qk.attr.geti(kMmViewBK));
            setInt(kFaVN, pvB[rank - 1]);
            setInt(kFaVK, pv.attr.geti(kMmViewBK));
            setInt(kFaC, C);
            setInt(kFaHd, hd);
            setFloat(kFaScale, smk.scale);
            if (smk.mask != kNoTensor)
            {
                setInts(kFaMStride, std::move(mStride));
                setInt(kFaMN, mN);
                setFloat(kFaMaskScale, smk.maskScale);
            }
            {
                const Shape         &os = g.desc(outT).shape;
                std::vector<int64_t> ov(os.begin(), os.end());
                setInts(kFaOut, std::move(ov));
            }
            // The GPU kernel stages group*headDim q values in shared memory; a product past the
            // staging cap cannot dispatch, so the site keeps its decomposed form. The group is the
            // first row axis K and V do not advance along (the op's own detection rule).
            {
                const std::vector<int64_t> &fd = fa.attr.getints(kFaDims);
                const std::vector<int64_t> &fk = fa.attr.getints(kFaKStride);
                const std::vector<int64_t> &fv = fa.attr.getints(kFaVStride);
                int64_t                     groupSize = 1;
                for (size_t d = 0; d < fd.size(); ++d)
                {
                    if (fd[d] > 1 && fk[d] == 0 && fv[d] == 0)
                    {
                        groupSize = fd[d];
                        break;
                    }
                }
                if (groupSize * hd > kFaMaxStaging)
                {
                    continue;
                }
            }
            added.push_back(std::move(fa));
            remove.insert(qkIdx);
            for (const ChainStep &c: chainUp)
            {
                remove.insert(c.node);
            }
            remove.insert((int) si);
            remove.insert(pvIdx);
            for (int t: tail)
            {
                remove.insert(t);
            }
            ++fused;
        }

        if (fused)
        {
            std::vector<Node> kept;
            kept.reserve(g.nodes.size());
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                if (!remove.count((int) i))
                {
                    kept.push_back(std::move(g.nodes[i]));
                }
            }
            for (Node &a: added)
            {
                kept.push_back(std::move(a));
            }
            g.nodes = std::move(kept);
            g.topoSort();
            eliminateDeadNodes(g);
            VKNN_INFO << "fuseDecodeAttention: fused " << fused << " decode-attention chain(s)";
        }
    }

} // namespace vknn
