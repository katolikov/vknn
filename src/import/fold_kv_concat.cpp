// foldFusedAttentionKvConcat: remove the per-token KV-cache Concat feeding a FusedAttention node.
//
// A with-past decoder concatenates the cached K/V rows with the current token's row every step
// (present = past ‖ new), then attention reads the concatenation and the present output feeds the
// cache fold. The concatenation copies the WHOLE cache per token; only its last row is new. This
// pass rewires a FusedAttention node to read the two sources directly — token s < pastLen from the
// past tensor, s >= pastLen from the new rows — so the Concat's only remaining consumer is the
// present graph output, and when that output is the concat result it is rewritten to the new-rows
// tensor (which inherits the present name). The Concat then dies by DCE and the cache-concat
// convention becomes the rows-only convention io_link.h documents; the engine-resident link fold
// and the example tools already read the row count from the present shape, so both conventions
// drive the same cache. The values every consumer reads are unchanged bit-for-bit — the fold moves
// no math, only the copy.
//
// Refusals keep the decomposed form: a non-bare Concat (fused work attached), a Concat axis that
// is not the attention token axis, K/V concats with mismatched geometry, addressing that does not
// decompose over the concat operands' canonical strides, or a concat result consumed by anything
// other than the one FusedAttention node and the graph output list.
//
// Runs at load only (never serialized), after fuseDecodeAttention, gated by Hint::KvConcatFold.
#include "core/fused_attention.h"
#include "passes_internal.h"
#include "vknn/graph.h"
#include "vknn/logging.h"
#include <algorithm>
#include <optional>
#include <set>
#include <vector>

namespace vknn {

    namespace {

        // Canonical row-major strides of `shape`.
        std::vector<int64_t> canonicalStrides(const Shape &shape) {
            std::vector<int64_t> cs(shape.size(), 1);
            for (int64_t i = (int64_t) shape.size() - 2; i >= 0; --i)
            {
                cs[(size_t) i] = cs[(size_t) i + 1] * shape[(size_t) i + 1];
            }
            return cs;
        }

        // Mixed-radix decomposition of `stride` over the canonical strides of `shape`: the per-axis
        // coordinate multipliers a view of `shape` expresses. Anything not exactly representable
        // (or stepping past an axis extent) is not a view of this tensor — the caller refuses.
        std::optional<std::vector<int64_t>> decomposeStride(int64_t stride, const Shape &shape, const std::vector<int64_t> &cs) {
            std::vector<int64_t> coeff(shape.size(), 0);
            int64_t              rem = stride;
            for (size_t i = 0; i < shape.size(); ++i)
            {
                coeff[i] = rem / cs[i];
                rem      = rem % cs[i];
                if (coeff[i] < 0 || coeff[i] >= shape[i])
                {
                    return std::nullopt;
                }
            }
            if (rem != 0)
            {
                return std::nullopt;
            }
            return coeff;
        }

        int64_t recompose(const std::vector<int64_t> &coeff, const std::vector<int64_t> &cs) {
            int64_t s = 0;
            for (size_t i = 0; i < coeff.size(); ++i)
            {
                s += coeff[i] * cs[i];
            }
            return s;
        }

        // One source operand's remapped addressing: the per-row-dim strides plus the per-token and
        // per-element strides, re-expressed over a concat operand's own canonical strides.
        struct RemappedSource {
            std::vector<int64_t> rowStride;
            int64_t              tokenStride = 0;
            int64_t              elemStride  = 0;
        };

        // Re-express addressing composed over the concat RESULT as addressing of one concat
        // OPERAND. The row-dim strides and the element stride must not step the concat axis; the
        // token stride must be exactly one concat-axis step.
        std::optional<RemappedSource> remapOntoOperand(const std::vector<int64_t> &rowStride, int64_t tokenStride, int64_t elemStride, const Shape &concatShape, int64_t axis, const Shape &operandShape) {
            const std::vector<int64_t> cs  = canonicalStrides(concatShape);
            const std::vector<int64_t> ocs = canonicalStrides(operandShape);
            RemappedSource             out;
            out.rowStride.reserve(rowStride.size());
            for (int64_t s: rowStride)
            {
                auto coeff = decomposeStride(s, concatShape, cs);
                if (!coeff || (*coeff)[(size_t) axis] != 0)
                {
                    return std::nullopt;
                }
                out.rowStride.push_back(recompose(*coeff, ocs));
            }
            auto tok = decomposeStride(tokenStride, concatShape, cs);
            if (!tok || (*tok)[(size_t) axis] != 1)
            {
                return std::nullopt;
            }
            for (size_t i = 0; i < tok->size(); ++i)
            {
                if (i != (size_t) axis && (*tok)[i] != 0)
                {
                    return std::nullopt;
                }
            }
            out.tokenStride = ocs[(size_t) axis];
            auto el         = decomposeStride(elemStride, concatShape, cs);
            if (!el || (*el)[(size_t) axis] != 0)
            {
                return std::nullopt;
            }
            out.elemStride = recompose(*el, ocs);
            return out;
        }

        // A concat is foldable only bare: no fused work, exactly two inputs, one output.
        bool bareConcat(const Node &nd) {
            if (nd.type != OpType::Concat || nd.inputs.size() != 2 || nd.outputs.size() != 1 || nd.inputs[0] == kNoTensor || nd.inputs[1] == kNoTensor)
            {
                return false;
            }
            if (nd.fusedResidual != kNoTensor || nd.fusedBias != kNoTensor || nd.subOp != 0)
            {
                return false;
            }
            for (const auto &kv: nd.attr.map)
            {
                if (kv.first != "axis")
                {
                    return false;
                }
            }
            return true;
        }

        // Every reference to `id` outside node `self`: node inputs, fused edges, and attr-referenced
        // side tensors are all consumers for liveness purposes.
        int externalConsumers(const Graph &g, TensorId id, int self) {
            int uses = 0;
            for (size_t n = 0; n < g.nodes.size(); ++n)
            {
                if ((int) n == self)
                {
                    continue;
                }
                const Node &nd = g.nodes[n];
                for (TensorId t: nd.inputs)
                {
                    uses += t == id ? 1 : 0;
                }
                uses += nd.fusedResidual == id ? 1 : 0;
                uses += nd.fusedBias == id ? 1 : 0;
            }
            return uses;
        }

    } // namespace

    int foldFusedAttentionKvConcat(Graph &g) {
        int folded = 0;
        // Producer index by tensor.
        std::vector<int> producer(g.tensors.size(), -1);
        for (int n = 0; n < (int) g.nodes.size(); ++n)
        {
            for (TensorId t: g.nodes[(size_t) n].outputs)
            {
                if (t != kNoTensor)
                {
                    producer[(size_t) t] = n;
                }
            }
        }

        for (size_t fi = 0; fi < g.nodes.size(); ++fi)
        {
            Node &fa = g.nodes[fi];
            if (fa.type != OpType::FusedAttention || fa.attr.geti(kFaSplit, 0) != 0 || fa.inputs.size() > 4)
            {
                continue;
            }
            const TensorId kSrc = fa.inputs[1], vSrc = fa.inputs[2];
            const int      kProd = producer[(size_t) kSrc], vProd = producer[(size_t) vSrc];
            if (kProd < 0 || vProd < 0 || kProd == vProd)
            {
                continue;
            }
            const Node &kCat = g.nodes[(size_t) kProd];
            const Node &vCat = g.nodes[(size_t) vProd];
            if (!bareConcat(kCat) || !bareConcat(vCat))
            {
                continue;
            }
            const Shape &kShape = g.desc(kSrc).shape;
            const Shape &vShape = g.desc(vSrc).shape;
            int64_t      kAxis  = kCat.attr.geti("axis", 0);
            int64_t      vAxis  = vCat.attr.geti("axis", 0);
            if (kAxis < 0)
            {
                kAxis += (int64_t) kShape.size();
            }
            if (vAxis < 0)
            {
                vAxis += (int64_t) vShape.size();
            }
            const Shape &kPastShape = g.desc(kCat.inputs[0]).shape;
            const Shape &kNewShape  = g.desc(kCat.inputs[1]).shape;
            const Shape &vPastShape = g.desc(vCat.inputs[0]).shape;
            const Shape &vNewShape  = g.desc(vCat.inputs[1]).shape;
            const int64_t C         = fa.attr.geti(kFaC);
            if (kAxis >= (int64_t) kShape.size() || vAxis >= (int64_t) vShape.size() || kShape[(size_t) kAxis] != C || vShape[(size_t) vAxis] != C)
            {
                continue;
            }
            const int64_t pastLen = kPastShape[(size_t) kAxis];
            if (pastLen <= 0 || pastLen >= C || vPastShape[(size_t) vAxis] != pastLen || kNewShape[(size_t) kAxis] != C - pastLen || vNewShape[(size_t) vAxis] != C - pastLen)
            {
                continue;
            }
            // The concat results must feed only this attention node (plus the graph output list).
            if (externalConsumers(g, kSrc, (int) fi) != 0 || externalConsumers(g, vSrc, (int) fi) != 0)
            {
                continue;
            }
            // fp32 pins on the concat result would be silently lost with it.
            if (g.desc(kSrc).storeFp32 || g.desc(vSrc).storeFp32)
            {
                continue;
            }

            const std::vector<int64_t> kStride = fa.attr.getints(kFaKStride);
            const std::vector<int64_t> vStride = fa.attr.getints(kFaVStride);
            auto                       kPast   = remapOntoOperand(kStride, fa.attr.geti(kFaKN), fa.attr.geti(kFaKK), kShape, kAxis, kPastShape);
            auto                       kNew    = remapOntoOperand(kStride, fa.attr.geti(kFaKN), fa.attr.geti(kFaKK), kShape, kAxis, kNewShape);
            // V's element-stride convention is transposed relative to K: kFaVK steps the token
            // axis and kFaVN the output axis (contract in core/fused_attention.h).
            auto                       vPast   = remapOntoOperand(vStride, fa.attr.geti(kFaVK), fa.attr.geti(kFaVN), vShape, vAxis, vPastShape);
            auto                       vNew    = remapOntoOperand(vStride, fa.attr.geti(kFaVK), fa.attr.geti(kFaVN), vShape, vAxis, vNewShape);
            if (!kPast || !kNew || !vPast || !vNew)
            {
                continue;
            }

            // Rewire the attention node onto the two sources.
            const TensorId kPastT = kCat.inputs[0], kNewT = kCat.inputs[1];
            const TensorId vPastT = vCat.inputs[0], vNewT = vCat.inputs[1];
            fa.inputs.resize(4, kNoTensor); // slot 3 = mask or none
            fa.inputs.push_back(kNewT);
            fa.inputs.push_back(vNewT);
            fa.inputs[1] = kPastT;
            fa.inputs[2] = vPastT;
            auto setInt = [&](const char *key, int64_t value) {
                Attr a;
                a.kind          = Attr::Int;
                a.i             = value;
                fa.attr.map[key] = a;
            };
            auto setInts = [&](const char *key, std::vector<int64_t> value) {
                Attr a;
                a.kind          = Attr::Ints;
                a.ints          = std::move(value);
                fa.attr.map[key] = a;
            };
            setInt(kFaSplit, 1);
            setInt(kFaPastLen, pastLen);
            setInts(kFaKStride, std::move(kPast->rowStride));
            setInt(kFaKN, kPast->tokenStride);
            setInt(kFaKK, kPast->elemStride);
            setInts(kFaVStride, std::move(vPast->rowStride));
            setInt(kFaVK, vPast->tokenStride);
            setInt(kFaVN, vPast->elemStride);
            setInts(kFaKNewStride, std::move(kNew->rowStride));
            setInt(kFaKNewN, kNew->tokenStride);
            setInt(kFaKNewK, kNew->elemStride);
            setInts(kFaVNewStride, std::move(vNew->rowStride));
            setInt(kFaVNewK, vNew->tokenStride);
            setInt(kFaVNewN, vNew->elemStride);

            // A concat result on the graph output list becomes the new-rows tensor under the same
            // name (the rows-only present convention); the dead concat tensor keeps a suffixed name.
            for (auto side: {std::pair<TensorId, TensorId> {kSrc, kNewT}, std::pair<TensorId, TensorId> {vSrc, vNewT}})
            {
                auto it = std::find(g.outputs.begin(), g.outputs.end(), side.first);
                if (it == g.outputs.end())
                {
                    continue;
                }
                *it                                      = side.second;
                const std::string presentName            = g.tensors[(size_t) side.first].name;
                const std::string retiredName            = presentName + "#concat";
                g.tensors[(size_t) side.first].name      = retiredName;
                g.tensors[(size_t) side.second].name     = presentName;
                g.tensors[(size_t) side.first].isOutput  = false;
                g.tensors[(size_t) side.second].isOutput = true;
                // Keep the name index consistent: the present name must resolve to the new-rows
                // tensor (the link and boundary paths look outputs up by name).
                g.tensorByName[presentName]              = side.second;
                g.tensorByName[retiredName]              = side.first;
            }
            ++folded;
        }

        if (folded)
        {
            eliminateDeadNodes(g);
            VKNN_INFO << "foldFusedAttentionKvConcat: folded " << folded << " KV concat(s) into split-source attention";
        }
        return folded;
    }

} // namespace vknn
