// Expand the ORT contrib operators (com.microsoft domain) an onnxruntime transformer export uses
// into the primitive ops the backends already execute — no contrib op has (or needs) a kernel:
//
//   SimplifiedLayerNormalization            -> OpType::RMSNorm (same (x, gamma) + epsilon contract)
//   SkipSimplifiedLayerNormalization        -> Add(input, skip[, bias]) + RMSNorm
//   SkipLayerNormalization                  -> Add(input, skip[, bias]) + LayerNorm
//   RotaryEmbedding (interleaved=0)         -> rotate-half RoPE subgraph over the cos/sin caches
//   MultiHeadAttention (pure q/k/v + mask)  -> reshape/transpose + QK^T/scale/mask/softmax/V subgraph
//   MatMulNBits (4-bit, no zero_points)     -> MatMul with the weight repacked into the int4 wq
//                                              format (core/quant_int4.h) — the packed payload runs
//                                              natively on the Vulkan int4 kernels and materializes
//                                              to fp16 for every other consumer
//   GroupQueryAttention (do_rotary, causal) -> rotate-half RoPE on q/k, Concat(past, new) presents,
//                                              repeat_kv Expand reads, and the scale/mask/Softmax
//                                              attention core — the same primitive pattern the
//                                              optimum with-past exports carry, so the load-time
//                                              passes (fuseRope, foldMatMulViews,
//                                              fuseDecodeAttention, KvConcatFold) re-fuse it with
//                                              no pass changes
//
// A node whose optional inputs/attributes fall outside the expanded form is LEFT IN PLACE: it keeps
// its real name through the support report ("MultiHeadAttention: ..." instead of "Unknown") and the
// plan fails loudly, never silently miscomputing.
//
// The attention/rotary expansions emit CONCRETE shape constants, so they need their data inputs'
// shapes resolved. The pass therefore runs as a fixpoint with inferShapes: each round expands every
// contrib node whose input shapes are known, then re-infers so the next layer's chain resolves
// (a decoder alternates norm -> projections -> attention per layer, so one round unlocks the next).
// The pass is a no-op (one scan) for graphs without contrib ops.
#include "core/quant_int4.h"
#include "import/passes.h"
#include "passes_internal.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace vknn {

    namespace {

        bool shapeKnown(const Graph &g, TensorId t) {
            if (t == kNoTensor)
            {
                return false;
            }
            const Shape &s = g.desc(t).shape;
            if (s.empty())
            {
                return false;
            }
            for (int64_t d: s)
            {
                if (d < 0)
                {
                    return false;
                }
            }
            return true;
        }

        TensorId optionalInput(const Node &nd, size_t idx) {
            return idx < nd.inputs.size() ? nd.inputs[idx] : kNoTensor;
        }

        // A fresh unnamed activation tensor (shape filled by the following inferShapes round).
        TensorId newTensor(Graph &g, const std::string &name) {
            TensorDesc d;
            d.name = name;
            return g.addTensor(d);
        }

        // A fresh activation tensor carrying an integer dtype (a position/index chain member), so
        // the runtime keeps its exact integer storage on the CPU backend.
        TensorId newTensorI64(Graph &g, const std::string &name) {
            TensorDesc d;
            d.name  = name;
            d.dtype = DType::Int64;
            return g.addTensor(d);
        }

        TensorId addConstI64Shaped(Graph &g, const std::string &name, Shape shape, const std::vector<int64_t> &v) {
            TensorDesc d;
            d.name          = name;
            d.shape         = std::move(shape);
            d.dtype         = DType::Int64;
            d.isInitializer = true;
            TensorId id     = g.addTensor(d);
            HostBuffer hb;
            hb.resizeElems((int64_t) v.size(), DType::Int64);
            std::memcpy(hb.i64(), v.data(), v.size() * 8);
            g.initializers[id] = std::move(hb);
            return id;
        }

        TensorId addConstI64(Graph &g, const std::string &name, const std::vector<int64_t> &v) {
            return addConstI64Shaped(g, name, {(int64_t) v.size()}, v);
        }

        TensorId addConstF32Shaped(Graph &g, const std::string &name, Shape shape, const std::vector<float> &v) {
            TensorDesc d;
            d.name          = name;
            d.shape         = std::move(shape);
            d.isInitializer = true;
            TensorId id     = g.addTensor(d);
            HostBuffer hb;
            hb.resizeElems((int64_t) v.size(), DType::Float32);
            std::memcpy(hb.f32(), v.data(), v.size() * 4);
            g.initializers[id] = std::move(hb);
            return id;
        }

        TensorId addConstF32(Graph &g, const std::string &name, float v) {
            return addConstF32Shaped(g, name, {1}, {v});
        }

        Node mkNode(OpType t, const std::string &name, std::vector<TensorId> in, std::vector<TensorId> out) {
            Node n;
            n.type    = t;
            n.name    = name;
            n.inputs  = std::move(in);
            n.outputs = std::move(out);
            return n;
        }

        void setAttrI(Node &n, const char *k, int64_t v) {
            Attr a;
            a.kind          = Attr::Int;
            a.i             = v;
            n.attr.map[k]   = a;
        }
        void setAttrF(Node &n, const char *k, float v) {
            Attr a;
            a.kind          = Attr::Float;
            a.f             = v;
            n.attr.map[k]   = a;
        }
        void setAttrInts(Node &n, const char *k, std::vector<int64_t> v) {
            Attr a;
            a.kind          = Attr::Ints;
            a.ints          = std::move(v);
            n.attr.map[k]   = a;
        }

        Node mkBinary(BinaryType op, const std::string &name, TensorId a, TensorId b, TensorId out) {
            Node n = mkNode(OpType::Binary, name, {a, b}, {out});
            n.subOp = (int32_t) op;
            return n;
        }

        // SimplifiedLayerNormalization -> RMSNorm, in place. Declines a non-last axis or a consumed
        // extra output (the optional inv_std_var).
        bool expandSimplifiedLayerNorm(Graph &g, Node &nd) {
            if (nd.inputs.size() < 2 || nd.attr.geti("axis", -1) != -1)
            {
                return false;
            }
            for (size_t i = 1; i < nd.outputs.size(); ++i)
            {
                if (nd.outputs[i] != kNoTensor)
                {
                    return false;
                }
            }
            const float eps = nd.attr.getf("epsilon", 1e-5f);
            nd.type         = OpType::RMSNorm;
            nd.inputs       = {nd.inputs[0], nd.inputs[1]};
            nd.outputs      = {nd.outputs[0]};
            nd.attr.map.clear();
            setAttrF(nd, "epsilon", eps);
            return true;
        }

        // SkipSimplifiedLayerNormalization / SkipLayerNormalization -> Add chain + RMSNorm/LayerNorm.
        // Output 3 (input_skip_bias_sum) is the residual passthrough and becomes the Add result.
        // SkipSimplified: (input, skip, gamma[, bias]); Skip: (input, skip, gamma[, beta][, bias]).
        bool expandSkipNorm(Graph &g, const Node &nd, std::vector<Node> &out) {
            const bool simplified = nd.type == OpType::SkipSimplifiedLayerNorm;
            if (nd.inputs.size() < 3)
            {
                return false;
            }
            // The optional mean/inv-std outputs (1 and 2) have no primitive equivalent here.
            if ((nd.outputs.size() > 1 && nd.outputs[1] != kNoTensor) || (nd.outputs.size() > 2 && nd.outputs[2] != kNoTensor))
            {
                return false;
            }
            const TensorId input = nd.inputs[0], skip = nd.inputs[1], gamma = nd.inputs[2];
            const TensorId beta = simplified ? kNoTensor : optionalInput(nd, 3);
            const TensorId bias = simplified ? optionalInput(nd, 3) : optionalInput(nd, 4);
            TensorId sum = nd.outputs.size() > 3 && nd.outputs[3] != kNoTensor ? nd.outputs[3]
                                                                               : newTensor(g, nd.name + "#sum");
            if (bias != kNoTensor)
            {
                TensorId pre = newTensor(g, nd.name + "#insum");
                out.push_back(mkNode(OpType::Add, nd.name + "#add", {input, skip}, {pre}));
                out.push_back(mkNode(OpType::Add, nd.name + "#addbias", {pre, bias}, {sum}));
            } else
            {
                out.push_back(mkNode(OpType::Add, nd.name + "#add", {input, skip}, {sum}));
            }
            Node norm = simplified ? mkNode(OpType::RMSNorm, nd.name + "#rms", {sum, gamma}, {nd.outputs[0]})
                                   : mkNode(OpType::LayerNorm, nd.name + "#ln",
                                            beta != kNoTensor ? std::vector<TensorId> {sum, gamma, beta} : std::vector<TensorId> {sum, gamma},
                                            {nd.outputs[0]});
            setAttrF(norm, "epsilon", nd.attr.getf("epsilon", 1e-5f));
            if (!simplified)
            {
                setAttrI(norm, "axis", -1);
            }
            out.push_back(std::move(norm));
            return true;
        }

        // The rotate-half subgraph shared by the RotaryEmbedding and GroupQueryAttention expansions
        // (x [B,S,H*head], positions with B*S elements, cos/sin caches [maxPos, head/2]):
        //   r4 = reshape(x, [B,S,H,head]); x1/x2 = last-axis halves
        //   cosU/sinU = unsqueeze(gather(cache, positions), axis 2)   [.., 1, head/2]
        //   y = reshape(concat(x1*cos - x2*sin, x1*sin + x2*cos), [B,S,H*head])
        // This is exactly the primitive form fuseRope re-collapses into one Rope dispatch at load.
        void emitRotateHalf(Graph &g, std::vector<Node> &out, const std::string &base, TensorId x,
                            TensorId pos, TensorId cosC, TensorId sinC,
                            int64_t B, int64_t S, int64_t H, int64_t head, TensorId y) {
            const int64_t half = head / 2;
            TensorId      r4   = newTensor(g, base + "#r4");
            out.push_back(mkNode(OpType::Reshape, base + "#reshape4",
                                 {x, addConstI64(g, base + "#s4", {B, S, H, head})}, {r4}));
            auto slice = [&](const char *tag, int64_t lo, int64_t hi) {
                TensorId t  = newTensor(g, base + tag);
                Node     sl = mkNode(OpType::Slice, base + "#slice" + tag, {r4}, {t});
                setAttrInts(sl, "starts", {lo});
                setAttrInts(sl, "ends", {hi});
                setAttrInts(sl, "axes", {3});
                out.push_back(std::move(sl));
                return t;
            };
            TensorId x1 = slice("#x1", 0, half);
            TensorId x2 = slice("#x2", half, head);
            auto gatherRows = [&](TensorId cache, const char *tag) {
                TensorId rows = newTensor(g, base + tag);
                Node     gn   = mkNode(OpType::Gather, base + "#gather" + tag, {cache, pos}, {rows});
                setAttrI(gn, "axis", 0);
                out.push_back(std::move(gn));
                TensorId un = newTensor(g, base + tag + std::string("u"));
                Node     uq = mkNode(OpType::Unsqueeze, base + "#unsq" + tag, {rows}, {un});
                setAttrInts(uq, "axes", {2});
                out.push_back(std::move(uq));
                return un;
            };
            TensorId cosU = gatherRows(cosC, "#cos");
            TensorId sinU = gatherRows(sinC, "#sin");

            TensorId x1c = newTensor(g, base + "#x1c"), x2s = newTensor(g, base + "#x2s");
            TensorId x1s = newTensor(g, base + "#x1s"), x2c = newTensor(g, base + "#x2c");
            out.push_back(mkBinary(BinaryType::Mul, base + "#mul_x1c", x1, cosU, x1c));
            out.push_back(mkBinary(BinaryType::Mul, base + "#mul_x2s", x2, sinU, x2s));
            out.push_back(mkBinary(BinaryType::Mul, base + "#mul_x1s", x1, sinU, x1s));
            out.push_back(mkBinary(BinaryType::Mul, base + "#mul_x2c", x2, cosU, x2c));
            TensorId o1 = newTensor(g, base + "#o1"), o2 = newTensor(g, base + "#o2");
            out.push_back(mkBinary(BinaryType::Sub, base + "#sub", x1c, x2s, o1));
            out.push_back(mkNode(OpType::Add, base + "#add", {x1s, x2c}, {o2}));
            TensorId cat = newTensor(g, base + "#cat");
            Node     cc  = mkNode(OpType::Concat, base + "#concat", {o1, o2}, {cat});
            setAttrI(cc, "axis", 3);
            out.push_back(std::move(cc));
            out.push_back(mkNode(OpType::Reshape, base + "#reshape3",
                                 {cat, addConstI64(g, base + "#s3", {B, S, H * head})}, {y}));
        }

        // RotaryEmbedding (x[B,S,H*head], position_ids[B,S] or [1,S], cos_cache/sin_cache
        // [maxPos, head/2], interleaved=0) -> the emitRotateHalf subgraph.
        // Declines interleaved=1, partial rotary_embedding_dim, and 4-D inputs (not in scope yet).
        bool expandRotary(Graph &g, const Node &nd, std::vector<Node> &out) {
            if (nd.inputs.size() < 4 || nd.attr.geti("interleaved", 0) != 0 || nd.attr.geti("rotary_embedding_dim", 0) != 0)
            {
                return false;
            }
            const TensorId x = nd.inputs[0], pos = nd.inputs[1], cosC = nd.inputs[2], sinC = nd.inputs[3];
            if (!shapeKnown(g, x) || !shapeKnown(g, cosC))
            {
                return false;
            }
            const Shape &xs = g.desc(x).shape;
            const Shape &cs = g.desc(cosC).shape;
            if (xs.size() != 3 || cs.size() != 2)
            {
                return false;
            }
            const int64_t B = xs[0], S = xs[1], E = xs[2];
            const int64_t half = cs[1], head = half * 2;
            if (head <= 0 || E % head != 0)
            {
                return false;
            }
            emitRotateHalf(g, out, nd.name, x, pos, cosC, sinC, B, S, E / head, head, nd.outputs[0]);
            return true;
        }

        // MultiHeadAttention in the PURE form this expansion covers: query/key/value activations of
        // equal head count, no qkv bias, no key padding mask, optional ADDITIVE attention bias
        // (input 5), no past/present KV (the export concatenates the cache outside the op):
        //   out[B,S,E] = softmax(scale * Q K^T + bias) V   over num_heads heads of E/num_heads.
        bool expandMha(Graph &g, const Node &nd, std::vector<Node> &out) {
            if (nd.inputs.size() < 3)
            {
                return false;
            }
            const TensorId q = nd.inputs[0], k = nd.inputs[1], v = nd.inputs[2];
            const TensorId qkvBias = optionalInput(nd, 3), padMask = optionalInput(nd, 4);
            const TensorId attnBias = optionalInput(nd, 5);
            const TensorId pastK = optionalInput(nd, 6), pastV = optionalInput(nd, 7);
            if (qkvBias != kNoTensor || padMask != kNoTensor || pastK != kNoTensor || pastV != kNoTensor)
            {
                return false;
            }
            for (size_t i = 1; i < nd.outputs.size(); ++i)
            {
                if (nd.outputs[i] != kNoTensor)
                {
                    return false; // a consumed present_key/value needs the KV-managed form
                }
            }
            const int64_t H = nd.attr.geti("num_heads", 0);
            if (H <= 0 || !shapeKnown(g, q) || !shapeKnown(g, k) || !shapeKnown(g, v))
            {
                return false;
            }
            const Shape &qs = g.desc(q).shape, &ks = g.desc(k).shape;
            if (qs.size() != 3 || ks.size() != 3 || g.desc(v).shape.size() != 3)
            {
                return false;
            }
            const int64_t B = qs[0], S = qs[1], E = qs[2], T = ks[1];
            if (E % H != 0 || ks[2] != E || g.desc(v).shape[2] != E)
            {
                return false; // unequal q/k/v widths would need the separate-head-size form
            }
            const int64_t hd    = E / H;
            const float   scale = nd.attr.getf("scale", (float) (1.0 / std::sqrt((double) hd)));
            const std::string base = nd.name;

            auto reshapeTranspose = [&](TensorId src, int64_t rows, const std::vector<int64_t> &perm, const char *tag) {
                TensorId r = newTensor(g, base + tag + std::string("r"));
                out.push_back(mkNode(OpType::Reshape, base + "#reshape" + tag,
                                     {src, addConstI64(g, base + tag + std::string("s"), {B, rows, H, hd})}, {r}));
                TensorId t  = newTensor(g, base + tag);
                Node     tr = mkNode(OpType::Transpose, base + "#transpose" + tag, {r}, {t});
                setAttrInts(tr, "perm", perm);
                out.push_back(std::move(tr));
                return t;
            };
            TensorId qT = reshapeTranspose(q, S, {0, 2, 1, 3}, "#q");  // [B,H,S,hd]
            TensorId kT = reshapeTranspose(k, T, {0, 2, 3, 1}, "#k");  // [B,H,hd,T]
            TensorId vT = reshapeTranspose(v, T, {0, 2, 1, 3}, "#v");  // [B,H,T,hd]

            TensorId scores = newTensor(g, base + "#scores");
            out.push_back(mkNode(OpType::MatMul, base + "#qk", {qT, kT}, {scores}));
            TensorId scaled = newTensor(g, base + "#scaled");
            out.push_back(mkBinary(BinaryType::Mul, base + "#scale",
                                   scores, addConstF32(g, base + "#scalec", scale), scaled));
            TensorId logits = scaled;
            if (attnBias != kNoTensor)
            {
                logits = newTensor(g, base + "#masked");
                out.push_back(mkNode(OpType::Add, base + "#mask", {scaled, attnBias}, {logits}));
            }
            TensorId probs = newTensor(g, base + "#probs");
            Node     sm    = mkNode(OpType::Softmax, base + "#softmax", {logits}, {probs});
            setAttrI(sm, "axis", -1);
            out.push_back(std::move(sm));
            TensorId ctx = newTensor(g, base + "#ctx");
            out.push_back(mkNode(OpType::MatMul, base + "#pv", {probs, vT}, {ctx}));
            TensorId ctxT = newTensor(g, base + "#ctxT");
            Node     tr   = mkNode(OpType::Transpose, base + "#transposeO", {ctx}, {ctxT});
            setAttrInts(tr, "perm", {0, 2, 1, 3});
            out.push_back(std::move(tr));
            out.push_back(mkNode(OpType::Reshape, base + "#reshapeO",
                                 {ctxT, addConstI64(g, base + "#so", {B, S, E})}, {nd.outputs[0]}));
            return true;
        }

        // The GroupQueryAttention runtime derivations shared by every layer: the new-token rotary
        // positions and the additive attention mask are functions of the op's seqlens_k input and
        // the static (S, P) bucket geometry alone — identical across the decoder's layers — so the
        // first expansion emits the derivation subgraph once and later layers reuse its tensors.
        struct GqaDerived {
            TensorId positions = kNoTensor; // [B, S] int64 new-token positions
            TensorId mask      = kNoTensor; // additive mask broadcastable over [B, H, S, T];
                                            // kNoTensor = all-zero (P == 0 and S == 1)
        };
        // Key: (seqlens_k tensor, S, P). The same tensor at a different geometry (a multi-graph
        // compile) derives separately.
        using GqaDerivedMap = std::map<std::tuple<TensorId, int64_t, int64_t>, GqaDerived>;

        // The additive-mask fill for a blocked position: the fp16 finite extreme, the engine's mask
        // convention (a saturating fp16 store keeps it finite instead of overflowing to -inf, and
        // exp() flushes it to zero in the fp32 softmax).
        constexpr float kMaskBlocked = -65504.0f;

        // com.microsoft::GroupQueryAttention -> primitive attention subgraph.
        //
        // Inputs: q [B,S,Hq*hd], k/v [B,S,Hkv*hd] (post-projection, packed), past_key/past_value
        // [B,Hkv,P,hd], seqlens_k (int32, B elements), total_sequence_length (unused — the bucket
        // geometry is static), cos_cache/sin_cache [maxPos, hd/2] when do_rotary. Outputs:
        // attention [B,S,Hq*hd], present key/value.
        //
        // ORT semantics reproduced (probe-validated against onnxruntime 1.27 CPU):
        //   past_len[b] = seqlens_k[b] + 1 - S   (valid past rows, and the first new token's
        //                                         absolute rotary position)
        //   new token i rotates at position past_len + i and attends past rows [0, past_len) plus
        //   new tokens [0, i] (always-causal); a padded past row j >= past_len is masked out.
        //
        // Present convention: present = Concat(past, new) — the new rows always land AFTER the full
        // P-row past block, the engine's KV-cache convention (the io_link fold and the example
        // drivers read the new rows from the tail; KvConcatFold rewrites the copy away at load).
        // onnxruntime instead writes the new rows at row past_len inside the present buffer. The two
        // layouts coincide exactly when the past is unpadded (past_len == P, every dynamic-shape ORT
        // run); under padding the attention VALUES still agree — the same key/value set is read —
        // and only the present row placement differs.
        //
        // Declined variants (node left in place, loud at plan): packed QKV (empty k/v), missing
        // past/seqlens, softcap, sliding window (local_window_size >= 0), smooth_softmax,
        // interleaved or partial rotary, non-divisible head grouping, extra trailing inputs
        // (position_ids / attention_bias forms), unresolved shapes.
        bool expandGqa(Graph &g, const Node &nd, std::vector<Node> &out, GqaDerivedMap &derived) {
            if (nd.inputs.size() < 7 || nd.outputs.empty() || nd.outputs[0] == kNoTensor)
            {
                return false;
            }
            for (size_t i = 9; i < nd.inputs.size(); ++i)
            {
                if (nd.inputs[i] != kNoTensor)
                {
                    return false; // a newer optional input (position_ids/attention_bias) is not covered
                }
            }
            const bool doRotary = nd.attr.geti("do_rotary", 0) != 0;
            if (nd.attr.geti("local_window_size", -1) != -1 || nd.attr.getf("softcap", 0.0f) != 0.0f ||
                nd.attr.geti("smooth_softmax", 0) != 0 || (doRotary && nd.attr.geti("rotary_interleaved", 0) != 0))
            {
                return false;
            }
            const int64_t Hq = nd.attr.geti("num_heads", 0), Hkv = nd.attr.geti("kv_num_heads", 0);
            if (Hq <= 0 || Hkv <= 0 || Hq % Hkv != 0)
            {
                return false;
            }
            const TensorId q = nd.inputs[0], k = nd.inputs[1], v = nd.inputs[2];
            const TensorId pastK = nd.inputs[3], pastV = nd.inputs[4], seqlens = nd.inputs[5];
            if (k == kNoTensor || v == kNoTensor || pastK == kNoTensor || pastV == kNoTensor || seqlens == kNoTensor)
            {
                return false; // packed-QKV and no-cache forms keep the unexpanded node
            }
            if (!shapeKnown(g, q) || !shapeKnown(g, k) || !shapeKnown(g, v) || !shapeKnown(g, pastK) || !shapeKnown(g, pastV) || !shapeKnown(g, seqlens))
            {
                return false;
            }
            const Shape &qs = g.desc(q).shape, &ks = g.desc(k).shape, &vs = g.desc(v).shape;
            const Shape &pks = g.desc(pastK).shape, &pvs = g.desc(pastV).shape;
            if (qs.size() != 3 || ks.size() != 3 || vs.size() != 3 || pks.size() != 4 || pvs.size() != 4 || pvs != pks)
            {
                return false;
            }
            const int64_t B = qs[0], S = qs[1], E = qs[2];
            if (E % Hq != 0)
            {
                return false;
            }
            const int64_t hd = E / Hq, P = pks[2], T = P + S, grp = Hq / Hkv;
            if (ks[0] != B || ks[1] != S || ks[2] != Hkv * hd || vs != ks)
            {
                return false;
            }
            if (pks[0] != B || pks[1] != Hkv || pks[3] != hd)
            {
                return false;
            }
            if (numElements(g.desc(seqlens).shape) != B)
            {
                return false;
            }
            TensorId cosC = kNoTensor, sinC = kNoTensor;
            if (doRotary)
            {
                cosC = optionalInput(nd, 7);
                sinC = optionalInput(nd, 8);
                if (cosC == kNoTensor || sinC == kNoTensor || !shapeKnown(g, cosC) || !shapeKnown(g, sinC))
                {
                    return false;
                }
                const Shape &cs = g.desc(cosC).shape;
                if (cs.size() != 2 || g.desc(sinC).shape != cs || cs[1] * 2 != hd)
                {
                    return false; // partial rotary (rotary_dim < head_dim) is not covered
                }
            }
            const float       scale = nd.attr.getf("scale", (float) (1.0 / std::sqrt((double) hd)));
            const std::string base  = nd.name;

            // --- shared positions/mask derivation (first layer emits, later layers reuse) ---
            GqaDerived &dv = derived[{seqlens, S, P}];
            if (dv.positions == kNoTensor)
            {
                // past_len[b] = seqlens_k[b] + 1 - S in exact int64 arithmetic ([B,1]); positions
                // [B,S] = past_len + [0..S). Values stay <= T, exact through the GPU compute-float
                // carry, and the position tensor itself is fp32-pinned at load (pinGatherIndexFp32).
                TensorId seqI = newTensorI64(g, base + "#seqlens_i64");
                Node     cast = mkNode(OpType::Cast, base + "#cast_seqlens", {seqlens}, {seqI});
                setAttrI(cast, "to", 7); // ONNX TensorProto INT64
                out.push_back(std::move(cast));
                TensorId seqB1 = newTensorI64(g, base + "#seqlens_b1");
                out.push_back(mkNode(OpType::Reshape, base + "#reshape_seqlens",
                                     {seqI, addConstI64(g, base + "#seqlens_shape", {B, 1})}, {seqB1}));
                TensorId pastLen = newTensorI64(g, base + "#past_len");
                out.push_back(mkBinary(BinaryType::Sub, base + "#sub_past_len", seqB1,
                                       addConstI64(g, base + "#s_minus_1", {S - 1}), pastLen));
                std::vector<int64_t> ar((size_t) S);
                for (int64_t i = 0; i < S; ++i)
                {
                    ar[(size_t) i] = i;
                }
                TensorId positions = newTensorI64(g, base + "#positions");
                out.push_back(mkNode(OpType::Add, base + "#add_positions",
                                     {pastLen, addConstI64Shaped(g, base + "#arange_s", {1, S}, ar)}, {positions}));
                dv.positions = positions;

                // Additive mask over the concat-form key axis [past(P) | new(S)]:
                //   past column j valid iff j < past_len (runtime, from seqlens_k)
                //   new column P+ii causally valid for query row i iff ii <= i (static triangle)
                // The two parts never overlap, so their sum never exceeds one kMaskBlocked.
                TensorId mask = kNoTensor;
                if (P > 0)
                {
                    std::vector<int64_t> cols((size_t) P);
                    for (int64_t j = 0; j < P; ++j)
                    {
                        cols[(size_t) j] = j;
                    }
                    TensorId colIdx   = addConstI64Shaped(g, base + "#past_cols", {1, 1, 1, P}, cols);
                    TensorId pastLen4 = newTensorI64(g, base + "#past_len4");
                    out.push_back(mkNode(OpType::Reshape, base + "#reshape_past_len",
                                         {pastLen, addConstI64(g, base + "#past_len4_shape", {B, 1, 1, 1})}, {pastLen4}));
                    TensorId colValid = newTensor(g, base + "#past_col_valid");
                    out.push_back(mkNode(OpType::Less, base + "#less_past_cols", {colIdx, pastLen4}, {colValid}));
                    TensorId pastMask = newTensor(g, base + "#past_mask");
                    out.push_back(mkNode(OpType::Where, base + "#where_past_mask",
                                         {colValid, addConstF32(g, base + "#mask_zero", 0.0f),
                                          addConstF32(g, base + "#mask_blocked", kMaskBlocked)},
                                         {pastMask}));
                    // Widen to the full T columns: the S new-token columns carry zero here (the
                    // causal triangle below owns them).
                    TensorId widened = newTensor(g, base + "#mask_t");
                    Node     cat     = mkNode(OpType::Concat, base + "#concat_mask",
                                              {pastMask, addConstF32Shaped(g, base + "#new_cols_zero", {B, 1, 1, S},
                                                                           std::vector<float>((size_t) (B * S), 0.0f))},
                                              {widened});
                    setAttrI(cat, "axis", 3);
                    out.push_back(std::move(cat));
                    mask = widened;
                }
                if (S > 1)
                {
                    std::vector<float> tri((size_t) (S * T), 0.0f);
                    for (int64_t i = 0; i < S; ++i)
                    {
                        for (int64_t j = P + i + 1; j < T; ++j)
                        {
                            tri[(size_t) (i * T + j)] = kMaskBlocked;
                        }
                    }
                    TensorId causal = addConstF32Shaped(g, base + "#causal_mask", {1, 1, S, T}, tri);
                    if (mask == kNoTensor)
                    {
                        mask = causal;
                    } else
                    {
                        TensorId combined = newTensor(g, base + "#mask_st");
                        out.push_back(mkNode(OpType::Add, base + "#add_causal", {mask, causal}, {combined}));
                        mask = combined;
                    }
                }
                dv.mask = mask;
            }

            // --- rotary embedding on q/k (rotate-half, fuseRope-collapsible) ---
            TensorId qRot = q, kRot = k;
            if (doRotary)
            {
                qRot = newTensor(g, base + "#q_rot");
                emitRotateHalf(g, out, base + "#q_rope", q, dv.positions, cosC, sinC, B, S, Hq, hd, qRot);
                kRot = newTensor(g, base + "#k_rot");
                emitRotateHalf(g, out, base + "#k_rope", k, dv.positions, cosC, sinC, B, S, Hkv, hd, kRot);
            }

            // --- head-major reshape/transpose: [B,S,H*hd] -> [B,H,S,hd] ---
            auto headMajor = [&](TensorId src, int64_t H, const char *tag) {
                TensorId r = newTensor(g, base + tag + std::string("_r"));
                out.push_back(mkNode(OpType::Reshape, base + "#reshape" + tag,
                                     {src, addConstI64(g, base + tag + std::string("_s"), {B, S, H, hd})}, {r}));
                TensorId t  = newTensor(g, base + tag);
                Node     tr = mkNode(OpType::Transpose, base + "#transpose" + tag, {r}, {t});
                setAttrInts(tr, "perm", {0, 2, 1, 3});
                out.push_back(std::move(tr));
                return t;
            };

            // --- presents: Concat(past, new) along the token axis ---
            auto emitPresent = [&](TensorId past, TensorId newRows, size_t outIdx, const char *tag) {
                TensorId present = nd.outputs.size() > outIdx && nd.outputs[outIdx] != kNoTensor
                                       ? nd.outputs[outIdx]
                                       : newTensor(g, base + tag + std::string("_present"));
                Node cat = mkNode(OpType::Concat, base + "#concat" + tag, {past, newRows}, {present});
                setAttrI(cat, "axis", 2);
                out.push_back(std::move(cat));
                return present;
            };
            TensorId presentK = emitPresent(pastK, headMajor(kRot, Hkv, "#k_new"), 1, "#k");
            TensorId presentV = emitPresent(pastV, headMajor(v, Hkv, "#v_new"), 2, "#v");

            // --- repeat_kv: broadcast the Hkv cache heads to the Hq query heads through an
            // Unsqueeze/Expand/Reshape chain, the form foldMatMulViews folds to stride-0 group
            // reads (kv head h serves query heads [h*grp, (h+1)*grp), the llama repeat_kv order) ---
            auto repeatKv = [&](TensorId cache, const char *tag) {
                TensorId u  = newTensor(g, base + tag + std::string("_u"));
                Node     uq = mkNode(OpType::Unsqueeze, base + "#unsqueeze" + tag, {cache}, {u});
                setAttrInts(uq, "axes", {2});
                out.push_back(std::move(uq));
                TensorId e = newTensor(g, base + tag + std::string("_e"));
                out.push_back(mkNode(OpType::Expand, base + "#expand" + tag,
                                     {u, addConstI64(g, base + tag + std::string("_es"), {B, Hkv, grp, T, hd})}, {e}));
                TensorId r = newTensor(g, base + tag);
                out.push_back(mkNode(OpType::Reshape, base + "#reshape" + tag,
                                     {e, addConstI64(g, base + tag + std::string("_rs"), {B, Hq, T, hd})}, {r}));
                return r;
            };
            TensorId kAll = repeatKv(presentK, "#k_all"); // [B,Hq,T,hd]
            TensorId vAll = repeatKv(presentV, "#v_all"); // [B,Hq,T,hd]
            TensorId kT   = newTensor(g, base + "#k_t");
            {
                Node tr = mkNode(OpType::Transpose, base + "#transpose_kt", {kAll}, {kT});
                setAttrInts(tr, "perm", {0, 1, 3, 2}); // [B,Hq,hd,T]
                out.push_back(std::move(tr));
            }

            // --- attention core: softmax(q K^T * scale + mask) V ---
            TensorId qT     = headMajor(qRot, Hq, "#q"); // [B,Hq,S,hd]
            TensorId scores = newTensor(g, base + "#scores");
            out.push_back(mkNode(OpType::MatMul, base + "#qk", {qT, kT}, {scores}));
            TensorId scaled = newTensor(g, base + "#scaled");
            out.push_back(mkBinary(BinaryType::Mul, base + "#scale",
                                   scores, addConstF32(g, base + "#scale_c", scale), scaled));
            TensorId logits = scaled;
            if (dv.mask != kNoTensor)
            {
                logits = newTensor(g, base + "#masked");
                out.push_back(mkNode(OpType::Add, base + "#mask_add", {scaled, dv.mask}, {logits}));
            }
            TensorId probs = newTensor(g, base + "#probs");
            Node     sm    = mkNode(OpType::Softmax, base + "#softmax", {logits}, {probs});
            setAttrI(sm, "axis", -1);
            out.push_back(std::move(sm));
            TensorId ctx = newTensor(g, base + "#ctx");
            out.push_back(mkNode(OpType::MatMul, base + "#pv", {probs, vAll}, {ctx}));
            TensorId ctxT = newTensor(g, base + "#ctx_t");
            Node     tr   = mkNode(OpType::Transpose, base + "#transpose_out", {ctx}, {ctxT});
            setAttrInts(tr, "perm", {0, 2, 1, 3});
            out.push_back(std::move(tr));
            out.push_back(mkNode(OpType::Reshape, base + "#reshape_out",
                                 {ctxT, addConstI64(g, base + "#out_shape", {B, S, E})}, {nd.outputs[0]}));
            return true;
        }

        // MatMulNBits (A, B_Q4, scales) with bits=4 and no zero_points/g_idx/bias -> a plain MatMul
        // whose weight initializer holds the int4 wq packing. ORT packs B as [N][K/bs][bs/2] bytes,
        // low nibble first, value = (q - 8) * scale — the -8 offset lands exactly on the signed
        // two's-complement nibble the wq layout stores, so the repack is index shuffling only.
        bool expandMatMulNBits(Graph &g, Node &nd) {
            const int64_t K = nd.attr.geti("K", 0), N = nd.attr.geti("N", 0);
            const int64_t bs = nd.attr.geti("block_size", 0);
            if (nd.attr.geti("bits", 4) != 4 || K <= 0 || N <= 0 || bs <= 0)
            {
                return false;
            }
            for (size_t i = 3; i < nd.inputs.size(); ++i)
            {
                if (nd.inputs[i] != kNoTensor)
                {
                    return false; // zero_points / g_idx / bias variants stay unexpanded
                }
            }
            const TensorId bq = nd.inputs[1], sc = nd.inputs[2];
            if (!g.isInitializer(bq) || !g.isInitializer(sc))
            {
                return false;
            }
            const int64_t kBlocks   = (K + bs - 1) / bs;
            const int64_t blobBytes = bs / 2;
            // The importer widens uint8 payloads to fp32 host values (0..255, exactly representable),
            // and scales arrive fp16 or fp32; initFloats reads both uniformly.
            const std::vector<float> bqf = initFloats(g, bq);
            const std::vector<float> scf = initFloats(g, sc);
            if ((int64_t) bqf.size() != N * kBlocks * blobBytes || (int64_t) scf.size() != N * kBlocks)
            {
                return false;
            }
            std::vector<int8_t> q((size_t) (K * N), 0);
            for (int64_t n = 0; n < N; ++n)
            {
                for (int64_t k = 0; k < K; ++k)
                {
                    const uint8_t byte = (uint8_t) bqf[(size_t) (n * kBlocks * blobBytes + (k / bs) * blobBytes + (k % bs) / 2)];
                    const int     nib  = (k & 1) ? (byte >> 4) : (byte & 0xF);
                    q[(size_t) (k * N + n)] = (int8_t) (nib - 8);
                }
            }
            std::vector<uint8_t>  packed = int4Pack(q, K, N);
            std::vector<uint16_t> scales((size_t) (kBlocks * N));
            for (int64_t n = 0; n < N; ++n)
            {
                for (int64_t b = 0; b < kBlocks; ++b)
                {
                    scales[(size_t) (b * N + n)] = floatToHalfSat(scf[(size_t) (n * kBlocks + b)]);
                }
            }
            const std::string wname = g.desc(bq).name.empty() ? nd.name + "#w" : g.desc(bq).name;
            TensorDesc wd;
            wd.name          = wname + "#i4w";
            wd.shape         = {K, N}; // the logical view; the payload is the packed bytes
            wd.dtype         = DType::Float16;
            wd.isInitializer = true;
            const TensorId weight = g.addTensor(wd);
            {
                HostBuffer hb;
                hb.bytes               = std::move(packed);
                g.initializers[weight] = std::move(hb);
            }
            TensorDesc sd;
            sd.name          = wname + "#i4s";
            sd.shape         = {kBlocks * N};
            sd.dtype         = DType::Float16;
            sd.isInitializer = true;
            const TensorId scaleId = g.addTensor(sd);
            {
                HostBuffer hb;
                std::vector<uint8_t> bytes((const uint8_t *) scales.data(), (const uint8_t *) scales.data() + scales.size() * 2);
                hb.bytes                = std::move(bytes);
                g.initializers[scaleId] = std::move(hb);
            }
            nd.type   = OpType::MatMul;
            nd.inputs = {nd.inputs[0], weight};
            nd.attr.map.clear();
            auto seti = [&](const char *key, int64_t v) { setAttrI(nd, key, v); };
            seti(kWq, kWqVersion);
            seti(kWqK, K);
            seti(kWqN, N);
            seti(kWqGroup, bs);
            seti(kWqNOut, 0);
            seti(kWqLayout, 0);
            seti(kWqScales, scaleId);
            // The B_Q4/scale source payloads are dead now; drop them so they are neither serialized
            // nor kept resident (pruneDeadInitializers would also catch them at the end of the
            // standard passes).
            g.initializers.erase(bq);
            g.desc(bq).isInitializer = false;
            g.initializers.erase(sc);
            g.desc(sc).isInitializer = false;
            return true;
        }

        // One expansion round: rewrite every contrib node whose inputs are ready. Emitted nodes
        // splice in at the original node's position, so the list stays topologically ordered.
        // `gqaDerived` persists across rounds — layers expand round by round as their input shapes
        // resolve, and every layer shares the first one's positions/mask derivation.
        int expandRound(Graph &g, GqaDerivedMap &gqaDerived) {
            int               expanded = 0;
            std::vector<Node> rebuilt;
            rebuilt.reserve(g.nodes.size());
            for (size_t i = 0; i < g.nodes.size(); ++i)
            {
                Node             &nd = g.nodes[i];
                std::vector<Node> repl;
                bool              did = false;
                switch (nd.type)
                {
                    case OpType::SimplifiedLayerNorm:
                        did = expandSimplifiedLayerNorm(g, nd);
                        break;
                    case OpType::SkipSimplifiedLayerNorm:
                    case OpType::SkipLayerNorm:
                        did = expandSkipNorm(g, nd, repl);
                        break;
                    case OpType::RotaryEmbedding:
                        did = expandRotary(g, nd, repl);
                        break;
                    case OpType::MultiHeadAttention:
                        did = expandMha(g, nd, repl);
                        break;
                    case OpType::GroupQueryAttention:
                        did = expandGqa(g, nd, repl, gqaDerived);
                        break;
                    case OpType::MatMulNBits:
                        did = expandMatMulNBits(g, nd);
                        break;
                    default:
                        break;
                }
                if (did)
                {
                    ++expanded;
                }
                if (repl.empty())
                {
                    rebuilt.push_back(std::move(nd));
                } else
                {
                    for (Node &r: repl)
                    {
                        rebuilt.push_back(std::move(r));
                    }
                }
            }
            g.nodes = std::move(rebuilt);
            return expanded;
        }

    } // namespace

    void lowerOrtContribOps(Graph &g) {
        auto hasContrib = [&] {
            for (const Node &nd: g.nodes)
            {
                switch (nd.type)
                {
                    case OpType::SimplifiedLayerNorm:
                    case OpType::SkipSimplifiedLayerNorm:
                    case OpType::SkipLayerNorm:
                    case OpType::RotaryEmbedding:
                    case OpType::MultiHeadAttention:
                    case OpType::GroupQueryAttention:
                    case OpType::MatMulNBits:
                        return true;
                    default:
                        break;
                }
            }
            return false;
        };
        if (!hasContrib())
        {
            return;
        }
        // Fixpoint with shape inference AND constant folding: expanding a layer's norm/rotary/
        // attention chain resolves the shapes the NEXT contrib node needs, and the shape-arithmetic
        // glue these exports compute at runtime (a repeat_kv's Shape/Concat/Reshape chain, the mask
        // reformat) only resolves through constFold — inferShapes alone cannot see a computed
        // Reshape target. Rounds are bounded by the longest contrib chain; the cap only guards a
        // degenerate graph.
        int           total = 0;
        GqaDerivedMap gqaDerived;
        for (int round = 0; round < 512; ++round)
        {
            const int n = expandRound(g, gqaDerived);
            total += n;
            if (n > 0)
            {
                inferShapes(g);
                continue;
            }
            if (constFold(g) == 0)
            {
                break; // neither expansion nor folding progressed: the rest is genuinely stuck
            }
            inferShapes(g);
        }
        int left = 0;
        for (const Node &nd: g.nodes)
        {
            switch (nd.type)
            {
                case OpType::SimplifiedLayerNorm:
                case OpType::SkipSimplifiedLayerNorm:
                case OpType::SkipLayerNorm:
                case OpType::RotaryEmbedding:
                case OpType::MultiHeadAttention:
                case OpType::MatMulNBits:
                case OpType::GroupQueryAttention:
                    ++left;
                    break;
                default:
                    break;
            }
        }
        VKNN_INFO << "lowerOrtContribOps: expanded " << total << " contrib op(s)" << (left ? (", " + std::to_string(left) + " left unexpanded (unsupported variant)") : "");
    }

} // namespace vknn
