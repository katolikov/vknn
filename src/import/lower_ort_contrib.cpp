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
//
// A node whose optional inputs/attributes fall outside the expanded form is LEFT IN PLACE: it keeps
// its real name through the support report ("MultiHeadAttention: ..." instead of "Unknown") and the
// plan fails loudly, never silently miscomputing. GroupQueryAttention (in-op RoPE + KV-cache
// management) is recognized at import but not yet expanded.
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
#include <string>
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

        TensorId addConstI64(Graph &g, const std::string &name, const std::vector<int64_t> &v) {
            TensorDesc d;
            d.name          = name;
            d.shape         = {(int64_t) v.size()};
            d.dtype         = DType::Int64;
            d.isInitializer = true;
            TensorId id     = g.addTensor(d);
            HostBuffer hb;
            hb.resizeElems((int64_t) v.size(), DType::Int64);
            std::memcpy(hb.i64(), v.data(), v.size() * 8);
            g.initializers[id] = std::move(hb);
            return id;
        }

        TensorId addConstF32(Graph &g, const std::string &name, float v) {
            TensorDesc d;
            d.name          = name;
            d.shape         = {1};
            d.isInitializer = true;
            TensorId id     = g.addTensor(d);
            HostBuffer hb;
            hb.resizeElems(1, DType::Float32);
            hb.f32()[0]        = v;
            g.initializers[id] = std::move(hb);
            return id;
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

        // RotaryEmbedding (x[B,S,H*head], position_ids[B,S] or [1,S], cos_cache/sin_cache
        // [maxPos, head/2], interleaved=0) -> rotate-half subgraph:
        //   r4 = reshape(x, [B,S,H,head]); x1/x2 = last-axis halves
        //   cosU/sinU = unsqueeze(gather(cache, position_ids), axis 2)   [B,S,1,head/2]
        //   y = reshape(concat(x1*cos - x2*sin, x1*sin + x2*cos), [B,S,H*head])
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
            const int64_t H = E / head;
            const std::string base = nd.name;

            TensorId r4 = newTensor(g, base + "#r4");
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
                                 {cat, addConstI64(g, base + "#s3", {B, S, E})}, {nd.outputs[0]}));
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
        int expandRound(Graph &g) {
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
        int total = 0;
        for (int round = 0; round < 512; ++round)
        {
            const int n = expandRound(g);
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
