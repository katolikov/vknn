// A synthetic with-past GQA decoder graph, built at any sequence length, for the host tests that
// exercise the multi-bucket decode driver flows (chunked prefill, speculative verification).
//
// The graph is small enough to run on the CPU byte oracle in milliseconds and shaped so the
// LOAD-TIME passes a real decoder hits engage in every bucket: embedding + position-embedding
// Gathers, q/k/v projections (q carrying the score scale), cache-concat present outputs, a
// Reshape/Expand/Reshape repeat_kv group expand that foldMatMulViews absorbs into operand-view
// strides, QK + (token-mask bias + causal constant) -> Softmax -> PV that fuseDecodeAttention fuses
// (at M == 1 in the decode bucket, at M > 1 in the widened ones), an output projection with
// residual, and the lm-head logits. A bucket set built from one DecoderSpec shares every weight
// byte-for-byte, so a multi-bucket save dedupes the pool.
//
// DecoderSpec::weightPhase shifts every weight generator, so two specs that differ only in the phase
// are two different models over the same token vocabulary — the draft/target pairing a speculative
// decode test needs.
#pragma once
#include "vknn/graph.h"
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace vknn { namespace synth {

    /// Geometry and weight identity of one synthetic decoder.
    struct DecoderSpec {
        int64_t vocab       = 48;
        int64_t qHeads      = 4;
        int64_t kvHeads     = 2;
        int64_t headDim     = 4;
        int64_t ctx         = 160;  ///< Cache slots C.
        int64_t maxPos      = 288;  ///< Position-table rows; covers a padded window's overhang past C.
        float   weightPhase = 0.0f; ///< Added to every weight generator's argument: a different model.

        int64_t hidden() const noexcept {
            return qHeads * headDim;
        }
        int64_t kvWidth() const noexcept {
            return kvHeads * headDim;
        }
    };

    /// Additive mask fill; exp() underflows it to exactly 0.
    inline constexpr float kMaskBias = -1e9f;

    inline TensorId addIn(Graph &g, const std::string &name, Shape s, DType dt) {
        TensorDesc d;
        d.name      = name;
        d.shape     = std::move(s);
        d.dtype     = dt;
        d.isInput   = true;
        TensorId id = g.addTensor(d);
        g.inputs.push_back(id);
        return id;
    }
    inline TensorId addOut(Graph &g, const std::string &name) {
        TensorDesc d;
        d.name      = name;
        d.isOutput  = true;
        TensorId id = g.addTensor(d);
        g.outputs.push_back(id);
        return id;
    }
    inline TensorId addTemp(Graph &g, const std::string &name) {
        TensorDesc d;
        d.name = name;
        return g.addTensor(d);
    }
    inline TensorId addF32(Graph &g, const std::string &name, Shape s, const std::vector<float> &v) {
        TensorDesc d;
        d.name          = name;
        d.shape         = std::move(s);
        d.isInitializer = true;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) v.size(), DType::Float32);
        std::memcpy(hb.f32(), v.data(), v.size() * 4);
        g.initializers[id] = hb;
        return id;
    }
    inline TensorId addI64(Graph &g, const std::string &name, const std::vector<int64_t> &v) {
        TensorDesc d;
        d.name          = name;
        d.shape         = {(int64_t) v.size()};
        d.dtype         = DType::Int64;
        d.isInitializer = true;
        TensorId   id   = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) v.size(), DType::Int64);
        std::memcpy(hb.i64(), v.data(), v.size() * 8);
        g.initializers[id] = hb;
        return id;
    }
    inline Node *addNode(Graph &g, OpType t, const std::string &name, std::vector<TensorId> ins, TensorId out) {
        Node n;
        n.type    = t;
        n.name    = name;
        n.inputs  = std::move(ins);
        n.outputs = {out};
        g.nodes.push_back(n);
        return &g.nodes.back();
    }
    inline void setAttrI(Node *n, const char *k, int64_t v) {
        Attr a;
        a.kind         = Attr::Int;
        a.i            = v;
        n->attr.map[k] = a;
    }
    inline void setAttrInts(Node *n, const char *k, std::vector<int64_t> v) {
        Attr a;
        a.kind         = Attr::Ints;
        a.ints         = std::move(v);
        n->attr.map[k] = a;
    }

    inline std::vector<float> sinFill(int64_t n, float freq, float amp, float phase) {
        std::vector<float> v((size_t) n);
        for (int64_t i = 0; i < n; ++i)
        {
            v[(size_t) i] = std::sin(freq * (float) i + phase) * amp;
        }
        return v;
    }

    /// The decoder graph at sequence length S. `withPositionIds` false drops the position_ids
    /// input (the GroupQueryAttention-style export that derives position from the mask), which
    /// the widened-bucket plan refuses.
    inline Graph buildDecoder(const DecoderSpec &spec, int64_t S, bool withPositionIds = true) {
        const int64_t C  = spec.ctx;
        const int64_t T  = C + S;
        const float   ph = spec.weightPhase;
        Graph         g;
        TensorId      ids  = addIn(g, "input_ids", {1, S}, DType::Int64);
        TensorId      pos  = withPositionIds ? addIn(g, "position_ids", {1, S}, DType::Int64) : kNoTensor;
        TensorId      am   = addIn(g, "attention_mask", {1, T}, DType::Int64);
        TensorId      pk   = addIn(g, "past_key_values.0.key", {1, spec.kvHeads, C, spec.headDim}, DType::Float32);
        TensorId      pv   = addIn(g, "past_key_values.0.value", {1, spec.kvHeads, C, spec.headDim}, DType::Float32);
        TensorId      embT = addF32(g, "embed_table", {spec.vocab, spec.hidden()}, sinFill(spec.vocab * spec.hidden(), 0.13f, 0.8f, ph));
        TensorId      posT = addF32(g, "pos_table", {spec.maxPos, spec.hidden()}, sinFill(spec.maxPos * spec.hidden(), 0.29f, 0.3f, ph));
        TensorId      wq   = addF32(g, "w_q", {spec.hidden(), spec.hidden()}, sinFill(spec.hidden() * spec.hidden(), 0.53f, 0.5f, ph));
        TensorId      wk   = addF32(g, "w_k", {spec.hidden(), spec.kvWidth()}, sinFill(spec.hidden() * spec.kvWidth(), 0.71f, 0.5f, ph));
        TensorId      wv   = addF32(g, "w_v", {spec.hidden(), spec.kvWidth()}, sinFill(spec.hidden() * spec.kvWidth(), 0.37f, 0.6f, ph));
        TensorId      wo   = addF32(g, "w_o", {spec.hidden(), spec.hidden()}, sinFill(spec.hidden() * spec.hidden(), 0.61f, 0.4f, ph));
        TensorId      lm   = addF32(g, "lm_head", {spec.hidden(), spec.vocab}, sinFill(spec.hidden() * spec.vocab, 0.43f, 0.7f, ph));

        TensorId emb = addTemp(g, "emb");
        setAttrI(addNode(g, OpType::Gather, "embed", {embT, ids}, emb), "axis", 0);
        TensorId h0 = emb;
        if (withPositionIds)
        {
            TensorId posEmb = addTemp(g, "pos_emb");
            setAttrI(addNode(g, OpType::Gather, "pos_embed", {posT, pos}, posEmb), "axis", 0);
            h0 = addTemp(g, "h0");
            addNode(g, OpType::Add, "embed_add", {emb, posEmb}, h0);
        }

        auto project = [&](const char *tag, TensorId w, int64_t heads) {
            TensorId flat = addTemp(g, std::string(tag) + "_flat");
            addNode(g, OpType::MatMul, std::string(tag) + "_proj", {h0, w}, flat);
            TensorId re = addTemp(g, std::string(tag) + "_re");
            addNode(g, OpType::Reshape, std::string(tag) + "_reshape", {flat, addI64(g, std::string(tag) + "_rs", {1, S, heads, spec.headDim})}, re);
            TensorId tr = addTemp(g, std::string(tag) + "_tr");
            setAttrInts(addNode(g, OpType::Transpose, std::string(tag) + "_transpose", {re}, tr), "perm", {0, 2, 1, 3});
            return tr;
        };
        TensorId qT = project("q", wq, spec.qHeads);  // [1, Hq, S, hd]
        TensorId kT = project("k", wk, spec.kvHeads); // [1, KV, S, hd]
        TensorId vT = project("v", wv, spec.kvHeads); // [1, KV, S, hd]

        // Cache-concat present outputs: past rows then the produced rows.
        TensorId presK = addOut(g, "present.0.key");
        setAttrI(addNode(g, OpType::Concat, "present_key", {pk, kT}, presK), "axis", 2);
        TensorId presV = addOut(g, "present.0.value");
        setAttrI(addNode(g, OpType::Concat, "present_value", {pv, vT}, presV), "axis", 2);

        // repeat_kv as a transformers export emits it: [1,KV,T,hd] -> Reshape [1,KV,1,T,hd] ->
        // Expand [1,KV,G,T,hd] -> Reshape [1,Hq,T,hd], so q head h reads kv head h / G.
        // foldMatMulViews absorbs this chain into the operand-view strides fuseDecodeAttention
        // consumes; a chain it cannot absorb leaves the attention unfused and the buckets running
        // a graph form no device ever runs.
        auto groupExpand = [&](const char *tag, TensorId cache) {
            const int64_t group = spec.qHeads / spec.kvHeads;
            TensorId      r1    = addTemp(g, std::string(tag) + "_g1");
            addNode(g, OpType::Reshape, std::string(tag) + "_greshape1", {cache, addI64(g, std::string(tag) + "_gs1", {1, spec.kvHeads, 1, T, spec.headDim})}, r1);
            TensorId ex = addTemp(g, std::string(tag) + "_gex");
            addNode(g, OpType::Expand, std::string(tag) + "_gexpand", {r1, addI64(g, std::string(tag) + "_gs2", {1, spec.kvHeads, group, T, spec.headDim})}, ex);
            TensorId r2 = addTemp(g, std::string(tag) + "_g2");
            addNode(g, OpType::Reshape, std::string(tag) + "_greshape2", {ex, addI64(g, std::string(tag) + "_gs3", {1, spec.qHeads, T, spec.headDim})}, r2);
            return r2;
        };
        TensorId kAll = groupExpand("k", presK); // [1, Hq, T, hd]
        TensorId vAll = groupExpand("v", presV);

        TensorId kQk = addTemp(g, "k_qk");
        setAttrInts(addNode(g, OpType::Transpose, "k_qk_transpose", {kAll}, kQk), "perm", {0, 1, 3, 2});
        // The score scale rides on q (the transformers `q * scaling` form), so the only node
        // between the QK MatMul and the Softmax is the additive mask — the standalone-Add shape
        // fuseDecodeAttention absorbs. A separate scale node fuses with the mask into one
        // pointwise unit, which the pass refuses, leaving the chain unfused.
        TensorId qS     = addTemp(g, "q_scaled");
        Node    *sc     = addNode(g, OpType::Binary, "q_scale", {qT, addF32(g, "scale_c", {1}, {0.5f})}, qS);
        sc->subOp       = (int32_t) BinaryType::Mul;
        TensorId scaled = addTemp(g, "scaled");
        addNode(g, OpType::MatMul, "qk", {qS, kQk}, scaled); // [1, Hq, S, T]

        // Additive mask operand: (token_mask - 1) * 1e9 broadcast over rows, plus the causal
        // constant for the S window columns (row i attends cache columns and window columns <= i).
        TensorId maskF = addTemp(g, "mask_f");
        setAttrI(addNode(g, OpType::Cast, "mask_cast", {am}, maskF), "to", 1); // ONNX FLOAT
        TensorId maskSub = addTemp(g, "mask_sub");
        Node    *ms      = addNode(g, OpType::Binary, "mask_sub_node", {maskF, addF32(g, "one_c", {1}, {1.0f})}, maskSub);
        ms->subOp        = (int32_t) BinaryType::Sub;
        TensorId maskMul = addTemp(g, "mask_bias");
        Node    *mm      = addNode(g, OpType::Binary, "mask_bias_node", {maskSub, addF32(g, "big_c", {1}, {-kMaskBias})}, maskMul);
        mm->subOp        = (int32_t) BinaryType::Mul; // (m-1) * 1e9 = -1e9 on masked columns
        TensorId mask4   = addTemp(g, "mask_4d");
        addNode(g, OpType::Reshape, "mask_reshape", {maskMul, addI64(g, "mask_rs", {1, 1, 1, T})}, mask4);
        std::vector<float> causal((size_t) (S * T), 0.f);
        for (int64_t i = 0; i < S; ++i)
        {
            for (int64_t j = 0; j < T; ++j)
            {
                causal[(size_t) (i * T + j)] = (j < C || j - C <= i) ? 0.f : kMaskBias;
            }
        }
        TensorId combined = addTemp(g, "mask_combined");
        addNode(g, OpType::Add, "mask_combine", {mask4, addF32(g, "causal_c", {1, 1, S, T}, causal)}, combined);
        TensorId maskedScores = addTemp(g, "masked_scores");
        addNode(g, OpType::Add, "mask_apply", {scaled, combined}, maskedScores);

        TensorId probs = addTemp(g, "probs");
        setAttrI(addNode(g, OpType::Softmax, "softmax", {maskedScores}, probs), "axis", -1);
        TensorId ctx = addTemp(g, "ctx");
        addNode(g, OpType::MatMul, "pv_matmul", {probs, vAll}, ctx); // [1, Hq, S, hd]
        TensorId ctxT = addTemp(g, "ctx_t");
        setAttrInts(addNode(g, OpType::Transpose, "ctx_transpose", {ctx}, ctxT), "perm", {0, 2, 1, 3});
        TensorId ctxR = addTemp(g, "ctx_r");
        addNode(g, OpType::Reshape, "ctx_reshape", {ctxT, addI64(g, "ctx_rs", {1, S, spec.hidden()})}, ctxR);
        TensorId attnOut = addTemp(g, "attn_out");
        addNode(g, OpType::MatMul, "o_proj", {ctxR, wo}, attnOut);
        TensorId h1 = addTemp(g, "h1");
        addNode(g, OpType::Add, "residual", {h0, attnOut}, h1);
        TensorId logits = addOut(g, "logits");
        addNode(g, OpType::MatMul, "lm_head_matmul", {h1, lm}, logits);
        return g;
    }

}} // namespace vknn::synth
