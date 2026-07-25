// Chunked prefill (llm.npu-style, arXiv 2407.05858): a with-past decoder's prompt runs as
// fixed-size chunk passes over the cached KV instead of one whole-window pad-to-bucket forward.
// These host tests pin the two halves of the feature on the CPU byte oracle:
//   - planChunkPrefillBucket derives the automatic chunk bucket's shape set from a decode bucket
//     (and refuses models it cannot serve),
//   - the chunked-resident prefill flow (engine-resident chunk KV + kvFoldRowRanges block folds +
//     one readResident materialization, exactly the vknn_chat flow) produces a BYTE-IDENTICAL
//     cache, logits row, and greedy token stream to the legacy whole-window host-flow prefill,
//     across prompts shorter than one chunk, an exact chunk multiple, one-over a multiple, and a
//     second conversation turn over a populated cache.
// The synthetic model is a small GQA decoder (grouped KV heads, additive causal mask built
// in-graph from the token mask, learned position embedding driven by position_ids, cache-concat
// present outputs) built at each bucket's sequence length, saved as one multi-bucket .vxm. Its
// attention chain is shaped so the LOAD-TIME passes a real decoder hits engage in every bucket:
// foldMatMulViews absorbs the repeat_kv expand into operand-view strides, fuseDecodeAttention
// fuses the chain (at M == 1 in the decode bucket, at M > 1 in the chunk and whole-window ones),
// and foldFusedAttentionKvConcat turns the present outputs into the rows-only convention — so the
// chunked flow is compared against the whole-window flow on the graph form the device runs.
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/io_link.h"
#include "vknn/session.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace vknn;

namespace {

    constexpr int64_t kVocab        = 48;
    constexpr int64_t kQHeads       = 4;
    constexpr int64_t kKvHeads      = 2;
    constexpr int64_t kHeadDim      = 4;
    constexpr int64_t kHidden       = kQHeads * kHeadDim;
    constexpr int64_t kKvWidth      = kKvHeads * kHeadDim;
    constexpr int64_t kCtx          = 160; // cache slots C; >= kChunkPrefillTokens so the plan emits
    constexpr int64_t kLegacyWindow = 96;  // whole-window prefill bucket (not a chunk multiple)
    constexpr int64_t kMaxPos       = kCtx + 128; // position table rows; covers pad-row overhang
    constexpr int64_t kEosId        = 2;          // in-vocab pad id for the chunked path
    constexpr float   kMaskBias     = -1e9f;      // additive mask fill; exp() underflows to exactly 0
    constexpr int     kDecodeSteps  = 8;

    TensorId addIn(Graph &g, const std::string &name, Shape s, DType dt) {
        TensorDesc d;
        d.name    = name;
        d.shape   = std::move(s);
        d.dtype   = dt;
        d.isInput = true;
        TensorId id = g.addTensor(d);
        g.inputs.push_back(id);
        return id;
    }
    TensorId addOut(Graph &g, const std::string &name) {
        TensorDesc d;
        d.name     = name;
        d.isOutput = true;
        TensorId id = g.addTensor(d);
        g.outputs.push_back(id);
        return id;
    }
    TensorId addTemp(Graph &g, const std::string &name) {
        TensorDesc d;
        d.name = name;
        return g.addTensor(d);
    }
    TensorId addF32(Graph &g, const std::string &name, Shape s, const std::vector<float> &v) {
        TensorDesc d;
        d.name          = name;
        d.shape         = std::move(s);
        d.isInitializer = true;
        TensorId id     = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) v.size(), DType::Float32);
        std::memcpy(hb.f32(), v.data(), v.size() * 4);
        g.initializers[id] = hb;
        return id;
    }
    TensorId addI64(Graph &g, const std::string &name, const std::vector<int64_t> &v) {
        TensorDesc d;
        d.name          = name;
        d.shape         = {(int64_t) v.size()};
        d.dtype         = DType::Int64;
        d.isInitializer = true;
        TensorId id     = g.addTensor(d);
        HostBuffer hb;
        hb.resizeElems((int64_t) v.size(), DType::Int64);
        std::memcpy(hb.i64(), v.data(), v.size() * 8);
        g.initializers[id] = hb;
        return id;
    }
    Node *addNode(Graph &g, OpType t, const std::string &name, std::vector<TensorId> ins, TensorId out) {
        Node n;
        n.type    = t;
        n.name    = name;
        n.inputs  = std::move(ins);
        n.outputs = {out};
        g.nodes.push_back(n);
        return &g.nodes.back();
    }
    void setAttrI(Node *n, const char *k, int64_t v) {
        Attr a;
        a.kind         = Attr::Int;
        a.i            = v;
        n->attr.map[k] = a;
    }
    void setAttrInts(Node *n, const char *k, std::vector<int64_t> v) {
        Attr a;
        a.kind         = Attr::Ints;
        a.ints         = std::move(v);
        n->attr.map[k] = a;
    }

    std::vector<float> sinFill(int64_t n, float freq, float amp) {
        std::vector<float> v((size_t) n);
        for (int64_t i = 0; i < n; ++i)
        {
            v[(size_t) i] = std::sin(freq * (float) i) * amp;
        }
        return v;
    }

    // The synthetic with-past GQA decoder at sequence length S: embedding + position-embedding
    // Gathers, q/k/v projections (q carrying the score scale), cache-concat present outputs, a
    // Reshape/Expand/Reshape repeat_kv group expand, QK + (token-mask bias + causal constant) ->
    // Softmax -> PV, output projection with residual, and the lm-head logits. All weights are deterministic, shared byte-for-byte
    // across the per-S builds so the multi-bucket save dedupes them.
    Graph buildDecoder(int64_t S, int64_t C = kCtx, bool withPositionIds = true) {
        const int64_t T = C + S;
        Graph         g;
        TensorId      ids  = addIn(g, "input_ids", {1, S}, DType::Int64);
        TensorId      pos  = withPositionIds ? addIn(g, "position_ids", {1, S}, DType::Int64) : kNoTensor;
        TensorId      am   = addIn(g, "attention_mask", {1, T}, DType::Int64);
        TensorId      pk   = addIn(g, "past_key_values.0.key", {1, kKvHeads, C, kHeadDim}, DType::Float32);
        TensorId      pv   = addIn(g, "past_key_values.0.value", {1, kKvHeads, C, kHeadDim}, DType::Float32);
        TensorId      embT = addF32(g, "embed_table", {kVocab, kHidden}, sinFill(kVocab * kHidden, 0.13f, 0.8f));
        TensorId      posT = addF32(g, "pos_table", {kMaxPos, kHidden}, sinFill(kMaxPos * kHidden, 0.29f, 0.3f));
        TensorId      wq   = addF32(g, "w_q", {kHidden, kHidden}, sinFill(kHidden * kHidden, 0.53f, 0.5f));
        TensorId      wk   = addF32(g, "w_k", {kHidden, kKvWidth}, sinFill(kHidden * kKvWidth, 0.71f, 0.5f));
        TensorId      wv   = addF32(g, "w_v", {kHidden, kKvWidth}, sinFill(kHidden * kKvWidth, 0.37f, 0.6f));
        TensorId      wo   = addF32(g, "w_o", {kHidden, kHidden}, sinFill(kHidden * kHidden, 0.61f, 0.4f));
        TensorId      lm   = addF32(g, "lm_head", {kHidden, kVocab}, sinFill(kHidden * kVocab, 0.43f, 0.7f));

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
            addNode(g, OpType::Reshape, std::string(tag) + "_reshape", {flat, addI64(g, std::string(tag) + "_rs", {1, S, heads, kHeadDim})}, re);
            TensorId tr = addTemp(g, std::string(tag) + "_tr");
            setAttrInts(addNode(g, OpType::Transpose, std::string(tag) + "_transpose", {re}, tr), "perm", {0, 2, 1, 3});
            return tr;
        };
        TensorId qT = project("q", wq, kQHeads);   // [1, Hq, S, hd]
        TensorId kT = project("k", wk, kKvHeads);  // [1, KV, S, hd]
        TensorId vT = project("v", wv, kKvHeads);  // [1, KV, S, hd]

        // Cache-concat present outputs: past rows then the produced rows.
        TensorId presK = addOut(g, "present.0.key");
        setAttrI(addNode(g, OpType::Concat, "present_key", {pk, kT}, presK), "axis", 2);
        TensorId presV = addOut(g, "present.0.value");
        setAttrI(addNode(g, OpType::Concat, "present_value", {pv, vT}, presV), "axis", 2);

        // repeat_kv as a transformers export emits it: [1,KV,T,hd] -> Reshape [1,KV,1,T,hd] ->
        // Expand [1,KV,G,T,hd] -> Reshape [1,Hq,T,hd], so q head h reads kv head h / G
        // (G = Hq / KV = 2). foldMatMulViews absorbs this chain into the operand-view strides
        // fuseDecodeAttention consumes; a chain it cannot absorb leaves the attention unfused and
        // the buckets running a graph form no device ever runs.
        auto groupExpand = [&](const char *tag, TensorId cache) {
            const int64_t group = kQHeads / kKvHeads;
            TensorId      r1    = addTemp(g, std::string(tag) + "_g1");
            addNode(g, OpType::Reshape, std::string(tag) + "_greshape1", {cache, addI64(g, std::string(tag) + "_gs1", {1, kKvHeads, 1, T, kHeadDim})}, r1);
            TensorId ex = addTemp(g, std::string(tag) + "_gex");
            addNode(g, OpType::Expand, std::string(tag) + "_gexpand", {r1, addI64(g, std::string(tag) + "_gs2", {1, kKvHeads, group, T, kHeadDim})}, ex);
            TensorId r2 = addTemp(g, std::string(tag) + "_g2");
            addNode(g, OpType::Reshape, std::string(tag) + "_greshape2", {ex, addI64(g, std::string(tag) + "_gs3", {1, kQHeads, T, kHeadDim})}, r2);
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
        TensorId qS = addTemp(g, "q_scaled");
        Node    *sc = addNode(g, OpType::Binary, "q_scale", {qT, addF32(g, "scale_c", {1}, {0.5f})}, qS);
        sc->subOp   = (int32_t) BinaryType::Mul;
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
        TensorId mask4 = addTemp(g, "mask_4d");
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
        addNode(g, OpType::Reshape, "ctx_reshape", {ctxT, addI64(g, "ctx_rs", {1, S, kHidden})}, ctxR);
        TensorId attnOut = addTemp(g, "attn_out");
        addNode(g, OpType::MatMul, "o_proj", {ctxR, wo}, attnOut);
        TensorId h1 = addTemp(g, "h1");
        addNode(g, OpType::Add, "residual", {h0, attnOut}, h1);
        TensorId logits = addOut(g, "logits");
        addNode(g, OpType::MatMul, "lm_head_matmul", {h1, lm}, logits);
        return g;
    }

    Graph compiled(int64_t S, int64_t C = kCtx, bool withPositionIds = true) {
        Graph g = buildDecoder(S, C, withPositionIds);
        runStandardPasses(g);
        return g;
    }

    // ---- host-side driver state ---------------------------------------------------------------

    struct Flow {
        std::unique_ptr<Session> sess;
        std::vector<float>       pastK, pastV;     // host cache [KV, C, hd]
        int                      p = 0;            // absolute position
        std::vector<float>       logitsRow;        // last real token's logits row
        std::vector<int64_t>     tokens;           // full greedy stream across all turns
    };

    IOTensor i64Tensor(const char *name, Shape s, const std::vector<int64_t> &v) {
        IOTensor t;
        t.name  = name;
        t.shape = std::move(s);
        t.dtype = DType::Int64;
        t.data.resize(v.size() * 8);
        std::memcpy(t.data.data(), v.data(), t.data.size());
        return t;
    }
    IOTensor f32Tensor(const char *name, Shape s, const std::vector<float> &v) {
        IOTensor t;
        t.name  = name;
        t.shape = std::move(s);
        t.dtype = DType::Float32;
        t.data.resize(v.size() * 4);
        std::memcpy(t.data.data(), v.data(), t.data.size());
        return t;
    }
    const IOTensor *outByName(const std::vector<IOTensor> &outs, const char *name) {
        for (const IOTensor &o: outs)
        {
            if (o.name == name)
            {
                return &o;
            }
        }
        return nullptr;
    }

    // One host-cache-flow pass of `len` tokens through the [1, window] bucket: bind ids/mask/
    // positions and the full cache, fold the produced rows (after the past block) back, keep the
    // last real token's logits row. The vknn_chat whole-window prefillPass at test scale; window
    // == 1 with the current token is the decode step.
    bool hostPass(Flow &f, int64_t window, const int64_t *toks, int len, int64_t padId) {
        const int64_t        T = kCtx + window;
        std::vector<int64_t> ids((size_t) window), pos((size_t) window), mask((size_t) T, 0);
        for (int64_t t = 0; t < window; ++t)
        {
            ids[(size_t) t] = t < len ? toks[t] : padId;
            pos[(size_t) t] = f.p + t;
        }
        for (int64_t j = 0; j < f.p && j < kCtx; ++j)
        {
            mask[(size_t) j] = 1;
        }
        for (int t = 0; t < len; ++t)
        {
            mask[(size_t) (kCtx + t)] = 1;
        }
        std::vector<IOTensor> ins {
            i64Tensor("input_ids", {1, window}, ids),
            i64Tensor("position_ids", {1, window}, pos),
            i64Tensor("attention_mask", {1, T}, mask),
            f32Tensor("past_key_values.0.key", {1, kKvHeads, kCtx, kHeadDim}, f.pastK),
            f32Tensor("past_key_values.0.value", {1, kKvHeads, kCtx, kHeadDim}, f.pastV),
        };
        std::vector<IOTensor> outs;
        if (f.sess->run(ins, outs) != Status::Ok)
        {
            return false;
        }
        const IOTensor *presK  = outByName(outs, "present.0.key");
        const IOTensor *presV  = outByName(outs, "present.0.value");
        const IOTensor *logits = outByName(outs, "logits");
        if (!presK || !presV || !logits)
        {
            return false;
        }
        // The produced rows sit AFTER any past block, exactly as vknn_chat's prefillPass reads
        // them: kCtx + window under the cache-concat present, 0 under the rows-only present the
        // split-KV fold produces (Hint::KvConcatFold). The offset comes from the reported row
        // count, so the pass serves both conventions.
        const int64_t presRows  = presK->shape[2];
        const int64_t newRowsAt = presRows - window;
        for (int part = 0; part < 2; ++part)
        {
            const float *src = (part ? presV : presK)->f32();
            float       *dst = (part ? f.pastV : f.pastK).data();
            for (int64_t h = 0; h < kKvHeads; ++h)
            {
                std::memcpy(dst + (h * kCtx + f.p) * kHeadDim, src + (h * presRows + newRowsAt) * kHeadDim, (size_t) len * kHeadDim * 4);
            }
        }
        f.p += len;
        const float *row = logits->f32() + (size_t) (len - 1) * kVocab;
        f.logitsRow.assign(row, row + kVocab);
        return true;
    }

    // The chunked-resident prefill, exactly the vknn_chat flow: link present -> past inside the
    // chunk bucket, run ceil(T/chunk) passes whose ranges fold the previous chunk's row block
    // (kvFoldRowRanges), first pass re-seeding the resident cache from the host buffers, then one
    // readResident materialization (resident past + the last chunk's pending rows) and clearLinks.
    bool chunkedPrefill(Flow &f, size_t chunkBucket, int64_t chunkS, const std::vector<int64_t> &prompt) {
        const int64_t T = kCtx + chunkS;
        const char   *presNames[2] = {"present.0.key", "present.0.value"};
        const char   *pastNames[2] = {"past_key_values.0.key", "past_key_values.0.value"};
        for (int part = 0; part < 2; ++part)
        {
            if (f.sess->linkOutputToInput(chunkBucket, presNames[part], pastNames[part], {}) != Status::Ok)
            {
                return false;
            }
        }
        int64_t chunkPresRows = 0;
        {
            const std::vector<IOInfo> outsInfo = f.sess->outputInfo(chunkBucket);
            for (const IOInfo &o: outsInfo)
            {
                if (o.name == "present.0.key")
                {
                    chunkPresRows = o.shape[2];
                }
            }
        }
        const int64_t newRowsAt = chunkPresRows - chunkS;
        int64_t       prevStart = -1, prevLen = 0;
        bool          first = true;
        size_t        done  = 0;
        while (done < prompt.size())
        {
            const int len = (int) std::min<size_t>(prompt.size() - done, (size_t) chunkS);
            EXPECT_LE(f.p + len, kCtx);
            const std::vector<LinkRange> ranges = kvFoldRowRanges(kKvHeads, chunkPresRows, kCtx, kHeadDim, newRowsAt, prevLen > 0 ? prevStart : -1, prevLen);
            for (int part = 0; part < 2; ++part)
            {
                if (f.sess->linkOutputToInput(chunkBucket, presNames[part], pastNames[part], ranges) != Status::Ok)
                {
                    return false;
                }
            }
            std::vector<int64_t> ids((size_t) chunkS), pos((size_t) chunkS), mask((size_t) T, 0);
            for (int64_t t = 0; t < chunkS; ++t)
            {
                ids[(size_t) t] = (int) t < len ? prompt[done + (size_t) t] : kEosId;
                pos[(size_t) t] = f.p + t;
            }
            for (int64_t j = 0; j < f.p && j < kCtx; ++j)
            {
                mask[(size_t) j] = 1;
            }
            for (int t = 0; t < len; ++t)
            {
                mask[(size_t) (kCtx + t)] = 1;
            }
            std::vector<IOTensor> ins {
                i64Tensor("input_ids", {1, chunkS}, ids),
                i64Tensor("position_ids", {1, chunkS}, pos),
                i64Tensor("attention_mask", {1, T}, mask),
            };
            if (first)
            {
                // Re-seed the resident cache from the host state (empty ranges above suppress a
                // stale fold, mirroring the driver's first-pass bind).
                ins.push_back(f32Tensor("past_key_values.0.key", {1, kKvHeads, kCtx, kHeadDim}, f.pastK));
                ins.push_back(f32Tensor("past_key_values.0.value", {1, kKvHeads, kCtx, kHeadDim}, f.pastV));
            }
            std::vector<IOTensor> outs;
            if (f.sess->run(ins, outs) != Status::Ok)
            {
                return false;
            }
            // Linked present outputs return no data; the logits stay a plain download.
            const IOTensor *presK = outByName(outs, "present.0.key");
            EXPECT_TRUE(presK && presK->data.empty());
            const IOTensor *logits = outByName(outs, "logits");
            if (!logits)
            {
                return false;
            }
            const float *row = logits->f32() + (size_t) (len - 1) * kVocab;
            f.logitsRow.assign(row, row + kVocab);
            prevStart = f.p;
            prevLen   = len;
            first     = false;
            f.p += len;
            done += (size_t) len;
        }
        // Materialize: resident past (all folds applied) + the last chunk's still-pending rows.
        for (int part = 0; part < 2; ++part)
        {
            IOTensor resident;
            if (f.sess->readResident(pastNames[part], resident) != Status::Ok)
            {
                return false;
            }
            std::vector<float> &host = part ? f.pastV : f.pastK;
            EXPECT_EQ(resident.data.size(), host.size() * 4);
            std::memcpy(host.data(), resident.data.data(), resident.data.size());
            IOTensor present;
            if (f.sess->readResident(presNames[part], present) != Status::Ok)
            {
                return false;
            }
            const float *src = reinterpret_cast<const float *>(present.data.data());
            for (int64_t h = 0; h < kKvHeads; ++h)
            {
                std::memcpy(host.data() + (h * kCtx + prevStart) * kHeadDim, src + (h * chunkPresRows + newRowsAt) * kHeadDim, (size_t) prevLen * kHeadDim * 4);
            }
        }
        f.sess->clearLinks();
        return true;
    }

    int64_t greedyArgMax(const std::vector<float> &row) {
        int64_t best = 0;
        for (int64_t i = 1; i < (int64_t) row.size(); ++i)
        {
            if (row[(size_t) i] > row[(size_t) best])
            {
                best = i;
            }
        }
        return best;
    }

    // One turn: prefill (chunked or whole-window) + kDecodeSteps greedy host-flow decode steps.
    void runTurn(Flow &f, const std::vector<int64_t> &prompt, bool chunked, size_t chunkBucket, int64_t chunkS) {
        if (chunked)
        {
            ASSERT_TRUE(chunkedPrefill(f, chunkBucket, chunkS, prompt));
        } else
        {
            size_t done = 0;
            while (done < prompt.size())
            {
                const int len = (int) std::min<size_t>(prompt.size() - done, (size_t) kLegacyWindow);
                ASSERT_TRUE(hostPass(f, kLegacyWindow, &prompt[done], len, 0)); // the legacy pad id
                done += (size_t) len;
            }
        }
        for (int step = 0; step < kDecodeSteps; ++step)
        {
            const int64_t next = greedyArgMax(f.logitsRow);
            f.tokens.push_back(next);
            ASSERT_TRUE(hostPass(f, 1, &next, 1, 0));
        }
    }

    Flow makeFlow(const std::string &vxmPath) {
        Flow f;
        Config cfg;
        cfg.backend = BackendKind::Cpu;
        f.sess      = Session::createFromVxm(vxmPath, cfg);
        f.pastK.assign((size_t) (kKvHeads * kCtx * kHeadDim), 0.f);
        f.pastV.assign((size_t) (kKvHeads * kCtx * kHeadDim), 0.f);
        return f;
    }

    std::vector<int64_t> promptOf(size_t n, int64_t seed) {
        std::vector<int64_t> p(n);
        for (size_t i = 0; i < n; ++i)
        {
            p[i] = (int64_t) ((seed + 7 * i) % kVocab);
        }
        return p;
    }

    long fileSize(const std::string &path) {
        FILE *file = fopen(path.c_str(), "rb");
        if (!file)
        {
            return -1;
        }
        fseek(file, 0, SEEK_END);
        long n = ftell(file);
        fclose(file);
        return n;
    }

} // namespace

// planChunkPrefillBucket derives the chunk bucket's shape set from the decode bucket: ids and
// positions widen to [1, kChunkPrefillTokens], the mask to [1, C + chunk], the past inputs keep
// their decode shapes.
TEST(ChunkPrefill, PlanDerivesShapesFromDecodeBucket) {
    std::vector<Graph> buckets;
    buckets.push_back(compiled(kLegacyWindow));
    buckets.push_back(compiled(1));
    ChunkPrefillPlan plan;
    ASSERT_TRUE(planChunkPrefillBucket(buckets, &plan));
    EXPECT_EQ(plan.decodeBucket, 1u);
    EXPECT_EQ(plan.shapes.at("input_ids"), (Shape {1, kChunkPrefillTokens}));
    EXPECT_EQ(plan.shapes.at("position_ids"), (Shape {1, kChunkPrefillTokens}));
    EXPECT_EQ(plan.shapes.at("attention_mask"), (Shape {1, kCtx + kChunkPrefillTokens}));
    EXPECT_EQ(plan.shapes.at("past_key_values.0.key"), (Shape {1, kKvHeads, kCtx, kHeadDim}));
    EXPECT_EQ(plan.shapes.at("past_key_values.0.value"), (Shape {1, kKvHeads, kCtx, kHeadDim}));
    EXPECT_EQ(plan.shapes.size(), 5u);
    EXPECT_EQ(plan.label.rfind("chunk-prefill", 0), 0u);
}

// The plan refuses what the chunked driver cannot serve: no position_ids input, a context shorter
// than one chunk, and a bucket set that already carries a chunk-capable prefill shape.
TEST(ChunkPrefill, PlanRefusals) {
    ChunkPrefillPlan plan;
    {
        std::vector<Graph> noPos;
        noPos.push_back(compiled(1, kCtx, /*withPositionIds=*/false));
        EXPECT_FALSE(planChunkPrefillBucket(noPos, &plan));
    }
    {
        std::vector<Graph> shortCtx;
        shortCtx.push_back(compiled(1, kChunkPrefillTokens / 2));
        EXPECT_FALSE(planChunkPrefillBucket(shortCtx, &plan));
    }
    {
        std::vector<Graph> already;
        already.push_back(compiled(1));
        already.push_back(compiled(kChunkPrefillTokens));
        EXPECT_FALSE(planChunkPrefillBucket(already, &plan));
    }
}

// The correctness gate: the chunked-resident prefill produces a byte-identical cache, logits row,
// and greedy token stream to the legacy whole-window host-flow prefill on the CPU backend, for a
// prompt shorter than one chunk, an exact two-chunk multiple, one-over that multiple, and a second
// turn over the populated cache. The multi-bucket .vxm carries decode + whole-window + chunk
// buckets over one deduped weight pool (the chunk bucket's size cost is graph metadata only).
TEST(ChunkPrefill, ChunkedPrefillMatchesLegacyByteExact) {
    const std::string twoPath   = testing::TempDir() + "chunk_prefill_two.vxm";
    const std::string threePath = testing::TempDir() + "chunk_prefill_three.vxm";
    {
        std::vector<Graph> two;
        two.push_back(compiled(1));
        two.push_back(compiled(kLegacyWindow));
        ASSERT_TRUE(saveGraphBinBuckets(two, {"decode", "prefill"}, twoPath));
        // The compile-side plan over these buckets must describe exactly the chunk graph appended
        // below (the emission path builds it from these shapes).
        ChunkPrefillPlan plan;
        ASSERT_TRUE(planChunkPrefillBucket(two, &plan));
        std::vector<Graph> three = std::move(two);
        Graph              chunk = compiled(kChunkPrefillTokens);
        for (TensorId in: chunk.inputs)
        {
            EXPECT_EQ(plan.shapes.at(chunk.desc(in).name), chunk.desc(in).shape) << chunk.desc(in).name;
        }
        three.push_back(std::move(chunk));
        ASSERT_TRUE(saveGraphBinBuckets(three, {"decode", "prefill", "chunk-prefill"}, threePath));
    }
    const long twoBytes = fileSize(twoPath), threeBytes = fileSize(threePath);
    ASSERT_GT(twoBytes, 0);
    ASSERT_GT(threeBytes, 0);
    printf("[chunk-prefill] synthetic .vxm: 2 buckets = %ld bytes, +chunk bucket = %ld bytes (+%ld)\n", twoBytes, threeBytes, threeBytes - twoBytes);

    const size_t chunkTokens = (size_t) kChunkPrefillTokens;
    const std::vector<std::vector<int64_t>> firstPrompts = {
        promptOf(5, 3),                // shorter than one chunk
        promptOf(2 * chunkTokens, 11), // exact chunk multiple
        promptOf(2 * chunkTokens + 1, 19), // one over a multiple
    };
    for (size_t caseIdx = 0; caseIdx < firstPrompts.size(); ++caseIdx)
    {
        Flow chunked = makeFlow(threePath);
        Flow legacy  = makeFlow(threePath);
        ASSERT_TRUE(chunked.sess);
        ASSERT_TRUE(legacy.sess);
        size_t chunkBucket = 0;
        {
            // The chunk bucket is the one whose input_ids second dim equals the chunk size.
            bool found = false;
            for (size_t b = 0; b < chunked.sess->bucketCount() && !found; ++b)
            {
                for (const IOInfo &in: chunked.sess->inputInfo(b))
                {
                    if (in.name == "input_ids" && in.shape == Shape {1, kChunkPrefillTokens})
                    {
                        chunkBucket = b;
                        found       = true;
                    }
                }
            }
            ASSERT_TRUE(found);
        }

        runTurn(chunked, firstPrompts[caseIdx], /*chunked=*/true, chunkBucket, kChunkPrefillTokens);
        runTurn(legacy, firstPrompts[caseIdx], /*chunked=*/false, chunkBucket, kChunkPrefillTokens);
        // Byte-compare the materialized cache and logits row after the first turn's decode steps
        // as well as the token stream (the caches were already compared implicitly through them,
        // but the explicit byte gate pins the resident-equivalent state).
        EXPECT_EQ(chunked.tokens, legacy.tokens) << "case " << caseIdx;
        EXPECT_EQ(chunked.p, legacy.p) << "case " << caseIdx;
        ASSERT_EQ(chunked.pastK.size(), legacy.pastK.size());
        EXPECT_EQ(0, std::memcmp(chunked.pastK.data(), legacy.pastK.data(), chunked.pastK.size() * 4)) << "case " << caseIdx << " key cache";
        EXPECT_EQ(0, std::memcmp(chunked.pastV.data(), legacy.pastV.data(), chunked.pastV.size() * 4)) << "case " << caseIdx << " value cache";
        EXPECT_EQ(chunked.logitsRow, legacy.logitsRow) << "case " << caseIdx;

        // Second turn over the populated cache: the chunked mask must mark the previous turn's
        // rows as valid past.
        const std::vector<int64_t> secondPrompt = promptOf(9, 23);
        runTurn(chunked, secondPrompt, /*chunked=*/true, chunkBucket, kChunkPrefillTokens);
        runTurn(legacy, secondPrompt, /*chunked=*/false, chunkBucket, kChunkPrefillTokens);
        EXPECT_EQ(chunked.tokens, legacy.tokens) << "case " << caseIdx << " turn 2";
        EXPECT_EQ(0, std::memcmp(chunked.pastK.data(), legacy.pastK.data(), chunked.pastK.size() * 4)) << "case " << caseIdx << " turn 2 key cache";
        EXPECT_EQ(0, std::memcmp(chunked.pastV.data(), legacy.pastV.data(), chunked.pastV.size() * 4)) << "case " << caseIdx << " turn 2 value cache";
        EXPECT_EQ(chunked.logitsRow, legacy.logitsRow) << "case " << caseIdx << " turn 2";
    }
}
