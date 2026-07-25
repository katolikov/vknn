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
// The synthetic model is synth::buildDecoder (tests/synthetic_decoder.h) built at each bucket's
// sequence length and saved as one multi-bucket .vxm, so the chunked flow is compared against the
// whole-window flow on the graph form the device runs.
#include "import/passes.h"
#include "synthetic_decoder.h"
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
    constexpr int64_t kCtx          = 160; // cache slots C; >= kChunkPrefillTokens so the plan emits
    constexpr int64_t kLegacyWindow = 96;  // whole-window prefill bucket (not a chunk multiple)
    constexpr int64_t kEosId        = 2;   // in-vocab pad id for the chunked path
    constexpr int     kDecodeSteps  = 8;

    // The one decoder these tests compile at every bucket width.
    synth::DecoderSpec spec() {
        synth::DecoderSpec s;
        s.vocab   = kVocab;
        s.qHeads  = kQHeads;
        s.kvHeads = kKvHeads;
        s.headDim = kHeadDim;
        s.ctx     = kCtx;
        return s;
    }

    Graph compiled(int64_t S, int64_t C = kCtx, bool withPositionIds = true) {
        synth::DecoderSpec sp = spec();
        sp.ctx                = C;
        Graph g               = synth::buildDecoder(sp, S, withPositionIds);
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
