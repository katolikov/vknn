// Greedy speculative decoding (spec_decode.h): a small draft decoder proposes kSpecDraftTokens
// tokens, the target verifies them in ONE forward through its [1, kSpecVerifyTokens] bucket, and a
// proposal is committed only when it equals the target's own argmax at that position.
//
// These host tests pin the feature on the CPU byte oracle, with no GPU involved:
//   - planSpecVerifyBucket derives the verification bucket's shape set from a decode bucket, and
//     refuses the models it cannot serve;
//   - the speculative decode loop (engine-resident verify KV + specVerifyFoldRanges folds of the
//     ACCEPTED rows only + one readResident materialization, exactly the vknn_chat flow) emits the
//     token stream the plain token-by-token decode loop emits, and leaves a BYTE-IDENTICAL KV cache,
//     across full acceptance, partial acceptance, total rejection, a round that runs into the
//     compiled context edge, an early end-of-stream, and a second conversation turn.
//
// Acceptance outcomes are forced two ways, because both matter. A SCRIPTED proposal source (the
// reference stream itself, that stream with one token corrupted, and a constant wrong id) pins the
// j == k / 0 < j < k / j == 0 branches deterministically regardless of how the two buckets round.
// A REAL second decoder session — a synth::buildDecoder at a different weight phase, driven through
// its own [1,1] bucket with its own host KV cache — proves the same loop over an actual draft model,
// including a draft whose every proposal is wrong.
//
// The equivalence proved here is the rule's, not the kernels'. Row 0 of a verification forward and
// the decode bucket's single-token forward must agree closely enough to pick the same argmax; on
// this CPU oracle they do, and the device gate re-proves it for the Vulkan M > 1 attention path.
#include "import/passes.h"
#include "synthetic_decoder.h"
#include "vknn/graph.h"
#include "vknn/session.h"
#include "vknn/spec_decode.h"
#include <algorithm>
#include <cstring>
#include <functional>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

using namespace vknn;

namespace {

    constexpr int64_t kVocab       = 48;
    constexpr int64_t kCtx         = 96; // cache slots C; several verification windows wide
    constexpr int     kBudget      = 14; // generated tokens per turn (the --max-tokens analogue)
    constexpr int64_t kNoEos       = -1; // the "never matches" sentinel vknn_chat uses
    constexpr int64_t kBadDraftId  = 7;  // the always-wrong scripted proposal
    constexpr float   kNearDraftPhase = 0.02f; // a lightly perturbed draft: agrees with the target often
    constexpr float   kFarDraftPhase  = 1.7f;  // a deliberately bad draft: a different model entirely

    synth::DecoderSpec decoderSpec(float weightPhase) {
        synth::DecoderSpec s;
        s.vocab       = kVocab;
        s.ctx         = kCtx;
        s.weightPhase = weightPhase;
        return s;
    }

    Graph compiled(const synth::DecoderSpec &spec, int64_t S, bool withPositionIds = true) {
        Graph g = synth::buildDecoder(spec, S, withPositionIds);
        runStandardPasses(g);
        return g;
    }

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
    int64_t argMaxRow(const float *row, int64_t n) {
        int64_t best = 0;
        for (int64_t i = 1; i < n; ++i)
        {
            if (row[i] > row[best])
            {
                best = i;
            }
        }
        return best;
    }

    const char *kPresNames[2] = {"present.0.key", "present.0.value"};
    const char *kPastNames[2] = {"past_key_values.0.key", "past_key_values.0.value"};

    // ---- one decoder on the host KV cache flow -------------------------------------------------
    // Binds the full cache per run and folds the produced row back on the host. The plain reference
    // loop, the prompt feed of every path, and the draft model all run on this — it is the
    // token-by-token decode vknn_chat falls back to, at test scale.
    struct HostDecoder {
        std::unique_ptr<Session> sess;
        synth::DecoderSpec       spec;
        size_t                   decodeBucket = 0;
        std::vector<float>       pastK, pastV; // [kvHeads, C, headDim]
        int                      p = 0;        // absolute position: rows 0..p-1 are live
        std::vector<float>       logitsRow;    // last forward's logits (predicts position p)
    };

    void resetCache(HostDecoder &d) {
        d.pastK.assign((size_t) (d.spec.kvHeads * d.spec.ctx * d.spec.headDim), 0.f);
        d.pastV.assign((size_t) (d.spec.kvHeads * d.spec.ctx * d.spec.headDim), 0.f);
        d.p = 0;
    }

    // Feed one token at position d.p through the [1,1] bucket; fold its present row into slot d.p.
    bool hostStep(HostDecoder &d, int64_t tok) {
        const synth::DecoderSpec &s = d.spec;
        const int64_t             T = s.ctx + 1;
        std::vector<int64_t>      mask((size_t) T, 0);
        for (int64_t j = 0; j < d.p && j < s.ctx; ++j)
        {
            mask[(size_t) j] = 1;
        }
        mask[(size_t) s.ctx] = 1; // the current token, appended at index C
        std::vector<IOTensor> ins {
            i64Tensor("input_ids", {1, 1}, {tok}),
            i64Tensor("position_ids", {1, 1}, {(int64_t) d.p}),
            i64Tensor("attention_mask", {1, T}, mask),
            f32Tensor(kPastNames[0], {1, s.kvHeads, s.ctx, s.headDim}, d.pastK),
            f32Tensor(kPastNames[1], {1, s.kvHeads, s.ctx, s.headDim}, d.pastV),
        };
        std::vector<IOTensor> outs;
        if (d.sess->run(ins, outs) != Status::Ok)
        {
            return false;
        }
        const IOTensor *presK  = outByName(outs, kPresNames[0]);
        const IOTensor *presV  = outByName(outs, kPresNames[1]);
        const IOTensor *logits = outByName(outs, "logits");
        if (!presK || !presV || !logits)
        {
            return false;
        }
        const int64_t presRows = presK->shape[2];
        const int64_t slot     = d.p < s.ctx ? d.p : s.ctx - 1; // the driver's overrun clamp
        for (int part = 0; part < 2; ++part)
        {
            const float *src = (part ? presV : presK)->f32();
            float       *dst = (part ? d.pastV : d.pastK).data();
            for (int64_t h = 0; h < s.kvHeads; ++h)
            {
                std::memcpy(dst + (h * s.ctx + slot) * s.headDim, src + (h * presRows + presRows - 1) * s.headDim, (size_t) s.headDim * 4);
            }
        }
        ++d.p;
        d.logitsRow.assign(logits->f32(), logits->f32() + s.vocab);
        return true;
    }

    // ---- proposal sources ----------------------------------------------------------------------
    // A round's kSpecDraftTokens proposals for the positions after `anchorPos`, extending the token
    // `next` that sits AT `anchorPos`. Every source feeds the same verification machinery — the
    // acceptance rule and the cache fold never learn where a proposal came from.
    using ProposalSource = std::function<void(int64_t next, int anchorPos, int64_t *out, int count)>;

    // A real second decoder: catch its cache up to `anchorPos` (rows past it belong to the previous
    // round's rejected proposals — stale, masked out by the round's own mask, and overwritten here),
    // then run `count` single-token forwards feeding each proposal forward.
    ProposalSource draftSessionSource(HostDecoder &draft) {
        return [&draft](int64_t next, int anchorPos, int64_t *out, int count) {
            draft.p       = anchorPos;
            int64_t token = next;
            for (int i = 0; i < count; ++i)
            {
                ASSERT_TRUE(hostStep(draft, token));
                out[i] = argMaxRow(draft.logitsRow.data(), draft.spec.vocab);
                token  = out[i];
            }
        };
    }

    // A scripted source reading `oracle` (a stream the target is known to produce) at the positions
    // after the anchor, with `wrongAt` proposals of each round corrupted from the front. wrongAt == 0
    // is a perfect draft (full acceptance), wrongAt == count an always-wrong one (total rejection).
    // `oracle` must run kSpecDraftTokens past the turn's token budget: the last round still proposes
    // a full window, and a proposal past the oracle's end would read as a mismatch and understate
    // the acceptance the case is pinning.
    ProposalSource scriptedSource(const std::vector<int64_t> &oracle, int promptLen, int wrongAt) {
        return [&oracle, promptLen, wrongAt](int64_t, int anchorPos, int64_t *out, int count) {
            for (int i = 0; i < count; ++i)
            {
                // The token at absolute position anchorPos + 1 + i, when the oracle reaches it.
                const int idx  = anchorPos + 1 + i - promptLen;
                const int64_t o = idx >= 0 && idx < (int) oracle.size() ? oracle[(size_t) idx] : kBadDraftId;
                out[i] = i < wrongAt ? (o + 1) % kVocab : o;
            }
        };
    }

    // ---- the two decode loops ------------------------------------------------------------------

    struct TurnResult {
        std::vector<int64_t> tokens;         ///< Emitted stream for this turn.
        std::vector<int>     acceptedRounds; ///< Accepted-draft count per speculative round.
        std::vector<int>     emittedRounds;  ///< Tokens each round actually emitted (1 + accepted, truncated).
    };

    // The plain reference: one forward per token through the [1,1] decode bucket.
    TurnResult plainDecode(HostDecoder &target, int64_t eos, int budget) {
        TurnResult result;
        for (int n = 0; n < budget; ++n)
        {
            const int64_t next = argMaxRow(target.logitsRow.data(), target.spec.vocab);
            if (next == eos)
            {
                break;
            }
            result.tokens.push_back(next);
            EXPECT_TRUE(hostStep(target, next));
        }
        return result;
    }

    // The speculative loop, mirroring the vknn_chat driver: link the verification bucket's present
    // outputs to its past inputs, run one batched forward per round with the ACCEPTED rows of the
    // previous round armed as the fold ranges, and materialize the resident cache back to the host
    // buffers at the end of the turn. A round that would cross the compiled context edge hands the
    // rest of the turn to the plain loop.
    TurnResult specDecode(HostDecoder &target, size_t verifyBucket, const ProposalSource &propose, int64_t eos, int budget) {
        const synth::DecoderSpec &s        = target.spec;
        const int                 drafts   = (int) kSpecDraftTokens;
        const int64_t             window   = kSpecVerifyTokens;
        const int64_t             maskLen  = s.ctx + window;
        TurnResult                result;
        int64_t                   presRows = 0;
        for (const IOInfo &o: target.sess->outputInfo(verifyBucket))
        {
            if (o.name == kPresNames[0])
            {
                presRows = o.shape[2];
            }
        }
        EXPECT_GE(presRows, window);
        for (int part = 0; part < 2; ++part)
        {
            EXPECT_EQ(Status::Ok, target.sess->linkOutputToInput(verifyBucket, kPresNames[part], kPastNames[part], {}));
        }

        int64_t next      = argMaxRow(target.logitsRow.data(), s.vocab);
        int     pendSlot  = -1; // slot the last round's accepted rows fold into
        int     pendRows  = 0;
        bool    firstPass = true;
        bool    ranAny    = false;
        bool    turnDone  = false; // a round hit end-of-stream or the token budget: no tail follows
        while ((int) result.tokens.size() < budget && next != eos && target.p + window <= s.ctx)
        {
            std::vector<int64_t> proposals((size_t) drafts, 0);
            propose(next, target.p, proposals.data(), drafts);

            const std::vector<LinkRange> ranges = specVerifyFoldRanges(s.kvHeads, presRows, s.ctx, s.headDim, window, pendSlot, pendRows);
            for (int part = 0; part < 2; ++part)
            {
                EXPECT_EQ(Status::Ok, target.sess->linkOutputToInput(verifyBucket, kPresNames[part], kPastNames[part], ranges));
            }
            std::vector<int64_t> ids((size_t) window), pos((size_t) window), mask((size_t) maskLen, 0);
            ids[0] = next;
            for (int i = 0; i < drafts; ++i)
            {
                ids[(size_t) i + 1] = proposals[(size_t) i];
            }
            for (int64_t t = 0; t < window; ++t)
            {
                pos[(size_t) t]             = target.p + t;
                mask[(size_t) (s.ctx + t)] = 1; // every window column is a real token
            }
            for (int64_t j = 0; j < target.p && j < s.ctx; ++j)
            {
                mask[(size_t) j] = 1;
            }
            std::vector<IOTensor> ins {
                i64Tensor("input_ids", {1, window}, ids),
                i64Tensor("position_ids", {1, window}, pos),
                i64Tensor("attention_mask", {1, maskLen}, mask),
            };
            if (firstPass)
            {
                // Binding a linked input reinitializes its resident state: the turn's first pass
                // seeds the verification bucket's cache from the host buffers.
                ins.push_back(f32Tensor(kPastNames[0], {1, s.kvHeads, s.ctx, s.headDim}, target.pastK));
                ins.push_back(f32Tensor(kPastNames[1], {1, s.kvHeads, s.ctx, s.headDim}, target.pastV));
            }
            std::vector<IOTensor> outs;
            EXPECT_EQ(Status::Ok, target.sess->run(ins, outs));
            const IOTensor *presK  = outByName(outs, kPresNames[0]);
            const IOTensor *logits = outByName(outs, "logits");
            EXPECT_TRUE(presK && presK->data.empty()); // linked outputs return no host data
            if (!logits)
            {
                ADD_FAILURE() << "verification pass produced no logits";
                break;
            }
            firstPass = false;
            ranAny    = true;

            std::vector<int64_t> targetArgMax((size_t) window);
            for (int64_t r = 0; r < window; ++r)
            {
                targetArgMax[(size_t) r] = argMaxRow(logits->f32() + r * s.vocab, s.vocab);
            }
            const int accepted = specAcceptedDrafts(proposals.data(), targetArgMax.data(), drafts);
            result.acceptedRounds.push_back(accepted);

            std::vector<int64_t> committed;
            committed.push_back(next);
            for (int i = 0; i < accepted; ++i)
            {
                committed.push_back(proposals[(size_t) i]);
            }
            const int emitted = specEmittedCount(committed.data(), (int) committed.size(), eos, budget - (int) result.tokens.size());
            for (int i = 0; i < emitted; ++i)
            {
                result.tokens.push_back(committed[(size_t) i]);
            }
            result.emittedRounds.push_back(emitted);
            pendSlot = target.p;
            pendRows = emitted;
            target.p += emitted;
            if (emitted < (int) committed.size())
            {
                turnDone = true; // end-of-stream or the token budget cut the round short
                break;
            }
            next = targetArgMax[(size_t) accepted]; // the correction, or the bonus token at full acceptance
        }
        // Materialize: the resident past (every armed fold applied) plus the last round's accepted
        // rows, which are still pending in the present outputs.
        if (ranAny)
        {
            for (int part = 0; part < 2; ++part)
            {
                IOTensor resident;
                EXPECT_EQ(Status::Ok, target.sess->readResident(kPastNames[part], resident));
                std::vector<float> &host = part ? target.pastV : target.pastK;
                EXPECT_EQ(resident.data.size(), host.size() * 4);
                std::memcpy(host.data(), resident.data.data(), resident.data.size());
                if (pendRows > 0)
                {
                    IOTensor present;
                    EXPECT_EQ(Status::Ok, target.sess->readResident(kPresNames[part], present));
                    const float *src = reinterpret_cast<const float *>(present.data.data());
                    for (int64_t h = 0; h < s.kvHeads; ++h)
                    {
                        std::memcpy(host.data() + (h * s.ctx + pendSlot) * s.headDim, src + (h * presRows + presRows - window) * s.headDim, (size_t) pendRows * s.headDim * 4);
                    }
                }
            }
        }
        target.sess->clearLinks();
        // Only the CONTEXT EDGE hands the rest of the turn to the plain loop; a round cut short by
        // end-of-stream or the token budget ends the turn, exactly as the driver does. `next` is
        // already in hand, so the tail feeds it and continues token by token from there.
        while (!turnDone && (int) result.tokens.size() < budget && next != eos)
        {
            result.tokens.push_back(next);
            EXPECT_TRUE(hostStep(target, next));
            next = argMaxRow(target.logitsRow.data(), s.vocab);
        }
        return result;
    }

    // ---- fixtures -------------------------------------------------------------------------------

    std::vector<int64_t> promptOf(size_t n, int64_t seed) {
        std::vector<int64_t> p(n);
        for (size_t i = 0; i < n; ++i)
        {
            p[(size_t) i] = (int64_t) ((seed + 7 * i) % kVocab);
        }
        return p;
    }

    // A two-bucket target .vxm (decode [1,1] + verification [1, kSpecVerifyTokens]) over one deduped
    // weight pool, and the path it was written to.
    std::string writeTarget(const synth::DecoderSpec &spec, const std::string &path) {
        std::vector<Graph> buckets;
        buckets.push_back(compiled(spec, 1));
        SpecVerifyPlan plan;
        EXPECT_TRUE(planSpecVerifyBucket(buckets, &plan));
        Graph verify = compiled(spec, kSpecVerifyTokens);
        for (TensorId in: verify.inputs)
        {
            EXPECT_EQ(plan.shapes.at(verify.desc(in).name), verify.desc(in).shape) << verify.desc(in).name;
        }
        buckets.push_back(std::move(verify));
        EXPECT_TRUE(saveGraphBinBuckets(buckets, {"decode", plan.label}, path));
        return path;
    }

    std::unique_ptr<HostDecoder> openDecoder(const std::string &path, const synth::DecoderSpec &spec) {
        auto   decoder = std::make_unique<HostDecoder>();
        Config cfg;
        cfg.backend    = BackendKind::Cpu;
        decoder->sess  = Session::createFromVxm(path, cfg);
        decoder->spec  = spec;
        if (decoder->sess)
        {
            for (size_t b = 0; b < decoder->sess->bucketCount(); ++b)
            {
                for (const IOInfo &in: decoder->sess->inputInfo(b))
                {
                    if (in.name == "input_ids" && in.shape == Shape {1, 1})
                    {
                        decoder->decodeBucket = b;
                    }
                }
            }
        }
        resetCache(*decoder);
        return decoder;
    }

    size_t verifyBucketOf(const Session &sess) {
        for (size_t b = 0; b < sess.bucketCount(); ++b)
        {
            for (const IOInfo &in: sess.inputInfo(b))
            {
                if (in.name == "input_ids" && in.shape == Shape {1, kSpecVerifyTokens})
                {
                    return b;
                }
            }
        }
        ADD_FAILURE() << "no [1, kSpecVerifyTokens] verification bucket";
        return 0;
    }

    // The target's own greedy stream over `prompt`, long enough for a scripted source to fill every
    // round's whole window: kSpecDraftTokens tokens past the turn budget the cases compare against.
    std::vector<int64_t> oracleStream(const std::string &path, const synth::DecoderSpec &spec, const std::vector<int64_t> &prompt, int budget);

    void feedPrompt(HostDecoder &decoder, const std::vector<int64_t> &prompt) {
        for (int64_t tok: prompt)
        {
            ASSERT_TRUE(hostStep(decoder, tok));
        }
    }

    std::vector<int64_t> oracleStream(const std::string &path, const synth::DecoderSpec &spec, const std::vector<int64_t> &prompt, int budget) {
        auto probe = openDecoder(path, spec);
        EXPECT_TRUE(probe->sess);
        for (int64_t tok: prompt)
        {
            EXPECT_TRUE(hostStep(*probe, tok));
        }
        return plainDecode(*probe, kNoEos, budget + (int) kSpecDraftTokens).tokens;
    }

} // namespace

// planSpecVerifyBucket derives the verification bucket's shape set from the decode bucket: ids and
// positions widen to [1, kSpecVerifyTokens], the mask to [1, C + kSpecVerifyTokens], the past inputs
// keep their decode shapes.
TEST(SpecDecode, PlanDerivesShapesFromDecodeBucket) {
    std::vector<Graph> buckets;
    buckets.push_back(compiled(decoderSpec(0.0f), 1));
    SpecVerifyPlan plan;
    ASSERT_TRUE(planSpecVerifyBucket(buckets, &plan));
    EXPECT_EQ(plan.decodeBucket, 0u);
    EXPECT_EQ(plan.cacheSlots, kCtx);
    EXPECT_EQ(plan.shapes.at("input_ids"), (Shape {1, kSpecVerifyTokens}));
    EXPECT_EQ(plan.shapes.at("position_ids"), (Shape {1, kSpecVerifyTokens}));
    EXPECT_EQ(plan.shapes.at("attention_mask"), (Shape {1, kCtx + kSpecVerifyTokens}));
    EXPECT_EQ(plan.shapes.at("past_key_values.0.key"), (Shape {1, 2, kCtx, 4}));
    EXPECT_EQ(plan.shapes.size(), 5u);
    EXPECT_EQ(plan.label.rfind("spec-verify", 0), 0u);
}

// The plan refuses what the verification pass cannot be driven over: no position_ids input (the pass
// feeds its window's absolute positions), a context shorter than one verification window, and a
// bucket set that already carries one at exactly this width.
TEST(SpecDecode, PlanRefusals) {
    SpecVerifyPlan plan;
    {
        std::vector<Graph> noPos;
        noPos.push_back(compiled(decoderSpec(0.0f), 1, /*withPositionIds=*/false));
        EXPECT_FALSE(planSpecVerifyBucket(noPos, &plan));
    }
    {
        synth::DecoderSpec tiny = decoderSpec(0.0f);
        tiny.ctx                = kSpecVerifyTokens - 1;
        std::vector<Graph> shortCtx;
        shortCtx.push_back(compiled(tiny, 1));
        EXPECT_FALSE(planSpecVerifyBucket(shortCtx, &plan));
    }
    {
        std::vector<Graph> already;
        already.push_back(compiled(decoderSpec(0.0f), 1));
        already.push_back(compiled(decoderSpec(0.0f), kSpecVerifyTokens));
        EXPECT_FALSE(planSpecVerifyBucket(already, &plan));
    }
}

// The correctness gate under SCRIPTED proposals, which pin each acceptance branch deterministically:
// a perfect draft (every round fully accepted, kSpecDraftTokens + 1 tokens per target forward), a
// draft whose leading proposals are wrong (partial acceptance, the target's own argmax replacing the
// first mismatch), and an always-wrong draft (total rejection, one token per round — the plain
// stream at extra cost). Every variant must emit the plain loop's stream and leave a byte-identical
// KV cache; the rejected rows must appear in neither.
TEST(SpecDecode, ScriptedAcceptancePatternsMatchPlainStream) {
    const std::string  path   = testing::TempDir() + "spec_decode_target.vxm";
    synth::DecoderSpec target = decoderSpec(0.0f);
    writeTarget(target, path);

    const std::vector<int64_t> prompt = promptOf(6, 3);

    // The reference: plain token-by-token decode over the [1,1] bucket. The scripted sources read a
    // longer run of the same stream, so a round near the budget still proposes a full window.
    auto reference = openDecoder(path, target);
    ASSERT_TRUE(reference->sess);
    feedPrompt(*reference, prompt);
    const TurnResult           plain  = plainDecode(*reference, kNoEos, kBudget);
    const std::vector<int64_t> oracle = oracleStream(path, target, prompt, kBudget);
    ASSERT_EQ((int) plain.tokens.size(), kBudget);
    ASSERT_TRUE(std::equal(plain.tokens.begin(), plain.tokens.end(), oracle.begin()));

    struct Variant {
        const char *name;
        int         wrongAt;      ///< Leading proposals corrupted per round.
        int         expectAccept; ///< Accepted drafts every round must report.
    };
    const Variant variants[] = {
        {"perfect draft", 0, (int) kSpecDraftTokens},
        {"first proposal wrong", 1, 0},
        {"first two proposals wrong", 2, 0},
        {"always wrong", (int) kSpecDraftTokens, 0},
    };
    for (const Variant &v: variants)
    {
        auto speculative = openDecoder(path, target);
        ASSERT_TRUE(speculative->sess);
        const size_t verifyBucket = verifyBucketOf(*speculative->sess);
        feedPrompt(*speculative, prompt);
        const TurnResult spec = specDecode(*speculative, verifyBucket, scriptedSource(oracle, (int) prompt.size(), v.wrongAt), kNoEos, kBudget);

        EXPECT_EQ(spec.tokens, plain.tokens) << v.name;
        EXPECT_EQ(speculative->p, reference->p) << v.name;
        ASSERT_EQ(speculative->pastK.size(), reference->pastK.size());
        EXPECT_EQ(0, std::memcmp(speculative->pastK.data(), reference->pastK.data(), speculative->pastK.size() * 4)) << v.name << " key cache";
        EXPECT_EQ(0, std::memcmp(speculative->pastV.data(), reference->pastV.data(), speculative->pastV.size() * 4)) << v.name << " value cache";
        ASSERT_FALSE(spec.acceptedRounds.empty()) << v.name;
        for (size_t r = 0; r < spec.acceptedRounds.size(); ++r)
        {
            EXPECT_EQ(spec.acceptedRounds[r], v.expectAccept) << v.name << " round " << r;
        }
        // A perfect draft commits kSpecDraftTokens + 1 tokens per target forward; total rejection
        // commits one. The round count is the whole point of the feature, so pin it.
        const size_t expectRounds = v.wrongAt == 0 ? (size_t) ((kBudget + kSpecVerifyTokens - 1) / kSpecVerifyTokens) : (size_t) kBudget;
        EXPECT_EQ(spec.acceptedRounds.size(), expectRounds) << v.name;
    }
}

// A partially accepted round: proposals that follow the target for a while and then diverge. The
// accepted prefix must be committed, the first mismatch replaced by the target's own argmax, and the
// rows after it left out of the cache.
TEST(SpecDecode, PartialAcceptanceMatchesPlainStream) {
    const std::string  path   = testing::TempDir() + "spec_decode_partial.vxm";
    synth::DecoderSpec target = decoderSpec(0.0f);
    writeTarget(target, path);
    const std::vector<int64_t> prompt = promptOf(5, 11);

    auto reference = openDecoder(path, target);
    ASSERT_TRUE(reference->sess);
    feedPrompt(*reference, prompt);
    const TurnResult           plain  = plainDecode(*reference, kNoEos, kBudget);
    const std::vector<int64_t> oracle = oracleStream(path, target, prompt, kBudget);

    // Correct up to `good` proposals per round, then wrong: acceptance lands strictly inside
    // (0, kSpecDraftTokens) so both the accepted-prefix fold and the correction run every round.
    for (int good = 1; good < (int) kSpecDraftTokens; ++good)
    {
        auto speculative = openDecoder(path, target);
        ASSERT_TRUE(speculative->sess);
        const size_t verifyBucket = verifyBucketOf(*speculative->sess);
        feedPrompt(*speculative, prompt);
        const int      promptLen = (int) prompt.size();
        ProposalSource partial   = [&oracle, promptLen, good](int64_t, int anchorPos, int64_t *out, int count) {
            for (int i = 0; i < count; ++i)
            {
                const int     idx = anchorPos + 1 + i - promptLen;
                const int64_t o   = idx >= 0 && idx < (int) oracle.size() ? oracle[(size_t) idx] : kBadDraftId;
                out[i]            = i < good ? o : (o + 1) % kVocab;
            }
        };
        const TurnResult spec = specDecode(*speculative, verifyBucket, partial, kNoEos, kBudget);
        EXPECT_EQ(spec.tokens, plain.tokens) << "good=" << good;
        EXPECT_EQ(0, std::memcmp(speculative->pastK.data(), reference->pastK.data(), speculative->pastK.size() * 4)) << "good=" << good << " key cache";
        EXPECT_EQ(0, std::memcmp(speculative->pastV.data(), reference->pastV.data(), speculative->pastV.size() * 4)) << "good=" << good << " value cache";
        for (size_t r = 0; r < spec.acceptedRounds.size(); ++r)
        {
            EXPECT_EQ(spec.acceptedRounds[r], good) << "good=" << good << " round " << r;
        }
    }
}

// REAL draft models: second synthetic decoders with their own sessions and their own host KV
// caches, driven through the same loop. Their proposals are their own, their cache rollback is the
// mask-driven one (rows past the committed position are stale and overwritten by the next round),
// and the emitted stream must be the target's plain stream for every one of them — a draft model can
// only change how many forwards the turn costs.
//
// Three drafts span the range that matters: the target's own weights (a perfect predictor, so the
// acceptance the loop can reach is bounded only by whether the [1,1] and [1, kSpecVerifyTokens]
// buckets pick the same argmax), a lightly perturbed model (a partial predictor), and a model at a
// far weight phase, which is DELIBERATELY bad and rejects on essentially every round.
TEST(SpecDecode, RealDraftModelsMatchPlainStream) {
    const std::string  targetPath = testing::TempDir() + "spec_decode_real_target.vxm";
    synth::DecoderSpec targetSpec = decoderSpec(0.0f);
    writeTarget(targetSpec, targetPath);
    const std::vector<int64_t> prompt = promptOf(7, 5);

    auto reference = openDecoder(targetPath, targetSpec);
    ASSERT_TRUE(reference->sess);
    feedPrompt(*reference, prompt);
    const TurnResult plain = plainDecode(*reference, kNoEos, kBudget);

    struct DraftCase {
        const char *name;
        float       phase;
    };
    const DraftCase drafts[] = {
        {"clone of the target", 0.0f},
        {"lightly perturbed", kNearDraftPhase},
        {"deliberately bad", kFarDraftPhase},
    };
    int bestAccepted = 0;
    for (const DraftCase &d: drafts)
    {
        const std::string  draftPath = testing::TempDir() + "spec_decode_real_draft_" + d.name[0] + std::to_string((int) (d.phase * 100)) + ".vxm";
        synth::DecoderSpec draftSpec = decoderSpec(d.phase);
        {
            std::vector<Graph> draftBuckets;
            draftBuckets.push_back(compiled(draftSpec, 1));
            ASSERT_TRUE(saveGraphBinBuckets(draftBuckets, {"decode"}, draftPath));
        }
        auto speculative = openDecoder(targetPath, targetSpec);
        auto draft       = openDecoder(draftPath, draftSpec);
        ASSERT_TRUE(speculative->sess);
        ASSERT_TRUE(draft->sess);
        const size_t verifyBucket = verifyBucketOf(*speculative->sess);
        feedPrompt(*speculative, prompt);
        feedPrompt(*draft, prompt); // the draft consumes the same prompt to keep its cache aligned

        const TurnResult spec = specDecode(*speculative, verifyBucket, draftSessionSource(*draft), kNoEos, kBudget);
        EXPECT_EQ(spec.tokens, plain.tokens) << d.name;
        EXPECT_EQ(speculative->p, reference->p) << d.name;
        EXPECT_EQ(0, std::memcmp(speculative->pastK.data(), reference->pastK.data(), speculative->pastK.size() * 4)) << d.name;
        EXPECT_EQ(0, std::memcmp(speculative->pastV.data(), reference->pastV.data(), speculative->pastV.size() * 4)) << d.name;
        int totalAccepted = 0;
        for (int a: spec.acceptedRounds)
        {
            totalAccepted += a;
            EXPECT_GE(a, 0) << d.name;
            EXPECT_LE(a, (int) kSpecDraftTokens) << d.name;
        }
        bestAccepted = totalAccepted > bestAccepted ? totalAccepted : bestAccepted;
        printf("[spec-decode] draft '%s': %zu rounds, %d accepted drafts, %zu tokens (%.2f tokens per target forward)\n",
               d.name, spec.acceptedRounds.size(), totalAccepted, spec.tokens.size(),
               spec.acceptedRounds.empty() ? 0.0 : (double) spec.tokens.size() / (double) spec.acceptedRounds.size());
    }
    // At least one real draft must actually get proposals through, or the whole suite would be
    // proving the rejection path only and the acceptance path would be untested against a live
    // second session.
    EXPECT_GT(bestAccepted, 0) << "no real draft was ever accepted; the accept path is untested here";
}

// A draft model whose every proposal is wrong exercises rejection on EVERY round: each round commits
// exactly the anchor token, folds exactly one cache row, and discards kSpecDraftTokens rows. The
// stream is still the plain stream — this is the worst case for throughput and the strongest case
// for the rollback being correct.
TEST(SpecDecode, AlwaysRejectedDraftMatchesPlainStream) {
    const std::string  path   = testing::TempDir() + "spec_decode_reject.vxm";
    synth::DecoderSpec target = decoderSpec(0.0f);
    writeTarget(target, path);
    const std::vector<int64_t> prompt = promptOf(4, 17);

    auto reference = openDecoder(path, target);
    ASSERT_TRUE(reference->sess);
    feedPrompt(*reference, prompt);
    const TurnResult plain = plainDecode(*reference, kNoEos, kBudget);

    auto speculative = openDecoder(path, target);
    ASSERT_TRUE(speculative->sess);
    const size_t verifyBucket = verifyBucketOf(*speculative->sess);
    feedPrompt(*speculative, prompt);
    // Every proposal is an id the target's argmax cannot equal at that position: take the plain
    // stream's own token and shift it.
    const int                  promptLen = (int) prompt.size();
    const std::vector<int64_t> oracle    = oracleStream(path, target, prompt, kBudget);
    ProposalSource             allWrong  = [&oracle, promptLen](int64_t, int anchorPos, int64_t *out, int count) {
        for (int i = 0; i < count; ++i)
        {
            const int     idx = anchorPos + 1 + i - promptLen;
            const int64_t o   = idx >= 0 && idx < (int) oracle.size() ? oracle[(size_t) idx] : kBadDraftId;
            out[i]            = (o + 1 + i) % kVocab;
        }
    };
    const TurnResult spec = specDecode(*speculative, verifyBucket, allWrong, kNoEos, kBudget);
    EXPECT_EQ(spec.tokens, plain.tokens);
    EXPECT_EQ((int) spec.acceptedRounds.size(), kBudget); // one committed token per round
    for (int a: spec.acceptedRounds)
    {
        EXPECT_EQ(a, 0);
    }
    EXPECT_EQ(0, std::memcmp(speculative->pastK.data(), reference->pastK.data(), speculative->pastK.size() * 4));
    EXPECT_EQ(0, std::memcmp(speculative->pastV.data(), reference->pastV.data(), speculative->pastV.size() * 4));
}

// The context edge and a second turn. A prompt that leaves fewer than a whole verification window
// before the compiled context edge forces the loop to hand the rest of the turn to the plain
// single-step path mid-turn; the second turn then starts already inside that margin, so it runs
// entirely on the plain path. Both must emit the plain loop's stream and leave its cache.
TEST(SpecDecode, ContextEdgeAndSecondTurnMatchPlainStream) {
    // A wider context than the other cases, so the turn reaches the edge with tokens still to
    // generate WITHOUT overrunning the cache (the overrun clamp is the single-step loop's own rule
    // and not what this case is pinning).
    synth::DecoderSpec target = decoderSpec(0.0f);
    target.ctx                = 128;
    const std::string path    = testing::TempDir() + "spec_decode_edge.vxm";
    writeTarget(target, path);
    constexpr int kEdgeBudget   = 6;
    constexpr int kSecondBudget = 1;
    // Room for one full verification window and then a margin narrower than the next one.
    const std::vector<int64_t> prompt  = promptOf((size_t) (target.ctx - 8), 29);
    const std::vector<int64_t> prompt2 = promptOf(1, 31);

    auto reference = openDecoder(path, target);
    ASSERT_TRUE(reference->sess);
    feedPrompt(*reference, prompt);
    const TurnResult plain1 = plainDecode(*reference, kNoEos, kEdgeBudget);
    const std::vector<float> refKeyAfterTurn1 = reference->pastK;
    const std::vector<float> refValAfterTurn1 = reference->pastV;
    const int                refPAfterTurn1   = reference->p;
    feedPrompt(*reference, prompt2);
    const TurnResult plain2 = plainDecode(*reference, kNoEos, kSecondBudget);

    auto speculative = openDecoder(path, target);
    ASSERT_TRUE(speculative->sess);
    const size_t verifyBucket = verifyBucketOf(*speculative->sess);
    feedPrompt(*speculative, prompt);
    const std::vector<int64_t> oracle1 = oracleStream(path, target, prompt, kEdgeBudget);
    const TurnResult           spec1   = specDecode(*speculative, verifyBucket, scriptedSource(oracle1, (int) prompt.size(), 0), kNoEos, kEdgeBudget);
    EXPECT_EQ(spec1.tokens, plain1.tokens);
    EXPECT_EQ(speculative->p, refPAfterTurn1);
    // The edge cut the turn short: the rounds emitted fewer tokens than the turn did, so the rest
    // came from the plain tail.
    int emittedByRounds = 0;
    for (int e: spec1.emittedRounds)
    {
        emittedByRounds += e;
    }
    EXPECT_GT((int) spec1.emittedRounds.size(), 0) << "no round ran before the edge";
    EXPECT_LT(emittedByRounds, (int) spec1.tokens.size()) << "the context edge never forced the plain tail";
    EXPECT_EQ(0, std::memcmp(speculative->pastK.data(), refKeyAfterTurn1.data(), speculative->pastK.size() * 4));
    EXPECT_EQ(0, std::memcmp(speculative->pastV.data(), refValAfterTurn1.data(), speculative->pastV.size() * 4));

    // Second turn: the cache is now inside the margin, so no verification window fits and the whole
    // turn takes the plain path — the degradation case, which must still emit the same stream.
    feedPrompt(*speculative, prompt2);
    const TurnResult spec2 = specDecode(*speculative, verifyBucket, scriptedSource(plain2.tokens, speculative->p, 0), kNoEos, kSecondBudget);
    EXPECT_TRUE(spec2.emittedRounds.empty()) << "a verification window fitted after all";
    EXPECT_EQ(spec2.tokens, plain2.tokens);
    EXPECT_EQ(speculative->p, reference->p);
    EXPECT_EQ(0, std::memcmp(speculative->pastK.data(), reference->pastK.data(), speculative->pastK.size() * 4));
}

// End-of-stream inside an accepted run. The plain loop stops at the first eos WITHOUT emitting or
// feeding it; a speculative round that commits past that point would emit extra tokens and fold
// cache rows the plain turn never wrote. specEmittedCount is the single rule both loops apply.
TEST(SpecDecode, EndOfStreamInsideAnAcceptedRun) {
    const std::string  path   = testing::TempDir() + "spec_decode_eos.vxm";
    synth::DecoderSpec target = decoderSpec(0.0f);
    writeTarget(target, path);
    const std::vector<int64_t> prompt = promptOf(6, 13);

    auto probe = openDecoder(path, target);
    ASSERT_TRUE(probe->sess);
    feedPrompt(*probe, prompt);
    const TurnResult unbounded = plainDecode(*probe, kNoEos, kBudget);
    ASSERT_GE(unbounded.tokens.size(), 4u);

    // Stop on a token that lands in the middle of a fully accepted round.
    for (int stopAt = 1; stopAt <= 3; ++stopAt)
    {
        const int64_t eos = unbounded.tokens[(size_t) stopAt];
        auto reference = openDecoder(path, target);
        ASSERT_TRUE(reference->sess);
        feedPrompt(*reference, prompt);
        const TurnResult plain = plainDecode(*reference, eos, kBudget);
        EXPECT_EQ((int) plain.tokens.size(), stopAt) << "stopAt=" << stopAt;

        auto speculative = openDecoder(path, target);
        ASSERT_TRUE(speculative->sess);
        const size_t verifyBucket = verifyBucketOf(*speculative->sess);
        feedPrompt(*speculative, prompt);
        const TurnResult spec = specDecode(*speculative, verifyBucket, scriptedSource(unbounded.tokens, (int) prompt.size(), 0), eos, kBudget);
        EXPECT_EQ(spec.tokens, plain.tokens) << "stopAt=" << stopAt;
        EXPECT_EQ(speculative->p, reference->p) << "stopAt=" << stopAt;
        EXPECT_EQ(0, std::memcmp(speculative->pastK.data(), reference->pastK.data(), speculative->pastK.size() * 4)) << "stopAt=" << stopAt;
        EXPECT_EQ(0, std::memcmp(speculative->pastV.data(), reference->pastV.data(), speculative->pastV.size() * 4)) << "stopAt=" << stopAt;
    }
}

// The acceptance and emission rules on their own, independent of any model: the exact-equality
// prefix test, the correction/bonus index, and the two truncations the plain loop applies.
TEST(SpecDecode, AcceptanceAndEmissionRules) {
    const int64_t drafts[4]   = {10, 11, 12, 13};
    const int64_t allMatch[5] = {10, 11, 12, 13, 14};
    EXPECT_EQ(specAcceptedDrafts(drafts, allMatch, 4), 4);
    const int64_t firstWrong[5] = {99, 11, 12, 13, 14};
    EXPECT_EQ(specAcceptedDrafts(drafts, firstWrong, 4), 0);
    const int64_t thirdWrong[5] = {10, 11, 99, 13, 14};
    EXPECT_EQ(specAcceptedDrafts(drafts, thirdWrong, 4), 2);
    EXPECT_EQ(specAcceptedDrafts(drafts, allMatch, 0), 0);

    const int64_t committed[3] = {5, 6, 7};
    EXPECT_EQ(specEmittedCount(committed, 3, kNoEos, 10), 3); // nothing truncates
    EXPECT_EQ(specEmittedCount(committed, 3, 6, 10), 1);      // eos at index 1
    EXPECT_EQ(specEmittedCount(committed, 3, 5, 10), 0);      // eos is the anchor itself
    EXPECT_EQ(specEmittedCount(committed, 3, kNoEos, 2), 2);  // the token budget
    EXPECT_EQ(specEmittedCount(committed, 3, 7, 2), 2);       // budget cuts before the eos
    EXPECT_EQ(specEmittedCount(committed, 3, kNoEos, 0), 0);

    // The fold ranges name only the accepted rows: 2 accepted of a 5-column window over a
    // cache-concat present carries 2 ranges per head starting at the first produced row.
    const std::vector<LinkRange> ranges = specVerifyFoldRanges(2, 100 + kSpecVerifyTokens, 100, 4, kSpecVerifyTokens, 7, 3);
    ASSERT_EQ(ranges.size(), 2u);
    EXPECT_EQ(ranges[0].sourceElem, 100 * 4);            // head 0, first produced row
    EXPECT_EQ(ranges[0].destElem, 7 * 4);                // slot 7
    EXPECT_EQ(ranges[0].count, 3 * 4);                   // three rows, no more
    EXPECT_TRUE(specVerifyFoldRanges(2, 105, 100, 4, kSpecVerifyTokens, 7, 0).empty());
    EXPECT_TRUE(specVerifyFoldRanges(2, 105, 100, 4, kSpecVerifyTokens, -1, 3).empty());
}
