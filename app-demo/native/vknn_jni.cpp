// JNI bridge for the on-device Qwen chat demo: wraps a VKNN Session as an autoregressive decoder,
// lifting the token loop from examples/llm/chat.cpp into a handle the Kotlin app drives one step at
// a time.
//
// The model is a with-past Qwen2 decoder compiled at a fixed context length C. A single-bucket .vxm
// serves prefill token by token and every decode step through one plan; a multi-bucket .vxm adds a
// prefill bucket (input_ids [1,S], S>1) that feeds a whole prompt window in ONE forward — prefill()
// dispatches to it when present. The past key/value buffers ARE the KV cache: fp32 host boundary
// tensors retained across steps; each step appends the new token's key/value into cache slot p.
// Every tensor-compute op runs on the Vulkan backend; only argmax/sampling is here — and greedy
// decode registers the decode bucket's logits for the ENGINE-side argmax (Session::setOutputArgMax),
// reading back 8 bytes per token instead of the vocab row.
#include "raster_core.h"
#include "vknn/runtime.h"
#include "vknn/session.h"
#include <algorithm>
#include <android/log.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <jni.h>
#include <random>
#include <string>
#include <vector>

using namespace vknn;

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "vknnchat", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "vknnchat", __VA_ARGS__)

namespace {

    // Next token id from a logits row: greedy at temp<=0, else temperature + top-k + top-p.
    int64_t sampleLogits(const float *lg, int64_t vocab, float temp, int topK, float topP, std::mt19937 &rng) {
        if (temp <= 0.0f)
        {
            int64_t best = 0;
            float   bv   = lg[0];
            for (int64_t i = 1; i < vocab; ++i)
            {
                if (lg[i] > bv)
                {
                    bv   = lg[i];
                    best = i;
                }
            }
            return best;
        }
        std::vector<std::pair<float, int64_t>> v((size_t) vocab);
        for (int64_t i = 0; i < vocab; ++i)
        {
            v[(size_t) i] = {lg[i] / temp, i};
        }
        const int keep = topK > 0 && topK < (int) vocab ? topK : (int) vocab;
        std::partial_sort(v.begin(), v.begin() + keep, v.end(), [](const auto &a, const auto &b) {
            return a.first > b.first;
        });
        v.resize((size_t) keep);
        float mx = v[0].first, sum = 0.0f;
        for (auto &e: v)
        {
            e.first = std::exp(e.first - mx);
            sum += e.first;
        }
        float  cum = 0.0f;
        size_t n   = v.size();
        for (size_t i = 0; i < v.size(); ++i)
        {
            cum += v[i].first / sum;
            if (cum >= topP)
            {
                n = i + 1;
                break;
            }
        }
        float r   = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) * (sum * (cum > 0 ? cum : 1.0f));
        float acc = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            acc += v[i].first;
            if (acc >= r)
            {
                return v[i].second;
            }
        }
        return v[n - 1].second;
    }

    // One loaded decoder: the Session plus the persistent boundary tensors, the layer index maps, the
    // running absolute position, and the last logits row (so sampling is a separate call from stepping).
    struct Decoder {
        std::unique_ptr<Session> sess;
        std::vector<IOTensor>    inputs;  // persistent, in decode-bucket input order; KV buffers survive across steps
        std::vector<IOTensor>    outputs; // last run()'s outputs
        std::vector<IOInfo>      inInfo, outInfo; // the DECODE bucket's IO (io info at S=1)

        int              idIdx = -1, maskIdx = -1, posIdx = -1, logitsIdx = -1;
        std::vector<int> pastKey, pastVal, presKey, presVal;
        std::vector<int> outIdxByInfo; // outInfo index -> index in outputs vector (stable across runs)
        bool             mapped = false;

        int     L = 0, kvHeads = 0, C = 0, headDim = 0;
        int     presRows = 0; // rows the present outputs carry ([1,KV,presRows,HD]); the newest row is the last
        int64_t vocab    = 0;
        int     p        = 0; // absolute position across the whole conversation

        // Bucket roles of a multi-bucket .vxm: the decode bucket feeds input_ids [1,1]; the prefill
        // bucket (optional) feeds [1,S] with S>1 and processes a whole prompt window in one forward.
        // A single-bucket model has decodeBucket 0 and no prefill bucket.
        int decodeBucket    = 0;
        int prefillBucket   = -1;
        int prefillS        = 0; // the prefill bucket's window (tokens per forward)
        int prefillMaskLen  = 0; // its attention_mask length (C + prefillS)
        int prefillPresRows = 0; // rows its present outputs carry; the S new rows are the last

        // Engine-resident KV cache (Session::linkOutputToInput): every present output is linked to
        // its past input, the fold happens inside the engine (on-device on the GPU backend), and only
        // id/mask/position are bound per step. False = the host-side cache loop (link setup failed).
        bool kvLinked = false;
        // The next step rebinds the host past buffers with empty ranges, re-seeding the engine cache
        // — the reset path and the first decode step after a prefill pass.
        bool rebindPastNextStep = false;
        // A linked decode step ran: the engine-resident cache is ahead of the host past buffers.
        bool residentDirty = false;

        // Engine-side argmax over the decode bucket's logits (Session::setOutputArgMax): a decode
        // run returns the logits entry with NO data and readOutputArgMax() serves the greedy token.
        bool argMaxEngaged = false;
        // `logits` holds the last forward's full row (a prefill pass, or a decode without argmax).
        bool logitsValid = false;
        // Provenance of the last forward's logits: a decode step (argmax serves the token when
        // engaged) vs a prefill pass (always a full host row; its bucket is never registered).
        bool lastFromDecode = false;

        std::vector<float>    logits; // last forward's logits row (vocab floats), when logitsValid
        std::vector<IOTensor> prefillOutputs;
        std::mt19937          rng {1234};

        int findIn(const std::string &n) const {
            for (size_t i = 0; i < inInfo.size(); ++i)
            {
                if (inInfo[i].name == n)
                {
                    return (int) i;
                }
            }
            return -1;
        }
        int findOut(const std::string &n) const {
            for (size_t i = 0; i < outInfo.size(); ++i)
            {
                if (outInfo[i].name == n)
                {
                    return (int) i;
                }
            }
            return -1;
        }
        void setI64(int idx, const std::vector<int64_t> &vals) {
            if (idx < 0)
            {
                return; // an optional input the model does not expose (e.g. a GQA export's position_ids)
            }
            std::memcpy(inputs[(size_t) idx].data.data(), vals.data(), vals.size() * sizeof(int64_t));
        }
        bool isPastInput(int idx) const {
            for (int l = 0; l < L; ++l)
            {
                if (idx == pastKey[l] || idx == pastVal[l])
                {
                    return true;
                }
            }
            return false;
        }

        // Declare the KV links on the DECODE bucket: every present output feeds its past input on
        // the next run. Empty ranges to start (the cache begins as zeros); step() re-links per token
        // with the fold ranges. The prefill bucket keeps the host cache flow (it runs once per
        // prompt window). On failure the decoder stays on the host-side cache loop.
        void setupKvLinks() {
            for (int l = 0; l < L; ++l)
            {
                if (sess->linkOutputToInput((size_t) decodeBucket, outInfo[(size_t) presKey[l]].name, inInfo[(size_t) pastKey[l]].name, {}) != Status::Ok ||
                    sess->linkOutputToInput((size_t) decodeBucket, outInfo[(size_t) presVal[l]].name, inInfo[(size_t) pastVal[l]].name, {}) != Status::Ok)
                {
                    LOGE("KV link setup failed at layer %d; using the host cache loop", l);
                    sess->clearLinks();
                    kvLinked = false;
                    return;
                }
            }
            kvLinked = true;
        }

        // Update every link's ranges: fold the previous token's present row (the LAST present row —
        // index C for a cache-concat present, 0 for a rows-only present) into cache slot `slot`, or
        // clear the pending fold when `slot` < 0 (reset/first step).
        bool setKvFoldSlot(int64_t slot) {
            const std::vector<LinkRange> ranges = kvFoldRanges(kvHeads, presRows, C, headDim, slot);
            for (int l = 0; l < L; ++l)
            {
                for (int part = 0; part < 2; ++part)
                {
                    const std::string &pres = outInfo[(size_t) (part ? presVal[l] : presKey[l])].name;
                    const std::string &past = inInfo[(size_t) (part ? pastVal[l] : pastKey[l])].name;
                    const Status       st   = sess->linkOutputToInput((size_t) decodeBucket, pres, past, ranges);
                    if (st != Status::Ok)
                    {
                        LOGE("KV link update failed for %s -> %s at slot %lld: %s (engine log has the reason)", pres.c_str(), past.c_str(), (long long) slot, statusStr(st));
                        return false;
                    }
                }
            }
            return true;
        }

        // Rebuild the host `inputs` past buffers from the engine-resident state — the device cache
        // plus the pending fold from the last run's present — so the host cache loop can take over
        // MID-STREAM with no token divergence. Call while the links still exist (readResident
        // resolves linked names only).
        bool resyncHostCache() {
            const int64_t pendingSlot = p > 0 ? std::min<int64_t>(p - 1, C - 1) : -1;
            for (int l = 0; l < L; ++l)
            {
                for (int part = 0; part < 2; ++part)
                {
                    const std::string &presName = outInfo[(size_t) (part ? presVal[l] : presKey[l])].name;
                    const std::string &pastName = inInfo[(size_t) (part ? pastVal[l] : pastKey[l])].name;
                    IOTensor           resident;
                    if (sess->readResident(pastName, resident) != Status::Ok)
                    {
                        return false;
                    }
                    IOTensor &hostPast = inputs[(size_t) (part ? pastVal[l] : pastKey[l])];
                    if (resident.data.size() != hostPast.data.size())
                    {
                        LOGE("resync: resident %s holds %zu bytes, host expects %zu", pastName.c_str(), resident.data.size(), hostPast.data.size());
                        return false;
                    }
                    std::memcpy(hostPast.data.data(), resident.data.data(), resident.data.size());
                    if (pendingSlot >= 0)
                    {
                        IOTensor present;
                        if (sess->readResident(presName, present) != Status::Ok)
                        {
                            return false;
                        }
                        const float *src = reinterpret_cast<const float *>(present.data.data());
                        float       *dst = reinterpret_cast<float *>(hostPast.data.data());
                        for (int h = 0; h < kvHeads; ++h)
                        {
                            const float *s = src + ((size_t) h * presRows + (presRows - 1)) * headDim;
                            float       *d = dst + ((size_t) h * C + pendingSlot) * headDim;
                            std::memcpy(d, s, (size_t) headDim * sizeof(float));
                        }
                    }
                }
            }
            return true;
        }

        // Feed one token at the current position: write id/pos/mask, run the plan on the GPU, fold the
        // new key/value into the KV cache (in-engine when linked), and keep the logits row. Returns 0
        // on success, -1 on error.
        int step(int64_t tok) {
            setI64(idIdx, {tok});
            setI64(posIdx, {(int64_t) p});
            std::vector<int64_t> am((size_t) C + 1, 0);
            for (int j = 0; j < p && j < C; ++j)
            {
                am[(size_t) j] = 1; // valid past slots
            }
            am[(size_t) C] = 1; // the current token (present slot C)
            setI64(maskIdx, am);

            Status runStatus = Status::Ok;
            bool   ranLinked = false;
            if (kvLinked)
            {
                // The engine folds the PREVIOUS token's row into slot p-1 at the start of this run;
                // a reset step clears the pending fold and rebinds the zeroed past buffers instead.
                const bool rebind = rebindPastNextStep;
                if (setKvFoldSlot(rebind || p == 0 ? -1 : std::min<int64_t>(p - 1, C - 1)))
                {
                    std::vector<IOTensor> bound {inputs[(size_t) idIdx], inputs[(size_t) maskIdx]};
                    if (posIdx >= 0)
                    {
                        bound.push_back(inputs[(size_t) posIdx]); // position_ids only when the model exposes it
                    }
                    if (rebind)
                    {
                        for (int l = 0; l < L; ++l)
                        {
                            bound.push_back(inputs[(size_t) pastKey[l]]);
                            bound.push_back(inputs[(size_t) pastVal[l]]);
                        }
                    }
                    runStatus          = sess->run(bound, outputs);
                    rebindPastNextStep = false;
                    ranLinked          = true;
                    residentDirty      = true;
                } else
                {
                    // A mid-stream link failure: bring the engine-resident cache (device rows + the
                    // pending fold) back into the host buffers, drop the links, and continue THIS and
                    // every later step on the host cache loop — same tokens, no lost state.
                    LOGE("switching to the host KV loop at p=%d (resyncing the cache from the engine)", p);
                    if (!resyncHostCache())
                    {
                        LOGE("host cache resync failed; cannot continue");
                        return -1;
                    }
                    sess->clearLinks();
                    kvLinked           = false;
                    rebindPastNextStep = false;
                    residentDirty      = false;
                }
            }
            if (!ranLinked)
            {
                runStatus = sess->run(inputs, outputs);
            }
            if (runStatus != Status::Ok)
            {
                LOGE("run() failed at p=%d", p);
                return -1;
            }
            if (!mapped)
            {
                outIdxByInfo.assign(outInfo.size(), -1);
                for (size_t j = 0; j < outputs.size(); ++j)
                {
                    for (size_t k = 0; k < outInfo.size(); ++k)
                    {
                        if (outputs[j].name == outInfo[k].name)
                        {
                            outIdxByInfo[k] = (int) j;
                        }
                    }
                }
                mapped = true;
            }
            if (!kvLinked)
            {
                const int slot = p < C ? p : C - 1; // guard the rare overrun past the compiled context
                for (int l = 0; l < L; ++l)
                {
                    for (int part = 0; part < 2; ++part)
                    {
                        const IOTensor &pres = outputs[(size_t) outIdxByInfo[(size_t) (part ? presVal[l] : presKey[l])]];
                        IOTensor       &past = inputs[(size_t) (part ? pastVal[l] : pastKey[l])];
                        const float    *src  = reinterpret_cast<const float *>(pres.data.data());
                        float          *dst  = reinterpret_cast<float *>(past.data.data());
                        for (int h = 0; h < kvHeads; ++h)
                        {
                            const float *s = src + ((size_t) h * presRows + (presRows - 1)) * headDim;
                            float       *d = dst + ((size_t) h * C + slot) * headDim;
                            std::memcpy(d, s, (size_t) headDim * sizeof(float));
                        }
                    }
                }
            }
            // Under the engine argmax the decode logits entry carries no data (readOutputArgMax
            // serves the token); otherwise keep the full row for host sampling.
            const IOTensor &lo = outputs[(size_t) outIdxByInfo[(size_t) logitsIdx]];
            if (!lo.data.empty())
            {
                const float *lp = reinterpret_cast<const float *>(lo.data.data());
                logits.assign(lp, lp + vocab);
                logitsValid = true;
            } else
            {
                logitsValid = false;
            }
            lastFromDecode = true;
            return 0;
        }

        // One prefill forward: `len` (<= prefillS) prompt tokens at absolute positions p..p+len-1
        // through the prefill bucket, on the HOST cache flow — bind the past buffers, fold the
        // produced present rows (the len new rows after the past block) back into cache slots
        // p..p+len-1 — leaving the last real token's logits row in `logits`. Pad slots carry mask 0,
        // so real tokens never attend them and their rows are not folded. Advances p.
        bool prefillPass(const int64_t *toks, int len) {
            std::vector<IOTensor> bound(inInfo.size());
            for (size_t i = 0; i < inInfo.size(); ++i)
            {
                bound[i].name  = inInfo[i].name;
                bound[i].dtype = inInfo[i].dtype;
                if (isPastInput((int) i))
                {
                    bound[i].shape = inInfo[i].shape; // shared shape with the decode bucket
                    bound[i].data  = inputs[i].data;  // the host KV cache
                    continue;
                }
                if ((int) i == idIdx || (int) i == posIdx)
                {
                    bound[i].shape = {1, (int64_t) prefillS};
                    std::vector<int64_t> vals((size_t) prefillS, 0);
                    for (int t = 0; t < prefillS; ++t)
                    {
                        vals[(size_t) t] = ((int) i == idIdx) ? (t < len ? toks[t] : 0) : (int64_t) (p + t);
                    }
                    bound[i].data.resize((size_t) prefillS * sizeof(int64_t));
                    std::memcpy(bound[i].data.data(), vals.data(), bound[i].data.size());
                } else if ((int) i == maskIdx)
                {
                    bound[i].shape = {1, (int64_t) prefillMaskLen};
                    std::vector<int64_t> mask((size_t) prefillMaskLen, 0);
                    for (int j = 0; j < p && j < C; ++j)
                    {
                        mask[(size_t) j] = 1; // valid past slots
                    }
                    for (int t = 0; t < len; ++t)
                    {
                        mask[(size_t) (C + t)] = 1; // the window's real tokens; pads stay masked
                    }
                    bound[i].data.resize((size_t) prefillMaskLen * sizeof(int64_t));
                    std::memcpy(bound[i].data.data(), mask.data(), bound[i].data.size());
                }
            }
            if (sess->run(bound, prefillOutputs) != Status::Ok)
            {
                LOGE("prefill run failed at p=%d", p);
                return false;
            }
            auto prefillOutByName = [&](const std::string &n) -> const IOTensor * {
                for (const IOTensor &o: prefillOutputs)
                {
                    if (o.name == n)
                    {
                        return &o;
                    }
                }
                return nullptr;
            };
            const IOTensor *logitsOut = prefillOutByName("logits");
            if (!logitsOut || logitsOut->data.empty())
            {
                LOGE("prefill logits missing");
                return false;
            }
            const int newRowsAt = prefillPresRows - prefillS;
            for (int l = 0; l < L; ++l)
            {
                for (int part = 0; part < 2; ++part)
                {
                    const IOTensor *pres = prefillOutByName(outInfo[(size_t) (part ? presVal[l] : presKey[l])].name);
                    if (!pres)
                    {
                        LOGE("prefill present outputs missing");
                        return false;
                    }
                    const float *src = reinterpret_cast<const float *>(pres->data.data());
                    float       *dst = reinterpret_cast<float *>(inputs[(size_t) (part ? pastVal[l] : pastKey[l])].data.data());
                    for (int h = 0; h < kvHeads; ++h)
                    {
                        for (int t = 0; t < len; ++t)
                        {
                            std::memcpy(dst + ((size_t) h * C + p + t) * headDim,
                                        src + ((size_t) h * prefillPresRows + newRowsAt + t) * headDim,
                                        (size_t) headDim * sizeof(float));
                        }
                    }
                }
            }
            const float *lp = reinterpret_cast<const float *>(logitsOut->data.data()) + (size_t) (len - 1) * vocab;
            logits.assign(lp, lp + vocab);
            logitsValid    = true;
            lastFromDecode = false;
            p += len;
            return true;
        }

        // Feed `count` prompt tokens from the current position: whole-window forwards through the
        // prefill bucket when the model carries one, decode-bucket steps otherwise. The prefill
        // passes run on the host cache flow, so the engine-resident decode cache syncs to the host
        // first and re-seeds on the next decode step. Returns 0 ok, -1 error, -2 context full.
        int prefill(const int64_t *ids, int count) {
            if (count <= 0)
            {
                return 0;
            }
            if (p + count > C - 1) // no slot would remain for the reply's first decode step
            {
                return -2;
            }
            int consumed = 0;
            if (prefillBucket >= 0)
            {
                if (kvLinked && residentDirty)
                {
                    if (!resyncHostCache())
                    {
                        LOGE("resident cache resync failed before prefill");
                        return -1;
                    }
                    residentDirty = false;
                }
                while (consumed < count)
                {
                    const int len = (int) std::min<int64_t>((int64_t) (count - consumed), (int64_t) prefillS);
                    if (!prefillPass(ids + consumed, len))
                    {
                        return -1;
                    }
                    consumed += len;
                }
                rebindPastNextStep = kvLinked && consumed > 0; // the next decode step re-seeds the engine cache
            }
            for (; consumed < count; ++consumed)
            {
                if (step(ids[consumed]) != 0)
                {
                    return -1;
                }
                ++p;
            }
            return 0;
        }

        // Next token: the engine-side argmax serves a greedy decode step (8-byte readback); the
        // stored full logits row serves everything else (prefill logits, host-argmax fallback,
        // temperature sampling). A decode row under the engine argmax is never downloaded, so a
        // temp > 0 request then falls back to the greedy token with a log line.
        int64_t sample(float temp, int topK, float topP) {
            if (argMaxEngaged && lastFromDecode)
            {
                int64_t index = 0;
                float   best  = 0.0f;
                if (sess->readOutputArgMax("logits", index, best) == Status::Ok)
                {
                    if (temp > 0.0f)
                    {
                        LOGE("temp=%.2f requested but the decode logits reduce engine-side; using the greedy token (reload to sample)", (double) temp);
                    }
                    return index;
                }
                LOGE("engine argmax readback failed");
            }
            if (!logitsValid)
            {
                LOGE("no logits row available to sample");
                return -1;
            }
            return sampleLogits(logits.data(), vocab, temp, topK, topP, rng);
        }
    };

    constexpr float kMaskFill = -1e4f; // fp16-safe additive mask fill: exp(-1e4) underflows to exactly 0
                                       // in fp16; -FLT_MAX/-65504 overflow to inf across the fp16 boundary
                                       // and 0*inf = NaN poisons the softmax.

    // One loaded vision-language model (mirrors examples/llm/vlm.cpp): a multi-graph .vxm with a
    // vision bucket (pixel_values -> image embeddings), token-embedding buckets at S=prefill and
    // S=1, and text-decoder buckets at the same two shapes, dispatched by bound input names+shapes.
    // The past entries of `dec` ARE the KV cache (fp32 host boundary, persistent across the whole
    // conversation); present rows fold back in after every run.
    struct Vlm {
        std::unique_ptr<Session> sess;
        std::vector<IOInfo>      visIn, visOut, decIn, decOut;
        std::vector<IOTensor>    dec;     // persistent decoder inputs, re-shaped per call (S=prefill / S=1)
        std::vector<IOTensor>    visInT;  // single "pixel_values" tensor
        std::vector<IOTensor>    outs;    // last run()'s outputs
        IOTensor                 imgHidden, imgPos; // image-turn extras: vision features + their row positions

        std::vector<int> pastKey, pastVal;
        int              idsIdx = -1, maskIdx = -1, posIdx = -1;
        int              imgPrefillBucket = -1; // -1 when the model has no image-prefill bucket
        int              L = 0, kvHeads = 0, C = 0, headDim = 0, H = 0;
        int              prefillS = 0, imgRows = 0, imgSide = 0;
        int64_t          vocab = 0;
        int              p     = 0; // absolute token position across the conversation

        // Engine-resident KV cache for the S=1 DECODE bucket (Session::linkOutputToInput): during a
        // decode run the fold happens inside the engine and only embeds/mask/positions are bound.
        // The PREFILL bucket keeps the host cache flow (it runs once per turn), so at each turn
        // boundary the device state materializes back into `dec` via readResident().
        int  decodeBucket       = -1;
        int  presRowsDecode     = 0;     // rows the DECODE bucket's present outputs carry ([1,KV,rows,HD])
        bool decodeLinked       = false; // links declared on the decode bucket
        bool cacheOnDevice      = false; // the decode bucket's resident past is ahead of dec[]
        bool rebindPastNextStep = false; // next decode step re-seeds the device cache from dec[]
        int  pendingSlot        = -1;    // cache slot of the fold the next decode run applies

        std::vector<float> logits; // last run's next-token logits row
        std::mt19937       rng {1234};

        const IOTensor *outByName(const std::string &n) const {
            for (const IOTensor &o: outs)
            {
                if (o.name == n)
                {
                    return &o;
                }
            }
            return nullptr;
        }

        void setDecShape(int idx, const Shape &s, DType dt) {
            dec[(size_t) idx].shape = s;
            dec[(size_t) idx].dtype = dt;
            dec[(size_t) idx].data.assign((size_t) numElements(s) * dtypeSize(dt), 0);
        }

        void pastName(int layer, int part, char (&buf)[64]) const {
            snprintf(buf, sizeof buf, part ? "past_key_values.%d.value" : "past_key_values.%d.key", layer);
        }
        void presentName(int layer, int part, char (&buf)[64]) const {
            snprintf(buf, sizeof buf, part ? "present.%d.value" : "present.%d.key", layer);
        }

        // Declare the decode-bucket KV links (empty ranges; per-step folds arrive via
        // setDecodeFoldSlot). On failure the decode loop stays on the host cache path.
        void setupDecodeLinks() {
            for (int l = 0; l < L; ++l)
            {
                for (int part = 0; part < 2; ++part)
                {
                    char pastBuf[64], presBuf[64];
                    pastName(l, part, pastBuf);
                    presentName(l, part, presBuf);
                    if (sess->linkOutputToInput((size_t) decodeBucket, presBuf, pastBuf, {}) != Status::Ok)
                    {
                        LOGE("VLM KV link setup failed at layer %d; using the host cache loop", l);
                        sess->clearLinks();
                        decodeLinked = false;
                        return;
                    }
                }
            }
            decodeLinked = true;
        }

        // Update every decode link: fold the previous run's present row into cache slot `slot`, or
        // clear the pending fold when `slot` < 0. The source row is the LAST row of the decode
        // bucket's present output — its row count comes from that bucket's own output shape
        // (presRowsDecode), never assumed: this decoder's present carries only the produced rows
        // ([1,KV,1,HD] at S=1), unlike a cache-concat decoder's [1,KV,C+1,HD].
        bool setDecodeFoldSlot(int64_t slot) {
            const std::vector<LinkRange> ranges = kvFoldRanges(kvHeads, presRowsDecode, C, headDim, slot);
            for (int l = 0; l < L; ++l)
            {
                for (int part = 0; part < 2; ++part)
                {
                    char pastBuf[64], presBuf[64];
                    pastName(l, part, pastBuf);
                    presentName(l, part, presBuf);
                    const Status st = sess->linkOutputToInput((size_t) decodeBucket, presBuf, pastBuf, ranges);
                    if (st != Status::Ok)
                    {
                        LOGE("VLM KV link update failed for %s -> %s at slot %lld: %s (engine log has the reason)", presBuf, pastBuf, (long long) slot, statusStr(st));
                        return false;
                    }
                }
            }
            return true;
        }

        // Copy the decode bucket's device-resident cache (plus the pending fold) back into the host
        // `dec` past buffers, so the prefill bucket sees the full conversation state.
        bool materializeDeviceCache() {
            for (int l = 0; l < L; ++l)
            {
                for (int part = 0; part < 2; ++part)
                {
                    char pastBuf[64], presBuf[64];
                    pastName(l, part, pastBuf);
                    presentName(l, part, presBuf);
                    IOTensor resident;
                    if (sess->readResident(pastBuf, resident) != Status::Ok)
                    {
                        return false;
                    }
                    IOTensor &hostPast = dec[(size_t) (part ? pastVal[l] : pastKey[l])];
                    if (resident.data.size() != hostPast.data.size())
                    {
                        LOGE("resident cache size mismatch for %s", pastBuf);
                        return false;
                    }
                    std::memcpy(hostPast.data.data(), resident.data.data(), resident.data.size());
                    if (pendingSlot >= 0)
                    {
                        IOTensor present;
                        if (sess->readResident(presBuf, present) != Status::Ok)
                        {
                            return false;
                        }
                        const float *src = reinterpret_cast<const float *>(present.data.data());
                        float       *dst = reinterpret_cast<float *>(hostPast.data.data());
                        for (int h = 0; h < kvHeads; ++h)
                        {
                            const float *s = src + ((size_t) h * presRowsDecode + (presRowsDecode - 1)) * headDim;
                            float       *d = dst + ((size_t) h * C + pendingSlot) * headDim;
                            std::memcpy(d, s, (size_t) headDim * sizeof(float));
                        }
                    }
                }
            }
            pendingSlot = -1;
            return true;
        }

        // Copy present rows [1,KV,S,HD] (rows 0..n-1) into cache slots startSlot.. of every layer.
        bool foldPresent(int S, int n, int startSlot) {
            for (int l = 0; l < L; ++l)
            {
                char pk[64], pv[64];
                snprintf(pk, sizeof pk, "present.%d.key", l);
                snprintf(pv, sizeof pv, "present.%d.value", l);
                for (int part = 0; part < 2; ++part)
                {
                    const IOTensor *pres = outByName(part ? pv : pk);
                    if (!pres)
                    {
                        return false;
                    }
                    const float *src = reinterpret_cast<const float *>(pres->data.data());
                    float       *dst = reinterpret_cast<float *>(dec[(size_t) (part ? pastVal[l] : pastKey[l])].data.data());
                    for (int h = 0; h < kvHeads; ++h)
                    {
                        for (int r = 0; r < n; ++r)
                        {
                            const float *sp = src + ((size_t) h * S + r) * headDim;
                            float       *dp = dst + ((size_t) h * C + startSlot + r) * headDim;
                            std::memcpy(dp, sp, (size_t) headDim * sizeof(float));
                        }
                    }
                }
            }
            return true;
        }

        // Run the vision bucket on fp32 CHW pixels; the image-embedding rows land in `out`.
        int visionEncode(const float *pixels, size_t nElems, std::vector<float> &out) {
            if ((int64_t) nElems != numElements(visInT[0].shape))
            {
                LOGE("vision input %zu floats, want %lld", nElems, (long long) numElements(visInT[0].shape));
                return -1;
            }
            visInT[0].data.resize(nElems * sizeof(float));
            std::memcpy(visInT[0].data.data(), pixels, nElems * sizeof(float));
            if (sess->run(visInT, outs) != Status::Ok)
            {
                LOGE("vision run failed");
                return -1;
            }
            const IOTensor *e = outByName(visOut[0].name);
            if (!e)
            {
                LOGE("vision output %s missing", visOut[0].name.c_str());
                return -1;
            }
            const float *f = reinterpret_cast<const float *>(e->data.data());
            out.assign(f, f + (size_t) imgRows * H);
            return 0;
        }

        // Prefill one prompt: bind the padded ids straight to the decoder (the embedding lookup is fused
        // in), build the additive mask + clamped positions, run the prefill bucket, and fold the real
        // present rows. For an image turn the vision features and their row positions are bound too, which
        // selects the image bucket where an on-GPU ScatterND splices them in. Pad rows are masked causally
        // anyway but are NEVER folded into the cache.
        int prefill(const int64_t *ids, int nReal, const float *imgEmb, int imgRowsGiven, int64_t imageToken, int64_t padId) {
            if (nReal <= 0 || nReal > prefillS)
            {
                LOGE("prompt %d tokens, prefill window %d", nReal, prefillS);
                return -1;
            }
            if (p + nReal + 1 > C)
            {
                LOGE("context full (%d + %d > %d)", p, nReal, C);
                return -2;
            }
            // A turn boundary: the decode loop left the cache device-resident; bring it back to the
            // host `dec` buffers (with the pending fold applied) for the prefill bucket, and mark the
            // next decode step to re-seed the device cache.
            if (decodeLinked && cacheOnDevice)
            {
                if (!materializeDeviceCache())
                {
                    LOGE("device cache materialization failed");
                    return -1;
                }
                cacheOnDevice      = false;
                rebindPastNextStep = true;
            }
            std::vector<int64_t> padded(ids, ids + nReal);
            padded.resize((size_t) prefillS, padId);
            setDecShape(idsIdx, {1, (int64_t) prefillS}, DType::Int64);
            std::memcpy(dec[(size_t) idsIdx].data.data(), padded.data(), (size_t) prefillS * sizeof(int64_t));

            // Image turn: the prompt rows holding image tokens. The decoder's on-GPU ScatterND overwrites
            // those rows of the internally computed token embeddings with the vision feature rows.
            std::vector<int> imageRows;
            for (int q = 0; q < nReal; ++q)
            {
                if (ids[q] == imageToken)
                {
                    imageRows.push_back(q);
                }
            }
            const bool imageTurn = !imageRows.empty();
            if (imageTurn && (!imgEmb || (int) imageRows.size() != imgRows || imgRowsGiven < imgRows || imgPrefillBucket < 0))
            {
                LOGE("image prompt needs %d image rows and an image bucket (have %d tokens / %d given, bucket %d)", imgRows, (int) imageRows.size(), imgRowsGiven, imgPrefillBucket);
                return -1;
            }

            // Additive mask [1,1,S,C+S]: past slots < p visible to every row, new tokens causal.
            setDecShape(maskIdx, {1, 1, (int64_t) prefillS, (int64_t) (C + prefillS)}, DType::Float32);
            float *m = reinterpret_cast<float *>(dec[(size_t) maskIdx].data.data());
            for (int q = 0; q < prefillS; ++q)
            {
                float *row = m + (size_t) q * (C + prefillS);
                for (int c = 0; c < C; ++c)
                {
                    row[c] = c < p ? 0.0f : kMaskFill;
                }
                for (int j = 0; j < prefillS; ++j)
                {
                    row[C + j] = j <= q ? 0.0f : kMaskFill;
                }
            }
            setDecShape(posIdx, {1, (int64_t) prefillS}, DType::Int64);
            int64_t *pos = reinterpret_cast<int64_t *>(dec[(size_t) posIdx].data.data());
            for (int q = 0; q < prefillS; ++q)
            {
                pos[q] = p + (q < nReal ? q : nReal - 1); // pad rows clamp; they are never folded
            }

            // An image turn also binds the vision features and their row positions, landing run() on the
            // image bucket (the on-GPU ScatterND does the splice); a text turn lands on the plain bucket.
            std::vector<IOTensor> bound(dec);
            if (imageTurn)
            {
                imgHidden.name  = "image_hidden_states";
                imgHidden.shape = {1, (int64_t) imgRows, (int64_t) H};
                imgHidden.dtype = DType::Float32;
                imgHidden.data.assign(reinterpret_cast<const uint8_t *>(imgEmb), reinterpret_cast<const uint8_t *>(imgEmb) + (size_t) imgRows * H * sizeof(float));
                imgPos.name  = "image_positions";
                imgPos.shape = {1, (int64_t) imgRows, 2};
                imgPos.dtype = DType::Float32;
                imgPos.data.assign((size_t) imgRows * 2 * sizeof(float), 0);
                float *positions2d = reinterpret_cast<float *>(imgPos.data.data());
                for (int i = 0; i < imgRows; ++i)
                {
                    positions2d[i * 2 + 0] = 0.0f;                          // batch
                    positions2d[i * 2 + 1] = (float) imageRows[(size_t) i]; // sequence row overwritten
                }
                bound.push_back(imgHidden);
                bound.push_back(imgPos);
            }
            if (sess->run(bound, outs) != Status::Ok)
            {
                LOGE("prefill run failed");
                return -1;
            }
            if (!foldPresent(prefillS, nReal, p))
            {
                LOGE("prefill present outputs missing");
                return -1;
            }
            const IOTensor *lg = outByName("logits");
            if (!lg)
            {
                return -1;
            }
            p += nReal;
            const float *lp = reinterpret_cast<const float *>(lg->data.data()) + (size_t) (nReal - 1) * vocab;
            logits.assign(lp, lp + vocab);
            return 0;
        }

        // One decode step: token id -> decoder S=1 bucket (embedding lookup fused in) -> logits row.
        int step(int64_t tok) {
            if (p + 1 > C - 1)
            {
                LOGE("context full at p=%d", p);
                return -2;
            }
            setDecShape(idsIdx, {1, 1}, DType::Int64);
            std::memcpy(dec[(size_t) idsIdx].data.data(), &tok, sizeof(int64_t));
            setDecShape(maskIdx, {1, 1, 1, (int64_t) C + 1}, DType::Float32);
            float *m = reinterpret_cast<float *>(dec[(size_t) maskIdx].data.data());
            for (int c = 0; c < C + 1; ++c)
            {
                m[c] = (c < p || c == C) ? 0.0f : kMaskFill;
            }
            setDecShape(posIdx, {1, 1}, DType::Int64);
            int64_t pos = p;
            std::memcpy(dec[(size_t) posIdx].data.data(), &pos, sizeof(int64_t));
            const int slot      = p < C ? p : C - 1;
            bool      ranLinked = false;
            if (decodeLinked)
            {
                // The engine folds the previous run's row into `pendingSlot` at the start of this
                // run. The first decode step of a turn re-seeds the device cache from the host `dec`
                // past (which prefill just updated) and clears any pending fold.
                const bool rebind = rebindPastNextStep || !cacheOnDevice;
                if (setDecodeFoldSlot(rebind ? -1 : pendingSlot))
                {
                    std::vector<IOTensor> bound {dec[(size_t) idsIdx], dec[(size_t) maskIdx], dec[(size_t) posIdx]};
                    if (rebind)
                    {
                        for (int l = 0; l < L; ++l)
                        {
                            bound.push_back(dec[(size_t) pastKey[l]]);
                            bound.push_back(dec[(size_t) pastVal[l]]);
                        }
                    }
                    if (sess->run(bound, outs) != Status::Ok)
                    {
                        return -1;
                    }
                    rebindPastNextStep = false;
                    cacheOnDevice      = true;
                    pendingSlot        = slot;
                    ranLinked          = true;
                } else
                {
                    // A mid-stream link failure: bring the engine-resident cache (device rows + the
                    // pending fold) back into the host `dec` buffers, drop the links, and continue
                    // THIS and every later step on the host cache loop — same tokens, no lost state.
                    LOGE("switching to the host KV loop at p=%d (resyncing the cache from the engine)", p);
                    if (cacheOnDevice && !materializeDeviceCache())
                    {
                        LOGE("host cache resync failed; cannot continue");
                        return -1;
                    }
                    sess->clearLinks();
                    decodeLinked       = false;
                    cacheOnDevice      = false;
                    rebindPastNextStep = false;
                }
            }
            if (!ranLinked)
            {
                if (sess->run(dec, outs) != Status::Ok)
                {
                    return -1;
                }
                if (!foldPresent(1, 1, slot))
                {
                    return -1;
                }
            }
            const IOTensor *lg = outByName("logits");
            if (!lg)
            {
                return -1;
            }
            const float *lp = reinterpret_cast<const float *>(lg->data.data());
            logits.assign(lp, lp + vocab);
            ++p;
            return 0;
        }
    };

    int findByName(const std::vector<IOInfo> &v, const std::string &n) {
        for (size_t i = 0; i < v.size(); ++i)
        {
            if (v[i].name == n)
            {
                return (int) i;
            }
        }
        return -1;
    }

    Precision precFromStr(const std::string &s) {
        if (s == "normal")
        {
            return Precision::Normal;
        }
        if (s == "high")
        {
            return Precision::High;
        }
        return Precision::Low;
    }

    std::string jstr(JNIEnv *env, jstring s) {
        if (!s)
        {
            return {};
        }
        const char *c = env->GetStringUTFChars(s, nullptr);
        std::string out(c ? c : "");
        if (c)
        {
            env->ReleaseStringUTFChars(s, c);
        }
        return out;
    }

    // Raises a Java RuntimeException carrying `message`; the pending exception surfaces when the
    // JNI call returns to Kotlin.
    void throwJavaRuntime(JNIEnv *env, const std::string &message) {
        jclass runtimeException = env->FindClass("java/lang/RuntimeException");
        if (runtimeException)
        {
            env->ThrowNew(runtimeException, message.c_str());
        }
    }

} // namespace

extern "C" {

// Load the .vxm and build a Decoder. `greedyArgMax` registers the decode bucket's logits for the
// engine-side argmax reduction — the registration holds for the session's lifetime and the decode
// logits row is then never downloaded, so it is requested only when the app decodes greedy.
// Returns a native handle (0 on failure).
JNIEXPORT jlong JNICALL Java_com_vknn_chat_NativeLib_nativeInit(JNIEnv *env, jobject, jstring jvxm, jstring jcache, jstring jprec, jstring jbackend, jboolean greedyArgMax) {
    auto *d = new Decoder();
    try
    {
        Config cfg;
        cfg.backend                = backendFromStr(jstr(env, jbackend));
        cfg.precision              = precFromStr(jstr(env, jprec));
        cfg.freeWeightsAfterUpload = true;
        d->sess                    = Runtime::load(jstr(env, jvxm), cfg, jstr(env, jcache));
        if (!d->sess)
        {
            LOGE("Runtime::load failed");
            delete d;
            return 0;
        }
        // Bucket roles (mirrors examples/llm/chat.cpp): the decode bucket feeds input_ids [1,1]; a
        // prefill bucket (optional) feeds [1,S] with S>1 — the widest one wins. A single-bucket
        // model is its own decode bucket.
        int decodeB = -1, prefillB = -1, prefillS = 0;
        for (size_t b = 0; b < d->sess->bucketCount(); ++b)
        {
            for (const IOInfo &in: d->sess->inputInfo(b))
            {
                if (in.name == "input_ids" && in.shape.size() == 2)
                {
                    if (in.shape[1] == 1 && decodeB < 0)
                    {
                        decodeB = (int) b;
                    } else if (in.shape[1] > 1 && (int) in.shape[1] > prefillS)
                    {
                        prefillB = (int) b;
                        prefillS = (int) in.shape[1];
                    }
                }
            }
        }
        if (decodeB < 0)
        {
            LOGE("model has no input_ids [1,1] decode bucket");
            delete d;
            return 0;
        }
        d->decodeBucket = decodeB;
        d->inInfo       = d->sess->inputInfo((size_t) decodeB);
        d->outInfo      = d->sess->outputInfo((size_t) decodeB);

        d->idIdx     = d->findIn("input_ids");
        d->maskIdx   = d->findIn("attention_mask");
        d->posIdx    = d->findIn("position_ids");
        d->logitsIdx = d->findOut("logits");
        for (int l = 0;; ++l)
        {
            char kb[64], vb[64], pk[64], pv[64];
            snprintf(kb, sizeof kb, "past_key_values.%d.key", l);
            snprintf(vb, sizeof vb, "past_key_values.%d.value", l);
            const int ik = d->findIn(kb), iv = d->findIn(vb);
            if (ik < 0 || iv < 0)
            {
                break;
            }
            snprintf(pk, sizeof pk, "present.%d.key", l);
            snprintf(pv, sizeof pv, "present.%d.value", l);
            d->pastKey.push_back(ik);
            d->pastVal.push_back(iv);
            d->presKey.push_back(d->findOut(pk));
            d->presVal.push_back(d->findOut(pv));
        }
        d->L = (int) d->pastKey.size();
        if (d->idIdx < 0 || d->maskIdx < 0 || d->logitsIdx < 0 || d->L == 0)
        {
            LOGE("model is not a with-past decoder (needs input_ids, attention_mask, logits, past_key_values.*)");
            delete d;
            return 0;
        }
        // position_ids is optional: a GQA export (e.g. Llama-3.2) derives position from the mask
        // internally and exposes no position_ids input, so posIdx < 0 is valid — the decode step just
        // skips binding it.
        const Shape &ks = d->inInfo[(size_t) d->pastKey[0]].shape; // [1, kv_heads, C, head_dim]
        d->kvHeads      = (int) ks[1];
        d->C            = (int) ks[2];
        d->headDim      = (int) ks[3];
        // Present rows from the PRESENT output's own shape, never assumed: a cache-concat decoder
        // carries C+1 rows, a rows-only decoder carries exactly the step's rows.
        if (d->presKey[0] >= 0 && d->outInfo[(size_t) d->presKey[0]].shape.size() == 4)
        {
            d->presRows = (int) d->outInfo[(size_t) d->presKey[0]].shape[2];
        }
        d->vocab = d->outInfo[(size_t) d->logitsIdx].shape.back();
        d->logits.assign((size_t) d->vocab, 0.0f);
        if (d->presRows <= 0)
        {
            LOGE("present outputs are missing or not [1,KV,rows,HD]; cannot drive the KV fold");
            delete d;
            return 0;
        }

        // The whole-window prefill fold is keyed to position_ids; a model that derives position
        // internally (no position_ids input) prefills token-by-token through the decode bucket instead
        // — the same fallback examples/llm/chat.cpp takes.
        if (d->posIdx < 0)
        {
            prefillB = -1;
        }
        // Prefill-bucket geometry, validated against the decode bucket: the past inputs must share
        // the decode shapes (one host cache serves both) and the mask must span past+S columns. Any
        // mismatch disables the fast prefill rather than miscomputing.
        if (prefillB >= 0)
        {
            bool geometryOk = true;
            for (const IOInfo &in: d->sess->inputInfo((size_t) prefillB))
            {
                if (in.name == d->inInfo[(size_t) d->pastKey[0]].name)
                {
                    geometryOk = geometryOk && in.shape == d->inInfo[(size_t) d->pastKey[0]].shape;
                }
                if (in.name == "attention_mask" && in.shape.size() == 2)
                {
                    d->prefillMaskLen = (int) in.shape[1];
                }
            }
            bool prefillHasLogits = false;
            for (const IOInfo &out: d->sess->outputInfo((size_t) prefillB))
            {
                if (out.name == "logits")
                {
                    prefillHasLogits = true;
                }
                if (out.name == "present.0.key" && out.shape.size() == 4)
                {
                    d->prefillPresRows = (int) out.shape[2];
                }
            }
            geometryOk = geometryOk && d->prefillMaskLen == d->C + prefillS && d->prefillPresRows >= prefillS && prefillHasLogits;
            if (!geometryOk)
            {
                LOGE("prefill bucket geometry mismatch (mask %d vs C+S %d, present rows %d); using token-by-token prefill", d->prefillMaskLen, d->C + prefillS, d->prefillPresRows);
                prefillB = -1;
            }
        }
        d->prefillBucket = prefillB;
        d->prefillS      = prefillB >= 0 ? prefillS : 0;

        d->inputs.resize(d->inInfo.size());
        for (size_t i = 0; i < d->inInfo.size(); ++i)
        {
            d->inputs[i].name  = d->inInfo[i].name;
            d->inputs[i].shape = d->inInfo[i].shape;
            d->inputs[i].dtype = d->inInfo[i].dtype;
            d->inputs[i].data.assign((size_t) d->inInfo[i].elems * dtypeSize(d->inInfo[i].dtype), 0);
        }
        d->setupKvLinks();
        // Greedy decode reads back only the engine-side argmax of the logits (8 bytes on the GPU
        // backend) instead of the whole vocab row per token; the token stream is unchanged
        // (first-occurrence argmax, exactly the host scan). On refusal the host scan stays.
        if (greedyArgMax)
        {
            if (d->sess->setOutputArgMax((size_t) d->decodeBucket, "logits") == Status::Ok)
            {
                d->argMaxEngaged = true;
            } else
            {
                LOGE("engine argmax unavailable for 'logits'; using the host scan");
            }
        }
        LOGI("loaded: L=%d kv_heads=%d C=%d head_dim=%d vocab=%lld kv_linked=%d prefillS=%d engine_argmax=%d", d->L, d->kvHeads, d->C, d->headDim, (long long) d->vocab, d->kvLinked ? 1 : 0,
             d->prefillS, d->argMaxEngaged ? 1 : 0);
        return reinterpret_cast<jlong>(d);
    } catch (const std::exception &loadError)
    {
        // A vknn::Error mid-load (e.g. the driver refusing a host allocation) must not cross the
        // JNI boundary uncaught — that aborts the process. Relay the reason to Kotlin instead.
        delete d;
        LOGE("load threw: %s", loadError.what());
        throwJavaRuntime(env, loadError.what());
        return 0;
    }
}

// int[7] = {L, kv_heads, C, head_dim, vocab, prefillS, engineArgMax}. prefillS is 0 for a model
// without a usable prefill bucket; engineArgMax is 1 when the decode logits reduce engine-side.
JNIEXPORT jintArray JNICALL Java_com_vknn_chat_NativeLib_nativeInfo(JNIEnv *env, jobject, jlong ptr) {
    auto     *d    = reinterpret_cast<Decoder *>(ptr);
    jint      v[7] = {d->L, d->kvHeads, d->C, d->headDim, (jint) d->vocab, d->prefillS, d->argMaxEngaged ? 1 : 0};
    jintArray a    = env->NewIntArray(7);
    env->SetIntArrayRegion(a, 0, 7, v);
    return a;
}

// Reset the conversation: position 0 and a cleared KV cache. Reseeds the sampler RNG. With the
// linked cache the zeroed past buffers rebind on the next step (with the pending fold cleared),
// reinitializing the engine-resident state.
JNIEXPORT void JNICALL Java_com_vknn_chat_NativeLib_nativeReset(JNIEnv *, jobject, jlong ptr, jint seed) {
    auto *d = reinterpret_cast<Decoder *>(ptr);
    d->p    = 0;
    d->rng.seed((unsigned) seed);
    for (int l = 0; l < d->L; ++l)
    {
        std::fill(d->inputs[(size_t) d->pastKey[l]].data.begin(), d->inputs[(size_t) d->pastKey[l]].data.end(), (uint8_t) 0);
        std::fill(d->inputs[(size_t) d->pastVal[l]].data.begin(), d->inputs[(size_t) d->pastVal[l]].data.end(), (uint8_t) 0);
    }
    d->rebindPastNextStep = d->kvLinked;
    d->residentDirty      = false;
    d->logitsValid        = false;
    d->lastFromDecode     = false;
}

// Feed one token at the current position (runs the plan on the GPU). Returns 0 ok, -1 error.
JNIEXPORT jint JNICALL Java_com_vknn_chat_NativeLib_nativeStep(JNIEnv *, jobject, jlong ptr, jint tok) {
    auto *d = reinterpret_cast<Decoder *>(ptr);
    if (d->step((int64_t) tok) != 0)
    {
        return -1;
    }
    ++d->p;
    return 0;
}

// Feed a whole prompt from the current position: one batched forward per prefill window when the
// model carries a prefill bucket (whole-window TTFT), decode-bucket steps otherwise. The first
// reply token then comes from nativeSample. Returns 0 ok, -1 error, -2 context full.
JNIEXPORT jint JNICALL Java_com_vknn_chat_NativeLib_nativePrefill(JNIEnv *env, jobject, jlong ptr, jlongArray jids) {
    auto       *d = reinterpret_cast<Decoder *>(ptr);
    const jsize n = env->GetArrayLength(jids);
    jlong      *ids = env->GetLongArrayElements(jids, nullptr);
    static_assert(sizeof(jlong) == sizeof(int64_t), "jlong is int64");
    const int rc = d->prefill(reinterpret_cast<const int64_t *>(ids), (int) n);
    env->ReleaseLongArrayElements(jids, ids, JNI_ABORT);
    return rc;
}

// Sample the next token from the last step's logits.
JNIEXPORT jint JNICALL Java_com_vknn_chat_NativeLib_nativeSample(JNIEnv *, jobject, jlong ptr, jfloat temp, jint topK, jfloat topP) {
    auto *d = reinterpret_cast<Decoder *>(ptr);
    return (jint) d->sample(temp, topK, topP);
}

JNIEXPORT void JNICALL Java_com_vknn_chat_NativeLib_nativeFree(JNIEnv *, jobject, jlong ptr) {
    delete reinterpret_cast<Decoder *>(ptr);
}

// ---- VLM bridge -------------------------------------------------------------------------------

// Load a multi-graph vision-language .vxm and build a Vlm. Returns a native handle (0 on failure).
JNIEXPORT jlong JNICALL Java_com_vknn_chat_NativeLib_nativeVlmInit(JNIEnv *env, jobject, jstring jvxm, jstring jcache, jstring jprec, jstring jbackend) {
    auto *m = new Vlm();
    try
    {
        Config cfg;
        cfg.backend                = backendFromStr(jstr(env, jbackend));
        cfg.precision              = precFromStr(jstr(env, jprec));
        cfg.freeWeightsAfterUpload = true;
        m->sess                    = Runtime::load(jstr(env, jvxm), cfg, jstr(env, jcache));
        if (!m->sess)
        {
            LOGE("Runtime::load failed");
            delete m;
            return 0;
        }

        // Discover the bucket roles by input names/shapes (see examples/llm/vlm.cpp). The embedding
        // lookup is fused into the decoders, so a decoder takes input_ids directly; an image turn also
        // binds image_hidden_states + image_positions, landing run() on the image bucket where an on-GPU
        // ScatterND does the splice.
        //   vision "pixel_values"; text-prefill "input_ids"[1,S]; image-prefill = text-prefill's inputs +
        //   image_hidden_states + image_positions; decode "input_ids"[1,1].
        int          visionB = -1, textPreB = -1, imgPreB = -1, decDecB = -1;
        const size_t nb = m->sess->bucketCount();
        for (size_t b = 0; b < nb; ++b)
        {
            std::vector<IOInfo> in    = m->sess->inputInfo(b);
            const int           idsAt = findByName(in, "input_ids");
            if (findByName(in, "pixel_values") >= 0)
            {
                visionB = (int) b;
            } else if (idsAt >= 0)
            {
                const int64_t S = in[(size_t) idsAt].shape.back();
                if (S == 1)
                {
                    decDecB = (int) b;
                } else if (findByName(in, "image_hidden_states") >= 0)
                {
                    imgPreB     = (int) b;
                    m->prefillS = (int) S;
                } else
                {
                    textPreB    = (int) b;
                    m->prefillS = (int) S;
                }
            }
        }
        if (visionB < 0 || textPreB < 0 || decDecB < 0)
        {
            LOGE("not a fused vision-decoder .vxm (%zu buckets)", nb);
            delete m;
            return 0;
        }
        m->visIn  = m->sess->inputInfo((size_t) visionB);
        m->visOut = m->sess->outputInfo((size_t) visionB);
        m->decIn  = m->sess->inputInfo((size_t) textPreB);
        m->decOut = m->sess->outputInfo((size_t) textPreB);

        for (int l = 0;; ++l)
        {
            char kb[64], vb[64];
            snprintf(kb, sizeof kb, "past_key_values.%d.key", l);
            snprintf(vb, sizeof vb, "past_key_values.%d.value", l);
            const int ik = findByName(m->decIn, kb), iv = findByName(m->decIn, vb);
            if (ik < 0 || iv < 0)
            {
                break;
            }
            m->pastKey.push_back(ik);
            m->pastVal.push_back(iv);
        }
        m->L                = (int) m->pastKey.size();
        m->idsIdx           = findByName(m->decIn, "input_ids");
        m->maskIdx          = findByName(m->decIn, "attention_mask");
        m->posIdx           = findByName(m->decIn, "position_ids");
        const int logitsIdx = findByName(m->decOut, "logits");
        if (m->L == 0 || m->idsIdx < 0 || m->maskIdx < 0 || m->posIdx < 0 || logitsIdx < 0)
        {
            LOGE("decoder bucket misses input_ids/attention_mask/position_ids/logits/past");
            delete m;
            return 0;
        }
        const Shape &ks = m->decIn[(size_t) m->pastKey[0]].shape; // [1, KV, C, HD]
        m->kvHeads      = (int) ks[1];
        m->C            = (int) ks[2];
        m->headDim      = (int) ks[3];
        const Shape &vs = m->visOut[0].shape; // [1, imageRows, H]
        m->imgRows      = (int) vs[1];
        m->H            = (int) vs.back(); // decoder splices the vision features in, so H is the vision width
        m->vocab        = m->decOut[(size_t) logitsIdx].shape.back();
        m->imgSide      = (int) m->visIn[0].shape.back();
        m->imgPrefillBucket = imgPreB;
        m->logits.assign((size_t) m->vocab, 0.0f);

        m->dec.resize(m->decIn.size());
        for (size_t i = 0; i < m->decIn.size(); ++i)
        {
            m->dec[i].name  = m->decIn[i].name;
            m->dec[i].shape = m->decIn[i].shape;
            m->dec[i].dtype = m->decIn[i].dtype;
            m->dec[i].data.assign((size_t) m->decIn[i].elems * dtypeSize(m->decIn[i].dtype), 0);
        }
        m->visInT.resize(1);
        m->visInT[0].name  = "pixel_values";
        m->visInT[0].shape = m->visIn[0].shape;
        m->visInT[0].dtype = DType::Float32;
        m->decodeBucket    = decDecB;
        // Present rows from the DECODE bucket's own present output shape (decOut describes the
        // PREFILL bucket, whose present carries prefillS rows — a different count). This decoder's
        // present holds only the produced rows: [1,KV,1,HD] at S=1.
        {
            const std::vector<IOInfo> decodeOut = m->sess->outputInfo((size_t) decDecB);
            const int                 presAt    = findByName(decodeOut, "present.0.key");
            if (presAt >= 0 && decodeOut[(size_t) presAt].shape.size() == 4)
            {
                m->presRowsDecode = (int) decodeOut[(size_t) presAt].shape[2];
            }
        }
        if (m->presRowsDecode > 0)
        {
            m->setupDecodeLinks();
        } else
        {
            LOGE("decode-bucket present outputs are missing or not [1,KV,rows,HD]; using the host cache loop");
        }

        LOGI("vlm loaded: L=%d kv_heads=%d C=%d head_dim=%d H=%d vocab=%lld prefillS=%d imgRows=%d img=%d presRowsDec=%d kv_linked=%d", m->L, m->kvHeads, m->C, m->headDim, m->H,
             (long long) m->vocab, m->prefillS, m->imgRows, m->imgSide, m->presRowsDecode, m->decodeLinked ? 1 : 0);
        return reinterpret_cast<jlong>(m);
    } catch (const std::exception &loadError)
    {
        // A vknn::Error mid-load (e.g. the driver refusing a host allocation) must not cross the
        // JNI boundary uncaught — that aborts the process. Relay the reason to Kotlin instead.
        delete m;
        LOGE("load threw: %s", loadError.what());
        throwJavaRuntime(env, loadError.what());
        return 0;
    }
}

// int[9] = {L, kv_heads, C, head_dim, vocab, prefillS, imageRows, H, imageSide}.
JNIEXPORT jintArray JNICALL Java_com_vknn_chat_NativeLib_nativeVlmInfo(JNIEnv *env, jobject, jlong ptr) {
    auto     *m    = reinterpret_cast<Vlm *>(ptr);
    jint      v[9] = {m->L, m->kvHeads, m->C, m->headDim, (jint) m->vocab, m->prefillS, m->imgRows, m->H, m->imgSide};
    jintArray a    = env->NewIntArray(9);
    env->SetIntArrayRegion(a, 0, 9, v);
    return a;
}

// Reset the conversation: position 0 and a cleared KV cache. Reseeds the sampler RNG. The zeroed
// host cache re-seeds the decode bucket's device-resident state on the next decode step.
JNIEXPORT void JNICALL Java_com_vknn_chat_NativeLib_nativeVlmReset(JNIEnv *, jobject, jlong ptr, jint seed) {
    auto *m = reinterpret_cast<Vlm *>(ptr);
    m->p    = 0;
    m->rng.seed((unsigned) seed);
    for (int l = 0; l < m->L; ++l)
    {
        std::fill(m->dec[(size_t) m->pastKey[l]].data.begin(), m->dec[(size_t) m->pastKey[l]].data.end(), (uint8_t) 0);
        std::fill(m->dec[(size_t) m->pastVal[l]].data.begin(), m->dec[(size_t) m->pastVal[l]].data.end(), (uint8_t) 0);
    }
    m->cacheOnDevice      = false;
    m->rebindPastNextStep = m->decodeLinked;
    m->pendingSlot        = -1;
}

// Run the vision bucket on fp32 CHW pixels [3*IMG*IMG]. Returns the [imageRows*H] embedding rows,
// or null on failure.
JNIEXPORT jfloatArray JNICALL Java_com_vknn_chat_NativeLib_nativeVisionEncode(JNIEnv *env, jobject, jlong ptr, jfloatArray jpix) {
    auto              *m = reinterpret_cast<Vlm *>(ptr);
    const jsize        n = env->GetArrayLength(jpix);
    jfloat            *p = env->GetFloatArrayElements(jpix, nullptr);
    std::vector<float> out;
    const int          rc = m->visionEncode(p, (size_t) n, out);
    env->ReleaseFloatArrayElements(jpix, p, JNI_ABORT);
    if (rc != 0)
    {
        return nullptr;
    }
    jfloatArray a = env->NewFloatArray((jsize) out.size());
    env->SetFloatArrayRegion(a, 0, (jsize) out.size(), out.data());
    return a;
}

// Prefill one prompt turn. `jimg` (nullable) holds imageRows*H embedding rows spliced over the
// prompt's image-token rows; `padId` fills the window past the real ids. The first sampled token
// comes from nativeVlmSample afterwards. Returns 0 ok, <0 error.
JNIEXPORT jint JNICALL Java_com_vknn_chat_NativeLib_nativeVlmPrefill(JNIEnv *env, jobject, jlong ptr, jlongArray jids, jfloatArray jimg, jint imageToken, jint padId) {
    auto       *m            = reinterpret_cast<Vlm *>(ptr);
    const jsize nReal        = env->GetArrayLength(jids);
    jlong      *ids          = env->GetLongArrayElements(jids, nullptr);
    jfloat     *img          = jimg ? env->GetFloatArrayElements(jimg, nullptr) : nullptr;
    const int   imgRowsGiven = jimg ? (int) (env->GetArrayLength(jimg) / (jsize) m->H) : 0;
    static_assert(sizeof(jlong) == sizeof(int64_t), "jlong is int64");
    const int rc = m->prefill(reinterpret_cast<const int64_t *>(ids), (int) nReal, img, imgRowsGiven, (int64_t) imageToken, (int64_t) padId);
    if (img)
    {
        env->ReleaseFloatArrayElements(jimg, img, JNI_ABORT);
    }
    env->ReleaseLongArrayElements(jids, ids, JNI_ABORT);
    return rc;
}

// Feed one token at the current position (embed + decode buckets on the GPU). Returns 0 ok, <0 error.
JNIEXPORT jint JNICALL Java_com_vknn_chat_NativeLib_nativeVlmStep(JNIEnv *, jobject, jlong ptr, jint tok) {
    return reinterpret_cast<Vlm *>(ptr)->step((int64_t) tok);
}

// Sample the next token from the last prefill/step logits.
JNIEXPORT jint JNICALL Java_com_vknn_chat_NativeLib_nativeVlmSample(JNIEnv *, jobject, jlong ptr, jfloat temp, jint topK, jfloat topP) {
    auto *m = reinterpret_cast<Vlm *>(ptr);
    return (jint) sampleLogits(m->logits.data(), m->vocab, temp, topK, topP, m->rng);
}

JNIEXPORT void JNICALL Java_com_vknn_chat_NativeLib_nativeVlmFree(JNIEnv *, jobject, jlong ptr) {
    delete reinterpret_cast<Vlm *>(ptr);
}

// ---- Splat bridge -----------------------------------------------------------------------------
// The YoNoSplat encoder Session (loaded once) plus the shared Vulkan rasterizer (raster_core.h)
// holding the uploaded Gaussians; each render re-submits the pipeline with a new camera pose only.

namespace {

    struct Splat {
        std::unique_ptr<Session>            session;
        std::unique_ptr<raster::Rasterizer> rasterizer;
        std::vector<float>                  cameraPoses; // views*16 row-major c2w (identity fallback)
        std::vector<float>                  intrinsics;  // views*9 normalized K
        int                                 views = 0, height = 0, width = 0;
        float                               pivotDepth = 1.0f; // median view-0 camera-space depth
        int                                 fogPermille = 0;   // last encode: permille of gaussians above opacity 0.1
    };

    // The encoder's predicted camera pose is an internal tensor, not a declared output; these names
    // cover the declared-output case and the graph tensor the export pipeline produces.
    const char *const kPoseTensorNames[] = {"camera_poses", "/enc/Reshape_7_output_0"};

} // namespace

// Load the encoder .vxm (session survives across captures). renderSize is the square rasterizer
// output side, independent of the encoder input side (<= 0 falls back to the encoder side).
// Returns a native handle (0 on failure).
JNIEXPORT jlong JNICALL Java_com_vknn_chat_NativeLib_nativeSplatLoad(JNIEnv *env, jobject, jstring jvxm, jstring jcache, jstring jprec, jstring jbackend, jint renderSize) {
    auto *splat = new Splat();
    try
    {
        Config cfg;
        cfg.backend                = backendFromStr(jstr(env, jbackend));
        cfg.precision              = precFromStr(jstr(env, jprec));
        cfg.freeWeightsAfterUpload = true;
        // Forces the predicted-pose tensor into a dedicated host-readable buffer after each run (its
        // debug file write targets a shell path and fails harmlessly inside an app sandbox).
        cfg.dumpTensors = "camera_poses,Reshape_7_output_0";
        splat->session  = Runtime::load(jstr(env, jvxm), cfg, jstr(env, jcache));
        if (!splat->session)
        {
            LOGE("splat: Runtime::load failed");
            delete splat;
            return 0;
        }
        const std::vector<IOInfo> inputs = splat->session->inputInfo();
        if (inputs.empty() || inputs[0].shape.size() != 5)
        {
            LOGE("splat: encoder image input is not [1,V,3,H,W]");
            delete splat;
            return 0;
        }
        splat->views  = (int) inputs[0].shape[1];
        splat->height = (int) inputs[0].shape[3];
        splat->width  = (int) inputs[0].shape[4];
        // The render resolution is decoupled from the encoder input side: the intrinsics are
        // normalized, so nativeSplatRender scales focal/center by the rasterizer size.
        const int renderSide = renderSize > 0 ? (int) renderSize : splat->width;
        splat->rasterizer    = std::make_unique<raster::Rasterizer>(renderSide, renderSide);
        if (!splat->rasterizer->ok())
        {
            LOGE("splat: no Vulkan for the rasterizer");
            delete splat;
            return 0;
        }
        LOGI("splat loaded: views=%d %dx%d render %dx%d", splat->views, splat->width, splat->height, renderSide, renderSide);
        return reinterpret_cast<jlong>(splat);
    } catch (const std::exception &loadError)
    {
        // A vknn::Error mid-load (e.g. the driver refusing a host allocation) must not cross the
        // JNI boundary uncaught — that aborts the process. Relay the reason to Kotlin instead.
        delete splat;
        LOGE("load threw: %s", loadError.what());
        throwJavaRuntime(env, loadError.what());
        return 0;
    }
}

// int[5] = {gaussians, views, height, width, fogPermille}. gaussians is 0 until an encode
// succeeds; fogPermille is the last encode's fraction (permille) of gaussians above opacity 0.1 —
// a coherent scene concentrates opacity in few splats, a degenerate capture spreads mid-opacity
// fog everywhere, so a high value flags a capture the viewer will show as haze.
JNIEXPORT jintArray JNICALL Java_com_vknn_chat_NativeLib_nativeSplatInfo(JNIEnv *env, jobject, jlong ptr) {
    auto     *splat   = reinterpret_cast<Splat *>(ptr);
    jint      info[5] = {splat->rasterizer ? splat->rasterizer->gaussians() : 0, splat->views, splat->height, splat->width, splat->fogPermille};
    jintArray out     = env->NewIntArray(5);
    env->SetIntArrayRegion(out, 0, 5, info);
    return out;
}

// Run the encoder on fp32 images [1,V,3,H,W] + normalized intrinsics [1,V,3,3]; the Gaussians
// upload to the rasterizer and the predicted camera poses are kept. Returns 0 ok, <0 error.
JNIEXPORT jint JNICALL Java_com_vknn_chat_NativeLib_nativeSplatEncode(JNIEnv *env, jobject, jlong ptr, jfloatArray jimages, jfloatArray jintrinsics) {
    auto     *splat      = reinterpret_cast<Splat *>(ptr);
    const int pixelElems = splat->views * 3 * splat->height * splat->width;
    const int intrElems  = splat->views * 9;
    if (env->GetArrayLength(jimages) != pixelElems || env->GetArrayLength(jintrinsics) != intrElems)
    {
        LOGE("splat: encode input sizes %d/%d, want %d/%d", env->GetArrayLength(jimages), env->GetArrayLength(jintrinsics), pixelElems, intrElems);
        return -1;
    }
    const std::vector<IOInfo> inputInfos = splat->session->inputInfo();
    std::vector<IOTensor>     inputs, outputs;
    for (const IOInfo &info: inputInfos)
    {
        IOTensor tensor;
        tensor.name  = info.name;
        tensor.shape = info.shape;
        tensor.dtype = DType::Float32;
        tensor.data.resize((size_t) numElements(info.shape) * 4);
        const bool  isImage = info.name == inputInfos[0].name; // first input = image (as the example maps)
        jfloatArray source  = isImage ? jimages : jintrinsics;
        env->GetFloatArrayRegion(source, 0, (jsize) numElements(info.shape), reinterpret_cast<jfloat *>(tensor.data.data()));
        inputs.push_back(std::move(tensor));
    }
    if (splat->session->run(inputs, outputs) != Status::Ok)
    {
        LOGE("splat: encoder run failed");
        return -2;
    }
    auto outputByName = [&](const char *name) -> const IOTensor * {
        for (const IOTensor &output: outputs)
        {
            if (output.name == name)
            {
                return &output;
            }
        }
        return nullptr;
    };
    const IOTensor *means = outputByName("means"), *covariances = outputByName("covariances");
    const IOTensor *harmonics = outputByName("harmonics"), *opacities = outputByName("opacities");
    if (!means || !covariances || !harmonics || !opacities)
    {
        LOGE("splat: encoder outputs missing");
        return -3;
    }
    const int          gaussianCount = (int) (numElements(means->shape) / 3);
    std::vector<float> colors((size_t) gaussianCount * 3);
    const float       *harmonicValues = harmonics->f32();
    for (size_t i = 0; i < colors.size(); ++i)
    {
        colors[i] = std::max(0.0f, raster::kC0 * harmonicValues[i] + 0.5f);
    }
    {
        const float *opacityValues = opacities->f32();
        int64_t      hazyCount     = 0;
        for (int64_t i = 0; i < gaussianCount; ++i)
        {
            hazyCount += opacityValues[i] > 0.1f ? 1 : 0;
        }
        splat->fogPermille = gaussianCount > 0 ? (int) (hazyCount * 1000 / gaussianCount) : 0;
    }
    splat->rasterizer->setGaussians(means->f32(), covariances->f32(), colors.data(), opacities->f32(), gaussianCount);

    splat->intrinsics.assign((size_t) intrElems, 0.0f);
    env->GetFloatArrayRegion(jintrinsics, 0, intrElems, splat->intrinsics.data());

    // Predicted camera poses: a declared output, else the internal graph tensor (host-resident via
    // Config::dumpTensors), else identity.
    const size_t poseFloats = (size_t) splat->views * 16;
    splat->cameraPoses.assign(poseFloats, 0.0f);
    for (int view = 0; view < splat->views; ++view)
    {
        splat->cameraPoses[(size_t) view * 16 + 0]  = 1.0f;
        splat->cameraPoses[(size_t) view * 16 + 5]  = 1.0f;
        splat->cameraPoses[(size_t) view * 16 + 10] = 1.0f;
        splat->cameraPoses[(size_t) view * 16 + 15] = 1.0f;
    }
    bool posesFound = false;
    for (const char *name: kPoseTensorNames)
    {
        if (const IOTensor *output = outputByName(name))
        {
            if ((size_t) numElements(output->shape) >= poseFloats)
            {
                std::memcpy(splat->cameraPoses.data(), output->f32(), poseFloats * 4);
                posesFound = true;
            }
            break;
        }
        const RtTensor *tensor = splat->session->tensor(name);
        if (tensor && tensor->hostValid && tensor->host.bytes.size() >= poseFloats * 4)
        {
            std::memcpy(splat->cameraPoses.data(), tensor->host.f32(), poseFloats * 4);
            posesFound = true;
            break;
        }
    }
    LOGI("splat: %d gaussians, poses %s", gaussianCount, posesFound ? "predicted" : "identity");

    // Pivot depth for the orbit viewer: median Gaussian depth in view-0 camera space.
    {
        const float *pose = splat->cameraPoses.data();
        float        rotationT[9];
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                rotationT[row * 3 + col] = pose[col * 4 + row];
            }
        }
        const float        translationZ = -(rotationT[6] * pose[3] + rotationT[7] * pose[7] + rotationT[8] * pose[11]);
        std::vector<float> depths;
        depths.reserve((size_t) gaussianCount);
        const float *meanValues = means->f32();
        for (int i = 0; i < gaussianCount; ++i)
        {
            const float *point = meanValues + (size_t) i * 3;
            const float  depth = rotationT[6] * point[0] + rotationT[7] * point[1] + rotationT[8] * point[2] + translationZ;
            if (depth > 0.0f)
            {
                depths.push_back(depth);
            }
        }
        if (!depths.empty())
        {
            std::nth_element(depths.begin(), depths.begin() + depths.size() / 2, depths.end());
            splat->pivotDepth = depths[depths.size() / 2];
        }
    }
    return 0;
}

// The per-view camera-to-world matrices [views*16], row-major.
JNIEXPORT jfloatArray JNICALL Java_com_vknn_chat_NativeLib_nativeSplatPoses(JNIEnv *env, jobject, jlong ptr) {
    auto       *splat = reinterpret_cast<Splat *>(ptr);
    jfloatArray out   = env->NewFloatArray((jsize) splat->cameraPoses.size());
    env->SetFloatArrayRegion(out, 0, (jsize) splat->cameraPoses.size(), splat->cameraPoses.data());
    return out;
}

// Median Gaussian depth in view-0 camera space; the orbit viewer's look-at distance.
JNIEXPORT jfloat JNICALL Java_com_vknn_chat_NativeLib_nativeSplatPivotDepth(JNIEnv *, jobject, jlong ptr) {
    return reinterpret_cast<Splat *>(ptr)->pivotDepth;
}

// Render from a row-major camera-to-world [16] using view-0 intrinsics. Returns packed ARGB
// [renderHeight*renderWidth] for a Bitmap (packed on the GPU by the composite pass), or null on
// failure. The normalized intrinsics scale by the rasterizer size, so the render resolution
// tracks nativeSplatLoad's renderSize rather than the encoder input side.
JNIEXPORT jintArray JNICALL Java_com_vknn_chat_NativeLib_nativeSplatRender(JNIEnv *env, jobject, jlong ptr, jfloatArray jcameraToWorld) {
    auto *splat = reinterpret_cast<Splat *>(ptr);
    if (!splat->rasterizer || splat->rasterizer->gaussians() == 0 || env->GetArrayLength(jcameraToWorld) != 16 || splat->intrinsics.size() < 9)
    {
        return nullptr;
    }
    float cameraToWorld[16];
    env->GetFloatArrayRegion(jcameraToWorld, 0, 16, cameraToWorld);
    const int    renderWidth = splat->rasterizer->width(), renderHeight = splat->rasterizer->height();
    const float *view0K = splat->intrinsics.data();
    const float  focalX = view0K[0] * renderWidth, focalY = view0K[4] * renderHeight;
    const float  centerX = view0K[2] * renderWidth, centerY = view0K[5] * renderHeight;
    static_assert(sizeof(jint) == sizeof(uint32_t), "jint is 32-bit");
    std::vector<uint32_t> packed((size_t) renderHeight * renderWidth);
    raster::Stats         stats;
    if (splat->rasterizer->renderPacked(cameraToWorld, focalX, focalY, centerX, centerY, packed.data(), &stats) != raster::Result::Ok)
    {
        LOGE("splat: render failed (%u entries)", stats.entries);
        return nullptr;
    }
    LOGI("splat: rendered %u entries in %.1f ms", stats.entries, stats.msCount + stats.msMain);
    jintArray out = env->NewIntArray((jsize) packed.size());
    env->SetIntArrayRegion(out, 0, (jsize) packed.size(), reinterpret_cast<const jint *>(packed.data()));
    return out;
}

JNIEXPORT void JNICALL Java_com_vknn_chat_NativeLib_nativeSplatFree(JNIEnv *, jobject, jlong ptr) {
    delete reinterpret_cast<Splat *>(ptr);
}

} // extern "C"
