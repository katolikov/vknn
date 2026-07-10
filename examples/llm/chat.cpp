// Device-side GPU decode loop for an autoregressive decoder (Qwen2 family).
//
// Reads whitespace-separated prompt token ids on stdin (one conversation turn per line), runs
// prefill + greedy/sampled decode on the VKNN session, and streams generated token ids to stdout
// (one integer per line, flushed), then a "END" sentinel line at end of turn. A host tokenizer
// front-end supplies the ids and detokenizes the stream live. The KV cache and the absolute token
// position persist across turns. Every tensor-compute op runs on the configured backend (Vulkan by
// default); only the token loop and argmax/sampling are host code here — tokenization is the front
// end's job.
//
//   vknn_chat model.vxm [--backend vulkan|cpu] [--precision low|normal|high] [--fp32-tensors CSV]
//             [--max-tokens N] [--temp T] [--top-k K] [--top-p P] [--eos ID] [--seed S]
//             [--no-kv-link] [--no-prefill] [--no-gpu-argmax] [--timing]
//
// Greedy decode (--temp 0, the default) registers the decode bucket's logits for the engine-side
// argmax (Session::setOutputArgMax): per token the engine reduces the logits on the GPU and the
// host reads back 8 bytes instead of the vocab row, with an identical token stream. A negative
// --eos disables early stop (no generated id ever matches it).
//
// A multi-bucket model whose second bucket takes input_ids [1,S] (S>1) prefills the prompt in
// S-token forwards instead of token-by-token — TTFT then costs one batched pass per S prompt
// tokens. The prefill pass runs on the host cache flow (its present rows fold into the host past
// buffers) and the first decode step re-seeds the engine-resident cache from them; --no-prefill
// forces the token-by-token path for A/B. Single-bucket models are unaffected.
//
// The model is a with-past decoder compiled at a fixed past length C (read from the .vxm). Each step
// feeds one token at absolute position p, the [1, kv_heads, C, head_dim] cache as the past key/value
// inputs, and an attention mask marking the p valid past slots plus the current token, so a single
// fixed-shape plan serves every step. The new token's key/value (present index C) lands in cache
// slot p. This matches the reference HF greedy stream token-for-token.
//
// The KV cache is ENGINE-RESIDENT by default: every present.N.{key,value} output is linked to its
// past_key_values.N.{key,value} input (Session::linkOutputToInput), so per token the engine folds
// the new row into the cache in place — on the GPU backend entirely on-device, with no host copy of
// the ~2*L*kv_heads*C*head_dim cache in either direction. --no-kv-link selects the host-side loop
// (bind the cache every step, fold present rows on the host) for A/B comparison; both paths produce
// the same token stream.
#include "vknn/runtime.h"
#include "vknn/session.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace vknn;

// Value after option `k` (scanned from arg 2, past the model path), or default `d`.
static const char *opt(int c, char **v, const char *k, const char *d) noexcept {
    for (int i = 2; i < c - 1; ++i)
    {
        if (!strcmp(v[i], k))
        {
            return v[i + 1];
        }
    }
    return d;
}

// True when the value-less flag `k` is present (scanned from arg 2, past the model path).
static bool flagSet(int c, char **v, const char *k) noexcept {
    for (int i = 2; i < c; ++i)
    {
        if (!strcmp(v[i], k))
        {
            return true;
        }
    }
    return false;
}

int main(int argc, char **argv) {
    if (argc < 2)
    {
        fprintf(stderr,
                "usage: %s model.vxm [--backend vulkan|cpu] [--precision low|normal|high]\n"
                "        [--fp32-tensors CSV] [--max-tokens N] [--temp T] [--top-k K] [--top-p P]\n"
                "        [--eos ID] [--seed S] [--no-kv-link] [--no-prefill] [--no-gpu-argmax] [--timing]\n",
                argv[0]);
        return 1;
    }
    std::string model = argv[1];

    Config cfg;
    cfg.backend                = backendFromStr(opt(argc, argv, "--backend", "vulkan"));
    cfg.precision              = precisionFromStr(opt(argc, argv, "--precision", "low"));
    cfg.fp32Tensors            = opt(argc, argv, "--fp32-tensors", "");
    cfg.freeWeightsAfterUpload = true;
    const int     maxTokens    = atoi(opt(argc, argv, "--max-tokens", "128"));
    const float   temp         = (float) atof(opt(argc, argv, "--temp", "0"));
    const int     topK         = atoi(opt(argc, argv, "--top-k", "0"));
    const float   topP         = (float) atof(opt(argc, argv, "--top-p", "1"));
    const int64_t eos          = atoll(opt(argc, argv, "--eos", "151643"));
    bool          kvLink       = !flagSet(argc, argv, "--no-kv-link"); // may drop to the host loop mid-stream on a link failure
    cfg.timing                 = flagSet(argc, argv, "--timing");      // per-run pack/submit/unpack walls
    if (flagSet(argc, argv, "--no-matmul-view-fold"))
    {
        cfg.setHint(Hint::MatMulViewFold, (int) Mode::Off);
    }
    if (flagSet(argc, argv, "--no-rope-fusion"))
    {
        cfg.setHint(Hint::RopeFusion, (int) Mode::Off);
    }
    if (flagSet(argc, argv, "--no-fused-attention"))
    {
        cfg.setHint(Hint::FusedAttention, (int) Mode::Off);
    }
    if (flagSet(argc, argv, "--no-kv-concat-fold"))
    {
        cfg.setHint(Hint::KvConcatFold, (int) Mode::Off);
    }
    std::mt19937 rng((unsigned) atoi(opt(argc, argv, "--seed", "1234")));

    auto sess = Runtime::load(model, cfg);
    if (!sess)
    {
        fprintf(stderr, "failed to load %s\n", model.c_str());
        return 1;
    }
    // Bucket roles for a multi-bucket .vxm: the decode bucket feeds input_ids [1,1]; a prefill
    // bucket (optional) feeds [1,S] with S>1 and processes a whole prompt window in ONE forward —
    // the difference between TTFT scaling as prompt-length decode steps and one batched pass. A
    // single-bucket model keeps the token-by-token prefill below, byte-identically to before.
    int decodeBucket = -1, prefillBucket = -1, prefillS = 0;
    for (size_t b = 0; b < sess->bucketCount(); ++b)
    {
        const std::vector<IOInfo> bins = sess->inputInfo(b);
        for (const IOInfo &in: bins)
        {
            if (in.name == "input_ids" && in.shape.size() == 2)
            {
                if (in.shape[1] == 1 && decodeBucket < 0)
                {
                    decodeBucket = (int) b;
                } else if (in.shape[1] > 1 && (int) in.shape[1] > prefillS)
                {
                    prefillBucket = (int) b;
                    prefillS      = (int) in.shape[1];
                }
            }
        }
    }
    if (decodeBucket < 0)
    {
        fprintf(stderr, "model has no input_ids [1,1] decode bucket\n");
        return 2;
    }
    if (flagSet(argc, argv, "--no-prefill"))
    {
        prefillBucket = -1; // token-by-token A/B reference
    }
    const std::vector<IOInfo> ins  = sess->inputInfo((size_t) decodeBucket);
    const std::vector<IOInfo> outs = sess->outputInfo((size_t) decodeBucket);

    auto findIn = [&](const std::string &n) -> int {
        for (size_t i = 0; i < ins.size(); ++i)
        {
            if (ins[i].name == n)
            {
                return (int) i;
            }
        }
        return -1;
    };
    auto findOut = [&](const std::string &n) -> int {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            if (outs[i].name == n)
            {
                return (int) i;
            }
        }
        return -1;
    };

    const int idIdx     = findIn("input_ids");
    const int maskIdx   = findIn("attention_mask");
    const int posIdx    = findIn("position_ids");
    const int logitsIdx = findOut("logits");
    // Enumerate layers by their past/present key/value tensors.
    std::vector<int> pastKey, pastVal, presKey, presVal;
    for (int l = 0;; ++l)
    {
        char kb[64], vb[64], pk[64], pv[64];
        snprintf(kb, sizeof kb, "past_key_values.%d.key", l);
        snprintf(vb, sizeof vb, "past_key_values.%d.value", l);
        const int ik = findIn(kb), iv = findIn(vb);
        if (ik < 0 || iv < 0)
        {
            break;
        }
        snprintf(pk, sizeof pk, "present.%d.key", l);
        snprintf(pv, sizeof pv, "present.%d.value", l);
        pastKey.push_back(ik);
        pastVal.push_back(iv);
        presKey.push_back(findOut(pk));
        presVal.push_back(findOut(pv));
    }
    const int L = (int) pastKey.size();
    if (idIdx < 0 || maskIdx < 0 || posIdx < 0 || logitsIdx < 0 || L == 0)
    {
        fprintf(stderr, "model is not a qwen2 with-past decoder (missing input_ids/attention_mask/"
                        "position_ids/logits/past_key_values.*)\n");
        return 2;
    }
    // Geometry from past_key_values.0.key = [1, kv_heads, C, head_dim]; logits = [1, seq, vocab].
    const Shape  &ks      = ins[pastKey[0]].shape;
    const int     kvHeads = (int) ks[1];
    const int     C       = (int) ks[2];
    const int     headDim = (int) ks[3];
    const int64_t vocab   = outs[logitsIdx].shape.back();
    // Present rows from the PRESENT output's own shape, never assumed: a cache-concat decoder
    // carries C+1 rows (the new token at index C), a rows-only decoder carries exactly the step's
    // rows (one row at S=1). The fold source is always the LAST present row.
    const int presRows = (presKey[0] >= 0 && outs[(size_t) presKey[0]].shape.size() == 4) ? (int) outs[(size_t) presKey[0]].shape[2] : 0;
    if (presRows <= 0)
    {
        fprintf(stderr, "present outputs are missing or not [1,KV,rows,HD]; cannot drive the KV fold\n");
        return 2;
    }
    fprintf(stderr, "[chat] %s: layers=%d kv_heads=%d C=%d head_dim=%d present_rows=%d vocab=%lld\n", model.c_str(), L, kvHeads, C, headDim, presRows, (long long) vocab);
    // chat only COMPARES --eos against generated ids (it is never fed to the model), so a negative
    // id is a valid "never matches" sentinel that disables early stop (timing harnesses use it). A
    // non-negative id past the vocab can never be generated either, and only signals a caller
    // mistake — fail with the value and the vocab size.
    if (eos >= vocab)
    {
        fprintf(stderr, "--eos %lld is out of range for this model (vocab %lld)\n", (long long) eos, (long long) vocab);
        return 2;
    }
    // Greedy decode reads back only the engine-side argmax of the logits (8 bytes on the GPU
    // backend) instead of downloading and scanning the whole vocab row per token; the stream is
    // unchanged (first-occurrence argmax, exactly this tool's host scan). Sampling (--temp > 0)
    // needs the full distribution, so it keeps the row readback, as does --no-gpu-argmax (the A/B
    // reference).
    bool gpuArgmax = temp <= 0.0f && !flagSet(argc, argv, "--no-gpu-argmax");
    if (gpuArgmax && sess->setOutputArgMax((size_t) decodeBucket, "logits") != Status::Ok)
    {
        fprintf(stderr, "[chat] engine argmax unavailable for 'logits'; using the host scan\n");
        gpuArgmax = false;
    }

    // Persistent boundary tensors, in model input order. Under --no-kv-link the past key/value
    // buffers ARE the KV cache (fp32 host boundary), retained across steps and turns; with linking
    // (the default) the cache stays engine-resident, those buffers are never allocated or bound, and
    // only the id/mask/position tensors travel per step.
    auto isPastInput = [&](size_t i) {
        for (int l = 0; l < L; ++l)
        {
            if ((int) i == pastKey[l] || (int) i == pastVal[l])
            {
                return true;
            }
        }
        return false;
    };
    // With a prefill bucket the host past buffers are needed even in linked mode: the prefill pass
    // runs on the host cache flow (bind past, fold present rows back), and the first decode step
    // re-seeds the engine-resident cache from them.
    std::vector<IOTensor> inputs(ins.size());
    for (size_t i = 0; i < ins.size(); ++i)
    {
        inputs[i].name  = ins[i].name;
        inputs[i].shape = ins[i].shape;
        inputs[i].dtype = ins[i].dtype;
        if (!(kvLink && isPastInput(i) && prefillBucket < 0))
        {
            inputs[i].data.assign((size_t) ins[i].elems * dtypeSize(ins[i].dtype), 0);
        }
    }
    auto setI64 = [&](int idx, const std::vector<int64_t> &vals) {
        std::memcpy(inputs[idx].data.data(), vals.data(), vals.size() * sizeof(int64_t));
    };

    // Prefill-bucket geometry, validated against the decode bucket: the past inputs must share the
    // decode shapes (one host cache serves both) and the mask must span past+S columns. Any
    // mismatch disables the fast prefill rather than miscomputing.
    int prefillMaskLen = 0, prefillPresRows = 0, prefillLogitsIdx = -1;
    if (prefillBucket >= 0)
    {
        const std::vector<IOInfo> pin  = sess->inputInfo((size_t) prefillBucket);
        const std::vector<IOInfo> pout = sess->outputInfo((size_t) prefillBucket);
        bool                      ok   = true;
        for (const IOInfo &in: pin)
        {
            if (in.name == ins[(size_t) pastKey[0]].name)
            {
                ok = ok && in.shape == ins[(size_t) pastKey[0]].shape;
            }
            if (in.name == "attention_mask" && in.shape.size() == 2)
            {
                prefillMaskLen = (int) in.shape[1];
            }
        }
        for (size_t i = 0; i < pout.size(); ++i)
        {
            if (pout[i].name == "logits")
            {
                prefillLogitsIdx = (int) i;
            }
            if (pout[i].name == "present.0.key" && pout[i].shape.size() == 4)
            {
                prefillPresRows = (int) pout[i].shape[2];
            }
        }
        ok = ok && prefillMaskLen == C + prefillS && prefillPresRows >= prefillS && prefillLogitsIdx >= 0;
        if (!ok)
        {
            fprintf(stderr, "[chat] prefill bucket geometry mismatch (mask %d vs C+S %d, present rows %d); using token-by-token prefill\n", prefillMaskLen, C + prefillS, prefillPresRows);
            prefillBucket = -1;
        } else
        {
            fprintf(stderr, "[chat] prefill bucket: S=%d, one forward per %d prompt tokens\n", prefillS, prefillS);
        }
    }

    // Declare the KV links up front: every present output feeds its past input on the next run. The
    // ranges start empty (the cache starts as zeros, matching the host loop's zero-filled buffers);
    // each step from p=1 on re-links with the ranges that fold the previous token's row into its slot.
    if (kvLink)
    {
        // Bucket-explicit links: a multi-bucket model carries the present/past pair in every
        // bucket, and only the decode bucket's cache is engine-resident (the prefill pass uses the
        // host cache flow). The overload is also correct for a single-bucket model (bucket 0).
        for (int l = 0; l < L; ++l)
        {
            if (sess->linkOutputToInput((size_t) decodeBucket, outs[(size_t) presKey[l]].name, ins[(size_t) pastKey[l]].name, {}) != Status::Ok ||
                sess->linkOutputToInput((size_t) decodeBucket, outs[(size_t) presVal[l]].name, ins[(size_t) pastVal[l]].name, {}) != Status::Ok)
            {
                fprintf(stderr, "[chat] KV link failed (see log); rerun with --no-kv-link\n");
                return 2;
            }
        }
        fprintf(stderr, "[chat] KV cache engine-resident (%d links)\n", 2 * L);
    }

    std::vector<IOTensor> outputs;
    // Map an output name to its index in the run() result vector (stable across runs).
    std::vector<int> outIdxByInfo(outs.size(), -1);
    bool             mapped = false;

    int  p             = 0;     // absolute position across the whole conversation
    bool residentDirty = false; // linked decode ran: the engine cache is ahead of the host buffers
    bool reseedCache   = false; // next linked decode step re-seeds the resident cache from the host

    // Copy the engine-resident cache (plus the pending fold of the last present row into
    // `pendingSlot`, when >= 0) back into the host past buffers. Used at a prefill turn boundary
    // and by the link-failure fallback — the host cache then holds the full conversation state.
    auto syncResidentToHost = [&](int64_t pendingSlot) -> bool {
        for (int l = 0; l < L; ++l)
        {
            for (int part = 0; part < 2; ++part)
            {
                IOTensor &hostPast = inputs[(size_t) (part ? pastVal[l] : pastKey[l])];
                if (hostPast.data.empty())
                {
                    hostPast.data.assign((size_t) ins[(size_t) (part ? pastVal[l] : pastKey[l])].elems * sizeof(float), 0);
                }
                IOTensor resident;
                if (sess->readResident(hostPast.name, resident) != Status::Ok || resident.data.size() != hostPast.data.size())
                {
                    fprintf(stderr, "[chat] resident cache readback failed for %s\n", hostPast.name.c_str());
                    return false;
                }
                std::memcpy(hostPast.data.data(), resident.data.data(), resident.data.size());
                if (pendingSlot >= 0)
                {
                    IOTensor present;
                    if (sess->readResident(outs[(size_t) (part ? presVal[l] : presKey[l])].name, present) != Status::Ok)
                    {
                        fprintf(stderr, "[chat] resident present readback failed\n");
                        return false;
                    }
                    const float *src = reinterpret_cast<const float *>(present.data.data());
                    float       *dst = reinterpret_cast<float *>(hostPast.data.data());
                    for (int h = 0; h < kvHeads; ++h)
                    {
                        std::memcpy(dst + ((size_t) h * C + pendingSlot) * headDim, src + ((size_t) h * presRows + (presRows - 1)) * headDim, (size_t) headDim * sizeof(float));
                    }
                }
            }
        }
        return true;
    };

    // Decode-loop phase walls (--timing): accumulated per decode step, reported per turn.
    double tmLinkMs = 0, tmRunMs = 0, tmPrepMs = 0, tmSampleMs = 0;
    int    tmSteps  = 0;
    auto   nowMs    = [] {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };

    // The last forward's logits row and its provenance. A decode step under engine argmax returns
    // the logits entry with no data (lastLogits stays null; readOutputArgMax serves the token); a
    // prefill pass always yields a full host row (its bucket is never argmax-registered).
    const float *lastLogits     = nullptr;
    bool         lastFromDecode = false;

    // Feed one token at the current position; false on error.
    auto step = [&](int64_t tok) -> bool {
        const double tPrep0 = nowMs();
        setI64(idIdx, {tok});
        setI64(posIdx, {(int64_t) p});
        std::vector<int64_t> am((size_t) C + 1, 0);
        for (int j = 0; j < p && j < C; ++j)
        {
            am[(size_t) j] = 1; // valid past slots
        }
        am[(size_t) C] = 1; // the current token (appended at index C)
        setI64(maskIdx, am);
        const double tLink0 = nowMs();
        tmPrepMs += tLink0 - tPrep0;
        double tRun0 = tLink0;

        Status runStatus = Status::Ok;
        bool   ranLinked = false;
        if (kvLink)
        {
            // The engine folds the PREVIOUS token's present row (the last row of each head) into
            // cache slot p-1 at the start of this run (min guards the overrun past the compiled
            // context, like the host loop's slot clamp). A re-seed step (the first decode after a
            // prefill pass) instead binds the FULL host past buffers — binding a linked input
            // reinitializes its resident state — with no pending fold.
            bool linksOk = true;
            {
                const int64_t                slot   = (!reseedCache && p > 0) ? std::min<int64_t>(p - 1, C - 1) : -1;
                const std::vector<LinkRange> ranges = kvFoldRanges(kvHeads, presRows, C, headDim, slot);
                for (int l = 0; l < L && linksOk; ++l)
                {
                    for (int part = 0; part < 2 && linksOk; ++part)
                    {
                        const std::string &pres = outs[(size_t) (part ? presVal[l] : presKey[l])].name;
                        const std::string &past = ins[(size_t) (part ? pastVal[l] : pastKey[l])].name;
                        const Status       st   = sess->linkOutputToInput((size_t) decodeBucket, pres, past, ranges);
                        if (st != Status::Ok)
                        {
                            fprintf(stderr, "[chat] KV link update failed for %s -> %s at slot %lld: %s (see log)\n", pres.c_str(), past.c_str(), (long long) slot, statusStr(st));
                            linksOk = false;
                        }
                    }
                }
            }
            tRun0 = nowMs();
            tmLinkMs += tRun0 - tLink0;
            if (linksOk)
            {
                if (reseedCache)
                {
                    runStatus   = sess->run(inputs, outputs); // full bind re-seeds the resident cache
                    reseedCache = false;
                } else
                {
                    std::vector<IOTensor> bound {inputs[(size_t) idIdx], inputs[(size_t) maskIdx], inputs[(size_t) posIdx]};
                    runStatus = sess->run(bound, outputs);
                }
                ranLinked     = true;
                residentDirty = true;
            } else
            {
                // A mid-stream link failure: bring the engine-resident cache (device rows + the
                // pending fold from the last run's present) back into the host past buffers, drop
                // the links, and continue THIS and every later step on the host cache loop — same
                // tokens, no lost state.
                fprintf(stderr, "[chat] switching to the host KV loop at p=%d (resyncing the cache from the engine)\n", p);
                if (!syncResidentToHost(p > 0 ? std::min<int64_t>(p - 1, C - 1) : -1))
                {
                    return false;
                }
                sess->clearLinks();
                kvLink = false;
            }
        }
        if (!ranLinked)
        {
            runStatus = sess->run(inputs, outputs);
        }
        tmRunMs += nowMs() - tRun0;
        ++tmSteps;
        if (runStatus != Status::Ok)
        {
            fprintf(stderr, "[chat] run failed\n");
            return false;
        }
        if (!mapped)
        {
            for (size_t j = 0; j < outputs.size(); ++j)
            {
                for (size_t k = 0; k < outs.size(); ++k)
                {
                    if (outputs[j].name == outs[k].name)
                    {
                        outIdxByInfo[k] = (int) j;
                    }
                }
            }
            mapped = true;
        }
        if (!kvLink)
        {
            // Host fold: append the new token's key/value (the last present row) into cache slot p
            // (min(p, C-1) guards the rare overrun past the compiled context).
            const int slot = p < C ? p : C - 1;
            for (int l = 0; l < L; ++l)
            {
                for (int part = 0; part < 2; ++part)
                {
                    const IOTensor &pres = outputs[(size_t) outIdxByInfo[part ? presVal[l] : presKey[l]]];
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
        const IOTensor &logitsOut = outputs[(size_t) outIdxByInfo[logitsIdx]];
        lastLogits                = logitsOut.data.empty() ? nullptr : reinterpret_cast<const float *>(logitsOut.data.data());
        lastFromDecode            = true;
        return true;
    };

    // One prefill pass: feed `len` prompt tokens (<= prefillS) starting at absolute position p in a
    // single forward through the prefill bucket, on the HOST cache flow — bind the past buffers,
    // fold the produced present rows (the S new rows after the past block) back into cache slots
    // p..p+len-1 — and return the last real token's logits row. Pad slots carry mask 0, so real
    // tokens never attend them and their (garbage) rows are simply not folded. Advances p.
    std::vector<IOTensor> prefillOutputs;
    auto                  prefillPass = [&](const int64_t *toks, int len) -> const float                  *{
        std::vector<IOTensor> pin(inputs.size());
        for (size_t i = 0; i < ins.size(); ++i)
        {
            pin[i].name  = ins[i].name;
            pin[i].dtype = ins[i].dtype;
            if (isPastInput(i))
            {
                pin[i].shape = ins[i].shape;
                pin[i].data  = inputs[i].data; // the host KV cache, shared shape with the decode bucket
                continue;
            }
            if ((int) i == idIdx || (int) i == posIdx)
            {
                pin[i].shape = {1, (int64_t) prefillS};
                std::vector<int64_t> v((size_t) prefillS, 0);
                for (int t = 0; t < prefillS; ++t)
                {
                    v[(size_t) t] = ((int) i == idIdx) ? (t < len ? toks[t] : 0) : (int64_t) (p + t);
                }
                pin[i].data.resize((size_t) prefillS * 8);
                std::memcpy(pin[i].data.data(), v.data(), v.size() * 8);
            } else if ((int) i == maskIdx)
            {
                pin[i].shape = {1, (int64_t) prefillMaskLen};
                std::vector<int64_t> m((size_t) prefillMaskLen, 0);
                for (int j = 0; j < p && j < C; ++j)
                {
                    m[(size_t) j] = 1; // valid past slots
                }
                for (int t = 0; t < len; ++t)
                {
                    m[(size_t) (C + t)] = 1; // the chunk's real tokens; pads stay masked
                }
                pin[i].data.resize((size_t) prefillMaskLen * 8);
                std::memcpy(pin[i].data.data(), m.data(), m.size() * 8);
            }
        }
        if (sess->run(pin, prefillOutputs) != Status::Ok)
        {
            fprintf(stderr, "[chat] prefill run failed\n");
            return (const float *) nullptr;
        }
        const IOTensor *logitsOut = nullptr;
        for (const IOTensor &o: prefillOutputs)
        {
            if (o.name == "logits")
            {
                logitsOut = &o;
            }
        }
        if (!logitsOut)
        {
            return (const float *) nullptr;
        }
        // Fold the chunk's present rows (after the past block) into the host cache.
        const int newRowsAt = prefillPresRows - prefillS;
        for (int l = 0; l < L; ++l)
        {
            for (int part = 0; part < 2; ++part)
            {
                const std::string &presName = outs[(size_t) (part ? presVal[l] : presKey[l])].name;
                const IOTensor    *pres     = nullptr;
                for (const IOTensor &o: prefillOutputs)
                {
                    if (o.name == presName)
                    {
                        pres = &o;
                    }
                }
                if (!pres)
                {
                    return (const float *) nullptr;
                }
                const float *src = reinterpret_cast<const float *>(pres->data.data());
                float       *dst = reinterpret_cast<float *>(inputs[(size_t) (part ? pastVal[l] : pastKey[l])].data.data());
                for (int h = 0; h < kvHeads; ++h)
                {
                    for (int t = 0; t < len; ++t)
                    {
                        std::memcpy(dst + ((size_t) h * C + p + t) * headDim, src + ((size_t) h * prefillPresRows + newRowsAt + t) * headDim, (size_t) headDim * sizeof(float));
                    }
                }
            }
        }
        p += len;
        return reinterpret_cast<const float *>(logitsOut->data.data()) + (size_t) (len - 1) * vocab;
    };

    // Pick the next token id from a logits row: greedy at temp<=0, else temperature + top-k + top-p.
    auto sample = [&](const float *logits) -> int64_t {
        if (temp <= 0.0f)
        {
            int64_t best = 0;
            float   bv   = logits[0];
            for (int64_t i = 1; i < vocab; ++i)
            {
                if (logits[i] > bv)
                {
                    bv   = logits[i];
                    best = i;
                }
            }
            return best;
        }
        std::vector<std::pair<float, int64_t>> v((size_t) vocab);
        for (int64_t i = 0; i < vocab; ++i)
        {
            v[(size_t) i] = {logits[i] / temp, i};
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
    };

    // One turn per stdin line of space-separated ids. Prefill feeds each prompt token (the last one's
    // logits give the first generated token); decode streams until eos or the token budget.
    std::string line;
    while (std::getline(std::cin, line))
    {
        std::istringstream   ss(line);
        std::vector<int64_t> prompt;
        int64_t              t;
        while (ss >> t)
        {
            prompt.push_back(t);
        }
        if (prompt.empty())
        {
            continue;
        }
        // An out-of-vocab prompt id would index past the embedding table; fail THIS TURN with the
        // value and the vocab size — never feed it to the model, never kill the persistent process.
        bool promptValid = true;
        for (size_t i = 0; i < prompt.size() && promptValid; ++i)
        {
            if (prompt[i] < 0 || prompt[i] >= vocab)
            {
                fprintf(stderr, "[chat] prompt token id %lld (position %zu) is out of range for this model (vocab %lld); turn skipped\n", (long long) prompt[i], i, (long long) vocab);
                promptValid = false;
            }
        }
        if (!promptValid)
        {
            printf("END\n");
            fflush(stdout);
            continue;
        }

        size_t consumed = 0;
        if (prefillBucket >= 0)
        {
            // Whole-window prefill: sync the engine-resident cache to the host once per turn (a
            // linked decode from the previous turn leaves a pending fold at slot p-1), then feed
            // the prompt in prefillS-sized forwards. A remainder past the compiled context falls
            // to the token-by-token loop below.
            if (kvLink && residentDirty)
            {
                if (!syncResidentToHost(p > 0 ? std::min<int64_t>(p - 1, C - 1) : -1))
                {
                    return 3;
                }
                residentDirty = false;
            }
            while (consumed < prompt.size())
            {
                const int len = (int) std::min<size_t>(prompt.size() - consumed, (size_t) prefillS);
                if (p + len > C)
                {
                    break;
                }
                const float *prefillLogits = prefillPass(&prompt[consumed], len);
                if (!prefillLogits)
                {
                    return 3;
                }
                lastLogits     = prefillLogits;
                lastFromDecode = false;
                consumed += (size_t) len;
            }
            reseedCache = kvLink && consumed > 0; // first decode step re-seeds the resident cache
        }
        for (; consumed < prompt.size(); ++consumed)
        {
            if (!step(prompt[consumed]))
            {
                return 3;
            }
            ++p;
        }
        for (int n = 0; n < maxTokens; ++n)
        {
            const double tSample0 = nowMs();
            int64_t      next;
            if (gpuArgmax && lastFromDecode)
            {
                float bestValue;
                if (sess->readOutputArgMax("logits", next, bestValue) != Status::Ok)
                {
                    fprintf(stderr, "[chat] engine argmax readback failed\n");
                    return 3;
                }
            } else
            {
                if (!lastLogits)
                {
                    fprintf(stderr, "[chat] logits row unavailable for sampling\n");
                    return 3;
                }
                next = sample(lastLogits);
            }
            tmSampleMs += nowMs() - tSample0;
            if (next == eos)
            {
                break;
            }
            printf("%lld\n", (long long) next);
            fflush(stdout);
            if (!step(next))
            {
                return 3;
            }
            ++p;
        }
        if (cfg.timing && tmSteps > 0)
        {
            fprintf(stderr, "[chat] step phases avg over %d step(s): prep=%.3fms links=%.3fms run=%.3fms sample=%.3fms\n",
                    tmSteps, tmPrepMs / tmSteps, tmLinkMs / tmSteps, tmRunMs / tmSteps, tmSampleMs / tmSteps);
            tmPrepMs = tmLinkMs = tmRunMs = tmSampleMs = 0;
            tmSteps  = 0;
        }
        printf("END\n");
        fflush(stdout);
    }
    return 0;
}
