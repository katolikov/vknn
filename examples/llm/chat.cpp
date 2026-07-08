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
//             [--no-kv-link] [--timing]
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
                "        [--eos ID] [--seed S]\n",
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
    const bool    kvLink       = !flagSet(argc, argv, "--no-kv-link");
    cfg.timing                 = flagSet(argc, argv, "--timing"); // per-run pack/submit/unpack walls
    std::mt19937  rng((unsigned) atoi(opt(argc, argv, "--seed", "1234")));

    auto sess = Runtime::load(model, cfg);
    if (!sess)
    {
        fprintf(stderr, "failed to load %s\n", model.c_str());
        return 1;
    }
    const std::vector<IOInfo> ins  = sess->inputInfo();
    const std::vector<IOInfo> outs = sess->outputInfo();

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
    fprintf(stderr, "[chat] %s: layers=%d kv_heads=%d C=%d head_dim=%d vocab=%lld\n", model.c_str(), L, kvHeads, C, headDim, (long long) vocab);

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
    std::vector<IOTensor> inputs(ins.size());
    for (size_t i = 0; i < ins.size(); ++i)
    {
        inputs[i].name  = ins[i].name;
        inputs[i].shape = ins[i].shape;
        inputs[i].dtype = ins[i].dtype;
        if (!(kvLink && isPastInput(i)))
        {
            inputs[i].data.assign((size_t) ins[i].elems * dtypeSize(ins[i].dtype), 0);
        }
    }
    auto setI64 = [&](int idx, const std::vector<int64_t> &vals) {
        std::memcpy(inputs[idx].data.data(), vals.data(), vals.size() * sizeof(int64_t));
    };

    // Declare the KV links up front: every present output feeds its past input on the next run. The
    // ranges start empty (the cache starts as zeros, matching the host loop's zero-filled buffers);
    // each step from p=1 on re-links with the ranges that fold the previous token's row into its slot.
    if (kvLink)
    {
        for (int l = 0; l < L; ++l)
        {
            if (sess->linkOutputToInput(outs[(size_t) presKey[l]].name, ins[(size_t) pastKey[l]].name, {}) != Status::Ok ||
                sess->linkOutputToInput(outs[(size_t) presVal[l]].name, ins[(size_t) pastVal[l]].name, {}) != Status::Ok)
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

    int p = 0; // absolute position across the whole conversation

    // Feed one token at the current position; return the logits (vocab floats) or nullptr on error.
    auto step = [&](int64_t tok) -> const float * {
        setI64(idIdx, {tok});
        setI64(posIdx, {(int64_t) p});
        std::vector<int64_t> am((size_t) C + 1, 0);
        for (int j = 0; j < p && j < C; ++j)
        {
            am[(size_t) j] = 1; // valid past slots
        }
        am[(size_t) C] = 1; // the current token (appended at index C)
        setI64(maskIdx, am);

        Status runStatus;
        if (kvLink)
        {
            // The engine folds the PREVIOUS token's present row (index C of each head) into cache
            // slot p-1 at the start of this run (min guards the overrun past the compiled context,
            // like the host loop's slot clamp).
            if (p > 0)
            {
                const int64_t          slot = (p - 1) < C ? (p - 1) : C - 1;
                std::vector<LinkRange> ranges((size_t) kvHeads);
                for (int64_t h = 0; h < kvHeads; ++h)
                {
                    ranges[(size_t) h] = {(h * (C + 1) + C) * headDim, (h * C + slot) * headDim, headDim};
                }
                for (int l = 0; l < L; ++l)
                {
                    if (sess->linkOutputToInput(outs[(size_t) presKey[l]].name, ins[(size_t) pastKey[l]].name, ranges) != Status::Ok ||
                        sess->linkOutputToInput(outs[(size_t) presVal[l]].name, ins[(size_t) pastVal[l]].name, ranges) != Status::Ok)
                    {
                        fprintf(stderr, "[chat] KV link update failed\n");
                        return nullptr;
                    }
                }
            }
            std::vector<IOTensor> bound {inputs[(size_t) idIdx], inputs[(size_t) maskIdx], inputs[(size_t) posIdx]};
            runStatus = sess->run(bound, outputs);
        } else
        {
            runStatus = sess->run(inputs, outputs);
        }
        if (runStatus != Status::Ok)
        {
            fprintf(stderr, "[chat] run failed\n");
            return nullptr;
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
            // Host fold: append the new token's key/value (present slot C) into cache slot p
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
                        const float *s = src + ((size_t) h * (C + 1) + C) * headDim;
                        float       *d = dst + ((size_t) h * C + slot) * headDim;
                        std::memcpy(d, s, (size_t) headDim * sizeof(float));
                    }
                }
            }
        }
        return reinterpret_cast<const float *>(outputs[(size_t) outIdxByInfo[logitsIdx]].data.data());
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

        const float *logits = nullptr;
        for (int64_t tk: prompt)
        {
            logits = step(tk);
            if (!logits)
            {
                return 3;
            }
            ++p;
        }
        for (int n = 0; n < maxTokens; ++n)
        {
            const int64_t next = sample(logits);
            if (next == eos)
            {
                break;
            }
            printf("%lld\n", (long long) next);
            fflush(stdout);
            logits = step(next);
            if (!logits)
            {
                return 3;
            }
            ++p;
        }
        printf("END\n");
        fflush(stdout);
    }
    return 0;
}
