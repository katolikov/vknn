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
//   vknn_chat model.vxm [--draft draft.vxm] [--config PATH] [--backend vulkan|cpu]
//             [--precision low|normal|high] [--fp32-tensors CSV] [--max-tokens N] [--temp T]
//             [--top-k K] [--top-p P] [--eos ID] [--seed S] [--chain N] [--no-kv-link]
//             [--no-prefill] [--no-gpu-argmax] [--timing]
//
// --config PATH seeds every knob from a JSON config file (Config::fromJsonFile); the flags below
// override whatever the file sets.
//
// Greedy decode (--temp 0, the default) registers the decode bucket's logits for the engine-side
// argmax (Session::setOutputArgMax): per token the engine reduces the logits on the GPU and the
// host reads back 8 bytes instead of the vocab row, with an identical token stream. A negative
// --eos disables early stop (no generated id ever matches it).
//
// --chain N (greedy + linked KV + engine argmax only) decodes in device-resident chains of N
// tokens (Session::configureDecodeChain, ADR-0015): the engine records N decode iterations into
// one command-buffer sequence — one submit + one fence per N tokens — feeding each iteration's
// token id / position / mask forward on-GPU from the previous iteration's argmax, with the KV fold
// slots for all N iterations precomputed here per chain. The token stream is bit-identical to
// --chain 1; EOS inside a chain trims the overshoot (the discarded iterations never print or count
// in tok/s). A chain never runs past the context edge un-clamped: near the edge it shortens so the
// fold-slot clamp semantics stay exactly the single-step loop's. Any ineligible combination
// (sampling, --no-kv-link, --no-gpu-argmax, no device chain path) falls back to the single-step
// loop with one stderr notice.
//
// A multi-bucket model whose second bucket takes input_ids [1,S] (S>1) prefills the prompt in
// S-token forwards instead of token-by-token — TTFT then costs one batched pass per S prompt
// tokens. The prefill pass runs on the host cache flow (its present rows fold into the host past
// buffers) and the first decode step re-seeds the engine-resident cache from them; --no-prefill
// forces the token-by-token path for A/B. Single-bucket models are unaffected.
//
// A CHUNK-prefill bucket — input_ids [1,S] with S <= kChunkPrefillTokens (io_link.h), emitted
// automatically by vknn_compile for a with-past decoder — upgrades that to the chunked-resident
// flow (llm.npu, arXiv 2407.05858): the prompt runs as ceil(T/S) sequential fixed-shape chunk
// passes whose KV cache stays ENGINE-RESIDENT inside the chunk bucket. Each pass's link ranges
// fold the PREVIOUS chunk's produced rows into their absolute cache slots (kvFoldRowRanges), the
// mask marks the cached past plus the chunk's real tokens (the graph's own causal mask orders the
// intra-chunk columns), and the last chunk pads to S with masked-out ids. One host readback per
// turn then materializes the built cache for the decode bucket's re-seed — per-chunk KV traffic is
// zero, against the whole-window path's full cache bind + present download per pass. Requires the
// engine-resident cache (linked mode), position_ids, and an fp32 KV boundary; anything else — and
// any link failure — falls back to the whole-window host flow above, with the identical token
// stream. --no-prefill disables both batched paths.
//
// The model is a with-past decoder compiled at a fixed past length C (read from the .vxm). Each step
// feeds one token at absolute position p, the [1, kv_heads, C, head_dim] cache as the past key/value
// inputs, and an attention mask marking the p valid past slots plus the current token, so a single
// fixed-shape plan serves every step. The new token's key/value (present index C) lands in cache
// slot p. This matches the reference HF greedy stream token-for-token.
//
// --draft draft.vxm turns on GREEDY SPECULATIVE DECODING (vknn/spec_decode.h). A draft model is a
// caller-supplied artifact the engine cannot manufacture, so it arrives as a driver argument the way
// the model path itself does — it names a file, it does not tune the engine. There is no flag to
// enable, size, or disable speculation: given a draft, every turn that qualifies speculates, using
// the kSpecDraftTokens compiled into the .vxm's verification bucket.
//
// Per round the draft proposes kSpecDraftTokens tokens from the pending token, and the target checks
// all of them plus their anchor in ONE forward through its [1, kSpecVerifyTokens] bucket. A proposal
// is committed only when it equals the target's own argmax at that position; the first mismatch is
// replaced by that argmax and ends the round, and a round with every proposal accepted also emits
// the extra column's argmax as a bonus token. The emitted stream is therefore the plain greedy
// stream, token for token, at one target forward per up-to-kSpecDraftTokens+1 tokens.
//
// The verification bucket's cache is ENGINE-RESIDENT between rounds, like the chunked prefill's: the
// next round's link ranges (specVerifyFoldRanges) fold ONLY the accepted rows of the previous round
// into their absolute slots, so the rejected rows are never copied anywhere and the rollback costs a
// shorter range list rather than an erase. One readback per turn materializes the built cache for
// the next turn's prefill. Speculation stands down — with one notice and the identical stream — for
// sampling (--temp > 0, which needs the modified-rejection scheme), --chain N (an explicit request
// for the other device path), --no-kv-link, a missing or unloadable draft, a draft whose vocabulary
// differs from the target's, a .vxm with no verification bucket, and any position too close to a
// compiled context edge to fit a whole verification window (the turn's tail decodes plainly).
//
// The KV cache is ENGINE-RESIDENT by default: every present.N.{key,value} output is linked to its
// past_key_values.N.{key,value} input (Session::linkOutputToInput), so per token the engine folds
// the new row into the cache in place — on the GPU backend entirely on-device, with no host copy of
// the ~2*L*kv_heads*C*head_dim cache in either direction. --no-kv-link selects the host-side loop
// (bind the cache every step, fold present rows on the host) for A/B comparison; both paths produce
// the same token stream.
#include "draft_decoder.h"
#include "vknn/runtime.h"
#include "vknn/session.h"
#include "vknn/spec_decode.h"
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
                "usage: %s model.vxm [--draft draft.vxm] [--config PATH] [--backend vulkan|cpu]\n"
                "        [--precision low|normal|high] [--fp32-tensors CSV] [--max-tokens N] [--temp T]\n"
                "        [--top-k K] [--top-p P] [--eos ID] [--seed S] [--chain N] [--no-kv-link]\n"
                "        [--no-prefill] [--no-gpu-argmax] [--timing]\n",
                argv[0]);
        return 1;
    }
    std::string model = argv[1];

    // A JSON config file (Config::fromJsonFile) seeds every knob; the individual flags below then
    // layer on top, so a command-line flag always wins over the file.
    Config            cfg;
    const std::string configPath = opt(argc, argv, "--config", "");
    if (!configPath.empty())
    {
        cfg = Config::fromJsonFile(configPath);
    }
    cfg.backend                = backendFromStr(opt(argc, argv, "--backend", "vulkan"));
    cfg.precision              = precisionFromStr(opt(argc, argv, "--precision", "low"));
    cfg.fp32Tensors            = opt(argc, argv, "--fp32-tensors", "");
    cfg.freeWeightsAfterUpload = true;
    const int     maxTokens    = atoi(opt(argc, argv, "--max-tokens", "128"));
    const float   temp         = (float) atof(opt(argc, argv, "--temp", "0"));
    const int     topK         = atoi(opt(argc, argv, "--top-k", "0"));
    const float   topP         = (float) atof(opt(argc, argv, "--top-p", "1"));
    const int64_t eos          = atoll(opt(argc, argv, "--eos", "151643"));
    bool          kvLink       = !flagSet(argc, argv, "--no-kv-link");             // may drop to the host loop mid-stream on a link failure
    cfg.timing                 = flagSet(argc, argv, "--timing");                  // per-run pack/submit/unpack walls
    cfg.timingSummary          = flagSet(argc, argv, "--timing-summary");          // silent per-segment submit/sync accumulators, one line at teardown
    const int maxSubmitNodes   = atoi(opt(argc, argv, "--max-submit-nodes", "0")); // >0 overrides Config::maxSubmitNodes (command-buffer chunking)
    if (maxSubmitNodes > 0)
    {
        cfg.maxSubmitNodes = maxSubmitNodes;
    }
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
    if (flagSet(argc, argv, "--kv-concat-fold"))
    {
        cfg.setHint(Hint::KvConcatFold, (int) Mode::On);
    }
    // --chain N: device-resident decode chains of N tokens, greedy + linked KV + engine argmax
    // only. Any ineligible combination drops to the single-step loop with one notice; the chain
    // length reaches the engine through Config::decodeChainSteps (set before load, so the decode
    // segment sizes its per-iteration buffers).
    int  chainSteps  = atoi(opt(argc, argv, "--chain", "1"));
    bool chainWanted = chainSteps > 1;
    if (chainWanted && temp > 0.0f)
    {
        fprintf(stderr, "[chat] --chain needs greedy decode (--temp 0); using the single-step loop\n");
        chainWanted = false;
    }
    if (chainWanted && !kvLink)
    {
        fprintf(stderr, "[chat] --chain needs the engine-resident KV cache (drop --no-kv-link); using the single-step loop\n");
        chainWanted = false;
    }
    if (chainWanted && flagSet(argc, argv, "--no-gpu-argmax"))
    {
        fprintf(stderr, "[chat] --chain needs the engine argmax (drop --no-gpu-argmax); using the single-step loop\n");
        chainWanted = false;
    }
    cfg.decodeChainSteps = chainWanted ? chainSteps : 1;
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
    // bucket with S <= kChunkPrefillTokens additionally drives the chunked-resident prefill (the
    // largest such S wins: fewer passes per prompt); the whole-window role keeps the largest S
    // overall as the fallback. A single-bucket model keeps the token-by-token prefill below,
    // byte-identically to before.
    // A bucket at exactly [1, kSpecVerifyTokens] additionally takes the speculative-verification
    // role. It stays eligible for the prefill and chunk roles as well: when a model carries no other
    // widened bucket it is better used as a narrow chunk bucket than not at all, and the two roles
    // never run at the same moment (a turn prefills, then decodes).
    int decodeBucket = -1, prefillBucket = -1, prefillS = 0;
    int chunkBucket = -1, chunkS = 0;
    int specBucket = -1;
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
                } else if (in.shape[1] > 1)
                {
                    if (in.shape[1] == kSpecVerifyTokens && specBucket < 0)
                    {
                        specBucket = (int) b;
                    }
                    if ((int) in.shape[1] > prefillS)
                    {
                        prefillBucket = (int) b;
                        prefillS      = (int) in.shape[1];
                    }
                    // The consumer is gated by the same switch as the emitter: a .vxm compiled
                    // while the chunk bucket was still emitted keeps that bucket, and routing to it
                    // would take the path that decodes a multi-chunk prompt as though only its last
                    // chunk were present. Such a model falls back to its whole-window bucket.
                    if (kChunkPrefillEnabled && in.shape[1] <= kChunkPrefillTokens && (int) in.shape[1] > chunkS)
                    {
                        chunkBucket = (int) b;
                        chunkS      = (int) in.shape[1];
                    }
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
        chunkBucket   = -1;
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
    // position_ids is optional: a GroupQueryAttention-style export derives each token's rotary
    // position internally from the attention mask, so it exposes no position_ids input. When absent
    // (posIdx < 0) the driver simply binds no position tensor; the mask and KV cache are enough.
    if (idIdx < 0 || maskIdx < 0 || logitsIdx < 0 || L == 0)
    {
        fprintf(stderr, "model is not a with-past decoder (missing input_ids/attention_mask/"
                        "logits/past_key_values.*)\n");
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
    // That row count drives EVERY layer's fold (the link ranges and the host copies all take their
    // source row from it), so every layer must report it. A model whose layers mix the two
    // conventions would have the odd ones' cache seeded from the wrong present rows silently (the
    // fold source stays in bounds, it just addresses the wrong block), so a mismatch is refused
    // here rather than decoded through.
    for (int l = 0; l < L; ++l)
    {
        for (int part = 0; part < 2; ++part)
        {
            const int idx = part ? presVal[l] : presKey[l];
            if (idx < 0 || outs[(size_t) idx].shape.size() != 4 || (int) outs[(size_t) idx].shape[2] != presRows)
            {
                fprintf(stderr, "present output '%s' reports a different row count than '%s' (%d); the layers disagree on the present convention and one cache fold cannot serve both\n",
                        idx >= 0 ? outs[(size_t) idx].name.c_str() : "(missing)", outs[(size_t) presKey[0]].name.c_str(), presRows);
                return 2;
            }
        }
    }
    // The KV cache and logits keep the model's declared boundary dtype on host readback (an fp16
    // export downloads fp16, not fp32). The host-side present-row folds copy raw rows, so they work
    // in the cache's element size; the prefill logits row is converted to fp32 for host sampling.
    const size_t kvElemBytes = dtypeSize(ins[(size_t) pastKey[0]].dtype);
    const bool   logitsFp16  = outs[(size_t) logitsIdx].dtype == DType::Float16;
    // A host present-row fold (the no-kv-link decode loop and token-by-token prefill) copies the
    // present output straight into the past buffer, so it needs the two to share an element size.
    // The engine-resident KV link never host-folds and is unaffected, so this only gates the host
    // fold paths rather than rejecting the model.
    const bool hostFoldDtypeOk = presKey[0] < 0 || dtypeSize(outs[(size_t) presKey[0]].dtype) == kvElemBytes;
    if (!kvLink && !hostFoldDtypeOk)
    {
        fprintf(stderr, "--no-kv-link needs the present and past KV outputs to share an element size (%zu vs %zu bytes); rerun with the engine-resident cache\n",
                dtypeSize(outs[(size_t) presKey[0]].dtype), kvElemBytes);
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
    // With a prefill (or chunk) bucket the host past buffers are needed even in linked mode: the
    // whole-window pass runs on the host cache flow, the chunked flow materializes its resident
    // cache into them once per turn, and the first decode step re-seeds the engine-resident cache
    // from them.
    std::vector<IOTensor> inputs(ins.size());
    for (size_t i = 0; i < ins.size(); ++i)
    {
        inputs[i].name  = ins[i].name;
        inputs[i].shape = ins[i].shape;
        inputs[i].dtype = ins[i].dtype;
        if (!(kvLink && isPastInput(i) && prefillBucket < 0 && chunkBucket < 0))
        {
            inputs[i].data.assign((size_t) ins[i].elems * dtypeSize(ins[i].dtype), 0);
        }
    }
    auto setI64 = [&](int idx, const std::vector<int64_t> &vals) {
        if (idx < 0)
        {
            return; // an optional input the model does not expose (e.g. a GQA export's position_ids)
        }
        std::memcpy(inputs[idx].data.data(), vals.data(), vals.size() * sizeof(int64_t));
    };
    // The bound-input set for a linked/chained run: input_ids + attention_mask, plus position_ids
    // only when the model exposes it (a GQA export derives position internally, so posIdx < 0).
    auto boundInputs = [&]() {
        std::vector<IOTensor> bound {inputs[(size_t) idIdx], inputs[(size_t) maskIdx]};
        if (posIdx >= 0)
        {
            bound.push_back(inputs[(size_t) posIdx]);
        }
        return bound;
    };

    // Batched-prefill geometry, validated against the decode bucket: the past inputs must share the
    // decode shapes (one host cache serves both) and the mask must span past+S columns. Any
    // mismatch disables that batched path rather than miscomputing.
    // Both batched paths fold the produced KV rows into the cache keyed to the position_ids the
    // driver feeds. A model that derives position internally from the mask (no position_ids input)
    // is prefilled token-by-token instead: the batched paths are only validated against the
    // position_ids convention, and a mismatched prefill seeds a wrong cache.
    if ((prefillBucket >= 0 || chunkBucket >= 0) && posIdx < 0)
    {
        fprintf(stderr, "[chat] batched prefill needs a position_ids input; this model derives position internally, prefilling token-by-token\n");
        prefillBucket = -1;
        chunkBucket   = -1;
    }
    // Validate one batched-prefill bucket of window S; fills the mask span, the present row count,
    // and the logits output index. False (with the mismatch line) disables the bucket.
    auto validateBatchedBucket = [&](int bucket, int windowS, const char *role, int *maskLen, int *presRows, int *logitsOutIdx) -> bool {
        const std::vector<IOInfo> pin  = sess->inputInfo((size_t) bucket);
        const std::vector<IOInfo> pout = sess->outputInfo((size_t) bucket);
        bool                      ok   = true;
        *maskLen                       = 0;
        *presRows                      = 0;
        *logitsOutIdx                  = -1;
        for (const IOInfo &in: pin)
        {
            if (in.name == ins[(size_t) pastKey[0]].name)
            {
                ok = ok && in.shape == ins[(size_t) pastKey[0]].shape;
            }
            if (in.name == "attention_mask" && in.shape.size() == 2)
            {
                *maskLen = (int) in.shape[1];
            }
        }
        for (size_t i = 0; i < pout.size(); ++i)
        {
            if (pout[i].name == "logits")
            {
                *logitsOutIdx = (int) i;
            }
            if (pout[i].name == "present.0.key" && pout[i].shape.size() == 4)
            {
                *presRows = (int) pout[i].shape[2];
            }
        }
        ok = ok && *maskLen == C + windowS && *presRows >= windowS && *logitsOutIdx >= 0;
        // One present row count serves every layer's fold in this bucket too (the batched paths read
        // the produced rows at presRows - windowS for all layers), so layers that disagree disable
        // the bucket instead of folding some of them from the wrong rows.
        for (const IOInfo &out: pout)
        {
            if (out.name.rfind("present.", 0) == 0 && out.shape.size() == 4 && (int) out.shape[2] != *presRows)
            {
                ok = false;
            }
        }
        if (!ok)
        {
            fprintf(stderr, "[chat] %s bucket geometry mismatch (mask %d vs C+S %d, present rows %d); path disabled\n", role, *maskLen, C + windowS, *presRows);
        }
        return ok;
    };
    int prefillMaskLen = 0, prefillPresRows = 0, prefillLogitsIdx = -1;
    if (prefillBucket >= 0)
    {
        if (!validateBatchedBucket(prefillBucket, prefillS, "prefill", &prefillMaskLen, &prefillPresRows, &prefillLogitsIdx))
        {
            prefillBucket = -1;
        } else
        {
            fprintf(stderr, "[chat] prefill bucket: S=%d, one forward per %d prompt tokens\n", prefillS, prefillS);
        }
    }
    // The chunked-resident flow additionally needs linked mode (its cache lives inside the chunk
    // bucket between passes) and an fp32 KV boundary (the once-per-turn readResident materializes
    // the cache in the engine's fp32 storage, byte-compatible with an fp32 host buffer only).
    int chunkMaskLen = 0, chunkPresRows = 0, chunkLogitsIdx = -1;
    if (chunkBucket >= 0 && !kvLink)
    {
        chunkBucket = -1; // --no-kv-link: the whole-window host flow serves the A/B reference
    }
    if (chunkBucket >= 0 && kvElemBytes != 4)
    {
        fprintf(stderr, "[chat] chunked prefill needs an fp32 KV boundary (have %zu-byte); using the whole-window path\n", kvElemBytes);
        chunkBucket = -1;
    }
    if (chunkBucket >= 0)
    {
        if (!validateBatchedBucket(chunkBucket, chunkS, "chunk-prefill", &chunkMaskLen, &chunkPresRows, &chunkLogitsIdx))
        {
            chunkBucket = -1;
        } else
        {
            fprintf(stderr, "[chat] chunk-prefill bucket: S=%d, resident chunk passes\n", chunkS);
        }
    }

    // --- speculative decoding: prerequisites --------------------------------------------------
    // Every refusal here is a NOTICE, not an error: speculation is a throughput optimization whose
    // prerequisite (a second model) the engine cannot manufacture, and the plain decode loop below
    // produces the identical stream. The gates are the ones the round's correctness argument rests
    // on — greedy sampling, an engine-resident cache to fold accepted rows into, a verification
    // bucket to run the batched forward through, and a draft over the same token vocabulary.
    const std::string             draftPath = opt(argc, argv, "--draft", "");
    std::unique_ptr<DraftDecoder> draft;
    int                           specMaskLen = 0, specPresRows = 0, specLogitsIdx = -1;
    if (!draftPath.empty())
    {
        if (temp > 0.0f)
        {
            fprintf(stderr, "[chat] --draft needs greedy decode (--temp 0): sampled speculation needs the modified-rejection scheme, which is not implemented; "
                            "decoding without speculation\n");
            specBucket = -1;
        } else if (chainWanted)
        {
            fprintf(stderr,
                    "[chat] --draft and --chain are two device paths for the same loop; --chain was asked for explicitly, so speculation stands down\n");
            specBucket = -1;
        } else if (!kvLink)
        {
            fprintf(stderr, "[chat] --draft needs the engine-resident KV cache (drop --no-kv-link); decoding without speculation\n");
            specBucket = -1;
        } else if (kvElemBytes != 4)
        {
            // The once-per-turn readResident materializes the verification cache in the engine's
            // fp32 storage, byte-compatible with an fp32 host buffer only — the chunked flow's rule.
            fprintf(stderr, "[chat] speculation needs an fp32 KV boundary (have %zu-byte); decoding without speculation\n", kvElemBytes);
            specBucket = -1;
        } else if (posIdx < 0)
        {
            fprintf(stderr, "[chat] speculation feeds a window of absolute positions; this model derives position internally, decoding without speculation\n");
            specBucket = -1;
        } else if (specBucket < 0)
        {
            fprintf(stderr, "[chat] this model has no input_ids [1,%lld] verification bucket (recompile with vknn_compile to get one); decoding without speculation\n", (long long) kSpecVerifyTokens);
        } else if (!validateBatchedBucket(specBucket, (int) kSpecVerifyTokens, "spec-verify", &specMaskLen, &specPresRows, &specLogitsIdx))
        { specBucket = -1; }
        if (specBucket >= 0)
        {
            draft = DraftDecoder::open(draftPath, cfg);
            if (draft && draft->vocab() != vocab)
            {
                fprintf(stderr, "[chat] draft vocabulary %lld does not match the target's %lld; its token ids are not this model's ids, decoding without speculation\n",
                        (long long) draft->vocab(), (long long) vocab);
                draft.reset();
            }
            if (!draft)
            {
                specBucket = -1;
            }
        }
    }
    const bool specActive     = specBucket >= 0 && draft;
    bool       specLogitsFp16 = false;
    if (specActive)
    {
        // The verification bucket declares its own logits dtype; reading the decode bucket's would
        // misinterpret every row of the block on a model whose buckets differ.
        for (const IOInfo &out: sess->outputInfo((size_t) specBucket))
        {
            if (out.name == "logits")
            {
                specLogitsFp16 = out.dtype == DType::Float16;
            }
        }
        // A round binds the host past buffers once per turn to re-seed the verification bucket's
        // resident cache, so they must exist even when no prefill bucket asked for them
        // (--no-prefill leaves them unallocated).
        for (int l = 0; l < L; ++l)
        {
            for (int part = 0; part < 2; ++part)
            {
                const size_t idx = (size_t) (part ? pastVal[l] : pastKey[l]);
                if (inputs[idx].data.empty())
                {
                    inputs[idx].data.assign((size_t) ins[idx].elems * kvElemBytes, 0);
                }
            }
        }
        fprintf(stderr, "[chat] speculative decode: %lld drafts verified per target forward (bucket S=%lld)\n", (long long) kSpecDraftTokens, (long long) kSpecVerifyTokens);
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

    // Configure the device decode chain once the links and the engine argmax are in place. Any
    // failure keeps the single-step loop with one notice; a chained run itself never falls back.
    bool chainConfigured = false; // the decode segment records chains (window set per run)
    bool chainActive     = false; // the decode loop drives chains (requires live KV links)
    if (chainWanted)
    {
        if (!gpuArgmax)
        {
            fprintf(stderr, "[chat] --chain needs the engine argmax path; using the single-step loop\n");
        } else if (posIdx < 0)
        {
            // The chain feeds each iteration's position forward by writing position_ids on-device;
            // a model that derives position internally (no position_ids input) has no tensor to
            // drive, so it stays on the single-step loop.
            fprintf(stderr, "[chat] --chain needs a position_ids input; this model derives position internally, using the single-step loop\n");
        } else if (sess->configureDecodeChain((size_t) decodeBucket, "input_ids", "position_ids", "attention_mask", "logits") != Status::Ok)
        {
            fprintf(stderr, "[chat] decode chain unavailable (see log); using the single-step loop\n");
        } else
        {
            chainConfigured = true;
            chainActive     = true;
            fprintf(stderr, "[chat] decode chains of %d tokens\n", chainSteps);
        }
    }

    std::vector<IOTensor> outputs;
    // Map an output name to its index in the run() result vector (stable across runs).
    std::vector<int> outIdxByInfo(outs.size(), -1);
    bool             mapped = false;

    // Every token the target has been fed, in order, and how many of them the draft's cache already
    // holds. The draft mirrors the conversation, not the target's engine state, so one list keeps it
    // aligned across every path that can advance the target without it: a turn where speculation
    // stood down, the token-by-token tail past the context edge, a turn that ended at end-of-stream.
    // Only maintained under --draft (a chained decode disables speculation, so the two never mix).
    std::vector<int64_t> fedTokens;
    int                  draftValid = 0;

    int  p             = 0;     // absolute position across the whole conversation
    bool residentDirty = false; // linked decode ran: the engine cache is ahead of the host buffers
    bool reseedCache   = false; // next linked decode step re-seeds the resident cache from the host
    // The last linked run's present rows still need their fold into slot p-1 (the single-step
    // steady state). False only after an EOS-trimmed chain whose overshoot iteration already
    // applied that fold on-device — folding again would move the OVERSHOOT row into a live slot.
    bool pendingResidentFold = true;

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
                    hostPast.data.assign((size_t) ins[(size_t) (part ? pastVal[l] : pastKey[l])].elems * kvElemBytes, 0);
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
                    const uint8_t *src = present.data.data();
                    uint8_t       *dst = hostPast.data.data();
                    for (int h = 0; h < kvHeads; ++h)
                    {
                        std::memcpy(dst + ((size_t) h * C + pendingSlot) * headDim * kvElemBytes, src + ((size_t) h * presRows + (presRows - 1)) * headDim * kvElemBytes, (size_t) headDim * kvElemBytes);
                    }
                }
            }
        }
        return true;
    };

    // Decode-loop phase walls (--timing): accumulated per decode step, reported per turn.
    double tmLinkMs = 0, tmRunMs = 0, tmPrepMs = 0, tmSampleMs = 0;
    int    tmSteps = 0;
    auto   nowMs   = [] {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };

    // The last forward's logits row and its provenance. A decode step under engine argmax returns
    // the logits entry with no data (lastLogits stays null; readOutputArgMax serves the token); a
    // prefill pass always yields a full host row (its bucket is never argmax-registered).
    const float       *lastLogits     = nullptr;
    bool               lastFromDecode = false;
    std::vector<float> decodeLogitsF32; // fp32 copy of an fp16 decode logits row for host sampling

    // Feed one token at the current position; false on error.
    auto step = [&](int64_t tok) -> bool {
        const double tPrep0 = nowMs();
        // A chain-configured decode segment stays chain-recorded; a single step runs its first
        // iteration only (the prompt / fallback path).
        if (chainConfigured && sess->setDecodeChainWindow((size_t) decodeBucket, p, 1) != Status::Ok)
        {
            fprintf(stderr, "[chat] decode chain window update failed (see log)\n");
            return false;
        }
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
                const int64_t                slot   = (!reseedCache && pendingResidentFold) ? kvFoldSlot(p, C) : -1;
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
                    std::vector<IOTensor> bound = boundInputs();
                    runStatus                   = sess->run(bound, outputs);
                }
                ranLinked           = true;
                residentDirty       = true;
                pendingResidentFold = true;
            } else
            {
                // A mid-stream link failure: bring the engine-resident cache (device rows + the
                // pending fold from the last run's present) back into the host past buffers, drop
                // the links, and continue THIS and every later step on the host cache loop — same
                // tokens, no lost state.
                fprintf(stderr, "[chat] switching to the host KV loop at p=%d (resyncing the cache from the engine)\n", p);
                if (!syncResidentToHost(pendingResidentFold ? kvFoldSlot(p, C) : -1))
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
                    const uint8_t  *src  = pres.data.data();
                    uint8_t        *dst  = past.data.data();
                    for (int h = 0; h < kvHeads; ++h)
                    {
                        const uint8_t *s = src + ((size_t) h * presRows + (presRows - 1)) * headDim * kvElemBytes;
                        uint8_t       *d = dst + ((size_t) h * C + slot) * headDim * kvElemBytes;
                        std::memcpy(d, s, (size_t) headDim * kvElemBytes);
                    }
                }
            }
        }
        const IOTensor &logitsOut = outputs[(size_t) outIdxByInfo[logitsIdx]];
        // An fp16-declared logits output downloads as raw fp16; convert the row to fp32 so the host
        // sampler reads real values (the engine-argmax greedy path leaves this null and is unaffected).
        if (logitsOut.data.empty())
        {
            lastLogits = nullptr;
        } else if (logitsFp16)
        {
            decodeLogitsF32.resize((size_t) vocab);
            halfToFloatBulk(reinterpret_cast<const fp16_t *>(logitsOut.data.data()), decodeLogitsF32.data(), vocab);
            lastLogits = decodeLogitsF32.data();
        } else
        {
            lastLogits = reinterpret_cast<const float *>(logitsOut.data.data());
        }
        lastFromDecode = true;
        return true;
    };

    // Run one device chain of `steps` decode iterations feeding `tok` at position p (iterations
    // i > 0 feed the previous iteration's argmax forward on-GPU). Every iteration's KV fold ranges
    // are precomputed here by the single-step rule (kvFoldSlot/kvFoldRanges — one source of truth)
    // and indexed per iteration on-device. p itself is advanced by the caller, which may trim the
    // logical advance below `steps` on a mid-chain EOS. Returns 1 on success, 0 on a fatal error,
    // -1 after a link failure fell the stream back to the host loop (`tok` was not fed).
    auto chainStep = [&](int64_t tok, int steps) -> int {
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
        // Iteration i folds the row iteration i-1 produced into slot p+i-1; iteration 0 folds the
        // previous run's pending row (or nothing on a re-seed / no-pending step) — the single-step
        // loop's exact rule per iteration.
        std::vector<std::vector<LinkRange>> rangeSets((size_t) steps);
        for (int i = 0; i < steps; ++i)
        {
            const int64_t slot    = i == 0 ? ((!reseedCache && pendingResidentFold) ? kvFoldSlot(p, C) : -1) : kvFoldSlot(p + i, C);
            rangeSets[(size_t) i] = kvFoldRanges(kvHeads, presRows, C, headDim, slot);
        }
        bool linksOk = true;
        for (int l = 0; l < L && linksOk; ++l)
        {
            for (int part = 0; part < 2 && linksOk; ++part)
            {
                const std::string &pres = outs[(size_t) (part ? presVal[l] : presKey[l])].name;
                const std::string &past = ins[(size_t) (part ? pastVal[l] : pastKey[l])].name;
                if (sess->linkOutputToInputChain((size_t) decodeBucket, pres, past, rangeSets) != Status::Ok)
                {
                    fprintf(stderr, "[chat] KV chain link update failed for %s -> %s (see log)\n", pres.c_str(), past.c_str());
                    linksOk = false;
                }
            }
        }
        if (linksOk && sess->setDecodeChainWindow((size_t) decodeBucket, p, steps) != Status::Ok)
        {
            fprintf(stderr, "[chat] decode chain window update failed (see log)\n");
            linksOk = false;
        }
        const double tRun0 = nowMs();
        tmLinkMs += tRun0 - tLink0;
        if (!linksOk)
        {
            // The single-step loop's recovery: resync the cache to the host, drop the links, and
            // continue on the host KV loop — same tokens, no lost state.
            fprintf(stderr, "[chat] switching to the host KV loop at p=%d (resyncing the cache from the engine)\n", p);
            if (!syncResidentToHost(pendingResidentFold ? kvFoldSlot(p, C) : -1))
            {
                return 0;
            }
            sess->clearLinks();
            kvLink = false;
            return -1;
        }
        Status runStatus;
        if (reseedCache)
        {
            runStatus   = sess->run(inputs, outputs); // full bind re-seeds the resident cache
            reseedCache = false;
        } else
        {
            std::vector<IOTensor> bound = boundInputs();
            runStatus                   = sess->run(bound, outputs);
        }
        residentDirty       = true;
        pendingResidentFold = true;
        tmRunMs += nowMs() - tRun0;
        tmSteps += steps; // a chain of N counts as N decode steps
        if (runStatus != Status::Ok)
        {
            fprintf(stderr, "[chat] run failed\n");
            return 0;
        }
        lastLogits     = nullptr; // chained decode is argmax-only; no logits row exists
        lastFromDecode = true;
        return 1;
    };

    // One prefill pass: feed `len` prompt tokens (<= prefillS) starting at absolute position p in a
    // single forward through the prefill bucket, on the HOST cache flow — bind the past buffers,
    // fold the produced present rows (the S new rows after the past block) back into cache slots
    // p..p+len-1 — and return the last real token's logits row. Pad slots carry mask 0, so real
    // tokens never attend them and their (garbage) rows are simply not folded. Advances p.
    std::vector<IOTensor> prefillOutputs;
    std::vector<float>    prefillLogitsF32; // fp32 copy of an fp16 logits row for host sampling
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
        // Ask the engine to read back only the last real token's logits row (the one this pass
        // returns) instead of the whole [S, vocab] matrix — the other rows never feed a token, so
        // downloading them is pure TTFT cost. Ok => "logits" arrives as a single [1,1,vocab] row
        // (offset 0); Unsupported (CPU backend / non-flat) keeps the full matrix and the row offset.
        const bool rowSliced = sess->setOutputRow((size_t) prefillBucket, "logits", len - 1) == Status::Ok;
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
                const uint8_t *src = pres->data.data();
                uint8_t       *dst = inputs[(size_t) (part ? pastVal[l] : pastKey[l])].data.data();
                for (int h = 0; h < kvHeads; ++h)
                {
                    for (int t = 0; t < len; ++t)
                    {
                        std::memcpy(dst + ((size_t) h * C + p + t) * headDim * kvElemBytes, src + ((size_t) h * prefillPresRows + newRowsAt + t) * headDim * kvElemBytes, (size_t) headDim * kvElemBytes);
                    }
                }
            }
        }
        p += len;
        // The last real token's logits row feeds the first decode token's host sampling. An fp16
        // logits output is converted to fp32 into a persistent row so the sampler reads real values.
        const size_t rowOffset = rowSliced ? 0 : (size_t) (len - 1) * vocab;
        if (logitsFp16)
        {
            prefillLogitsF32.resize((size_t) vocab);
            halfToFloatBulk(reinterpret_cast<const fp16_t *>(logitsOut->data.data()) + rowOffset, prefillLogitsF32.data(), vocab);
            return prefillLogitsF32.data();
        }
        return reinterpret_cast<const float *>(logitsOut->data.data()) + rowOffset;
    };

    // --- chunked-resident prefill --------------------------------------------------------------
    // The chunk bucket's KV cache is engine-resident BETWEEN chunk passes: pass i's link ranges
    // fold pass i-1's produced rows (a contiguous row block per head, kvFoldRowRanges) into their
    // absolute cache slots at the start of run i, so no cache bytes cross the host per chunk. The
    // session holds one link set at a time, so a turn switches the links chunk-bucket-ward for its
    // chunk passes and back to the decode bucket afterwards (readResident is name-keyed and stays
    // unambiguous that way). Pad ids on the last chunk use the eos id (or 0 under a negative
    // --eos sentinel): their mask stays 0, their rows never fold, their logits never feed a token.
    const int64_t chunkPadId = (eos >= 0 && eos < vocab) ? eos : 0;

    // Link every chunk-bucket present output to its past input (ranges re-armed per pass). False
    // leaves the session's links cleared for the caller to restore.
    auto linkChunkBucket = [&]() -> bool {
        for (int l = 0; l < L; ++l)
        {
            for (int part = 0; part < 2; ++part)
            {
                const std::string &pres = outs[(size_t) (part ? presVal[l] : presKey[l])].name;
                const std::string &past = ins[(size_t) (part ? pastVal[l] : pastKey[l])].name;
                if (sess->linkOutputToInput((size_t) chunkBucket, pres, past, {}) != Status::Ok)
                {
                    fprintf(stderr, "[chat] chunk-prefill link failed for %s -> %s (see log)\n", pres.c_str(), past.c_str());
                    return false;
                }
            }
        }
        return true;
    };

    // Re-establish the decode bucket's links after a chunked turn (empty ranges; the per-step
    // folds re-arm before each decode run).
    auto relinkDecode = [&]() -> bool {
        for (int l = 0; l < L; ++l)
        {
            for (int part = 0; part < 2; ++part)
            {
                const std::string &pres = outs[(size_t) (part ? presVal[l] : presKey[l])].name;
                const std::string &past = ins[(size_t) (part ? pastVal[l] : pastKey[l])].name;
                if (sess->linkOutputToInput((size_t) decodeBucket, pres, past, {}) != Status::Ok)
                {
                    fprintf(stderr, "[chat] decode re-link failed for %s -> %s (see log)\n", pres.c_str(), past.c_str());
                    return false;
                }
            }
        }
        return true;
    };

    // One chunk pass: feed `len` prompt tokens (<= chunkS) at absolute position p, with link
    // ranges folding the PREVIOUS chunk's rows (slots prevStart..prevStart+prevLen-1) at run
    // start. The first pass of a turn binds the host past buffers, reinitializing the chunk
    // bucket's resident cache to the conversation state (empty ranges then — nothing pends).
    // Leaves the last real token's logits row in lastLogits.
    std::vector<IOTensor> chunkOutputs;
    std::vector<float>    chunkLogitsF32; // fp32 copy of an fp16 logits row for host sampling
    auto                  runChunk = [&](const int64_t *toks, int len, int prevStart, int prevLen, bool bindPast) -> bool {
        const int                    newRowsAt = chunkPresRows - chunkS; // produced rows sit after any past block
        const std::vector<LinkRange> ranges = kvFoldRowRanges(kvHeads, chunkPresRows, C, headDim, newRowsAt, prevLen > 0 ? prevStart : -1, prevLen);
        for (int l = 0; l < L; ++l)
        {
            for (int part = 0; part < 2; ++part)
            {
                const std::string &pres = outs[(size_t) (part ? presVal[l] : presKey[l])].name;
                const std::string &past = ins[(size_t) (part ? pastVal[l] : pastKey[l])].name;
                if (sess->linkOutputToInput((size_t) chunkBucket, pres, past, ranges) != Status::Ok)
                {
                    fprintf(stderr, "[chat] chunk-prefill link update failed for %s -> %s (see log)\n", pres.c_str(), past.c_str());
                    return false;
                }
            }
        }
        IOTensor ids, mask, pos;
        ids.name  = inputs[(size_t) idIdx].name;
        ids.dtype = ins[(size_t) idIdx].dtype;
        ids.shape = {1, (int64_t) chunkS};
        pos.name  = inputs[(size_t) posIdx].name;
        pos.dtype = ins[(size_t) posIdx].dtype;
        pos.shape = {1, (int64_t) chunkS};
        {
            std::vector<int64_t> idVals((size_t) chunkS), posVals((size_t) chunkS);
            for (int t = 0; t < chunkS; ++t)
            {
                idVals[(size_t) t]  = t < len ? toks[t] : chunkPadId;
                posVals[(size_t) t] = (int64_t) (p + t);
            }
            ids.data.resize((size_t) chunkS * 8);
            std::memcpy(ids.data.data(), idVals.data(), ids.data.size());
            pos.data.resize((size_t) chunkS * 8);
            std::memcpy(pos.data.data(), posVals.data(), pos.data.size());
        }
        mask.name  = inputs[(size_t) maskIdx].name;
        mask.dtype = ins[(size_t) maskIdx].dtype;
        mask.shape = {1, (int64_t) chunkMaskLen};
        {
            // Valid past slots (all previously folded rows, including earlier chunks of this
            // turn) plus the chunk's real tokens; pads stay masked. The graph's own causal mask
            // orders the intra-chunk columns.
            std::vector<int64_t> maskVals((size_t) chunkMaskLen, 0);
            for (int j = 0; j < p && j < C; ++j)
            {
                maskVals[(size_t) j] = 1;
            }
            for (int t = 0; t < len; ++t)
            {
                maskVals[(size_t) (C + t)] = 1;
            }
            mask.data.resize((size_t) chunkMaskLen * 8);
            std::memcpy(mask.data.data(), maskVals.data(), mask.data.size());
        }
        const bool            rowSliced = sess->setOutputRow((size_t) chunkBucket, "logits", len - 1) == Status::Ok;
        std::vector<IOTensor> bound {ids, mask, pos};
        if (bindPast)
        {
            for (int l = 0; l < L; ++l)
            {
                bound.push_back(inputs[(size_t) pastKey[l]]);
                bound.push_back(inputs[(size_t) pastVal[l]]);
            }
        }
        if (sess->run(bound, chunkOutputs) != Status::Ok)
        {
            fprintf(stderr, "[chat] chunk-prefill run failed\n");
            return false;
        }
        const IOTensor *logitsOut = nullptr;
        for (const IOTensor &o: chunkOutputs)
        {
            if (o.name == "logits")
            {
                logitsOut = &o;
            }
        }
        if (!logitsOut || logitsOut->data.empty())
        {
            fprintf(stderr, "[chat] chunk-prefill logits missing\n");
            return false;
        }
        const size_t rowOffset = rowSliced ? 0 : (size_t) (len - 1) * vocab;
        if (logitsFp16)
        {
            chunkLogitsF32.resize((size_t) vocab);
            halfToFloatBulk(reinterpret_cast<const fp16_t *>(logitsOut->data.data()) + rowOffset, chunkLogitsF32.data(), vocab);
            lastLogits = chunkLogitsF32.data();
        } else
        {
            lastLogits = reinterpret_cast<const float *>(logitsOut->data.data()) + rowOffset;
        }
        lastFromDecode = false;
        return true;
    };

    // Materialize the chunk bucket's resident cache into the host past buffers: the resident past
    // (every fold applied through the last pass's ranges) plus the LAST chunk's still-pending rows
    // read from the resident present outputs. The host cache then holds the full conversation
    // state for the decode re-seed. Requires the chunk links to be the session's current set.
    auto syncChunkResidentToHost = [&](int lastStart, int lastLen) -> bool {
        const int    newRowsAt    = chunkPresRows - chunkS;
        const size_t presentBytes = (size_t) kvHeads * chunkPresRows * headDim * kvElemBytes;
        for (int l = 0; l < L; ++l)
        {
            for (int part = 0; part < 2; ++part)
            {
                IOTensor &hostPast = inputs[(size_t) (part ? pastVal[l] : pastKey[l])];
                if (hostPast.data.empty())
                {
                    hostPast.data.assign((size_t) ins[(size_t) (part ? pastVal[l] : pastKey[l])].elems * kvElemBytes, 0);
                }
                IOTensor resident;
                if (sess->readResident(hostPast.name, resident) != Status::Ok || resident.data.size() != hostPast.data.size())
                {
                    fprintf(stderr, "[chat] chunk resident cache readback failed for %s\n", hostPast.name.c_str());
                    return false;
                }
                std::memcpy(hostPast.data.data(), resident.data.data(), resident.data.size());
                IOTensor present;
                if (sess->readResident(outs[(size_t) (part ? presVal[l] : presKey[l])].name, present) != Status::Ok || present.data.size() != presentBytes)
                {
                    fprintf(stderr, "[chat] chunk resident present readback failed\n");
                    return false;
                }
                const uint8_t *src = present.data.data();
                uint8_t       *dst = hostPast.data.data();
                for (int h = 0; h < kvHeads; ++h)
                {
                    std::memcpy(dst + ((size_t) h * C + lastStart) * headDim * kvElemBytes, src + ((size_t) h * chunkPresRows + newRowsAt) * headDim * kvElemBytes, (size_t) lastLen * headDim * kvElemBytes);
                }
            }
        }
        return true;
    };

    // One turn's chunked prefill over `turnPrompt`. Returns 1 when at least one chunk ran
    // (consumedOut advanced, host cache materialized, decode links restored), 0 when no chunk fits
    // before the context edge (session state untouched), -1 on failure — p, consumedOut, and the
    // host cache are back at the turn entry (chunk folds never touch host bytes), the links are
    // back on the decode bucket when possible (else the stream drops to the host KV loop), and
    // the caller re-prefills through the whole-window/token-by-token path.
    auto chunkedPrefillTurn = [&](const std::vector<int64_t> &turnPrompt, size_t &consumedOut) -> int {
        const int turnStartP = p;
        {
            const int firstLen = (int) std::min<size_t>(turnPrompt.size(), (size_t) chunkS);
            if (p + firstLen > C)
            {
                return 0; // the token-by-token tail (with its slot clamp) owns the context edge
            }
        }
        auto restoreDecodeLinks = [&]() {
            sess->clearLinks();
            if (!relinkDecode())
            {
                sess->clearLinks();
                fprintf(stderr, "[chat] switching to the host KV loop (decode re-link failed)\n");
                kvLink = false; // the host cache holds the conversation state
            }
        };
        sess->clearLinks();
        if (!linkChunkBucket())
        {
            restoreDecodeLinks();
            return -1;
        }
        int  prevStart = -1, prevLen = 0;
        bool first = true;
        while (consumedOut < turnPrompt.size())
        {
            const int len = (int) std::min<size_t>(turnPrompt.size() - consumedOut, (size_t) chunkS);
            if (p + len > C)
            {
                break;
            }
            if (!runChunk(&turnPrompt[consumedOut], len, prevStart, prevLen, first))
            {
                p           = turnStartP;
                consumedOut = 0;
                restoreDecodeLinks();
                return -1;
            }
            prevStart = p;
            prevLen   = len;
            first     = false;
            p += len;
            consumedOut += (size_t) len;
        }
        if (prevLen > 0 && !syncChunkResidentToHost(prevStart, prevLen))
        {
            // A partial sync only rewrites slots this turn's re-prefill overwrites again; the
            // earlier-turn slots it copied equal the host bytes they replaced.
            p           = turnStartP;
            consumedOut = 0;
            restoreDecodeLinks();
            return -1;
        }
        restoreDecodeLinks();
        return prevLen > 0 ? 1 : 0;
    };

    // --- greedy speculative decode --------------------------------------------------------------
    // One turn's rounds run inside the VERIFICATION bucket, whose KV cache stays engine-resident
    // between rounds exactly as the chunk bucket's does between chunk passes. Round r's link ranges
    // fold round r-1's ACCEPTED rows into their absolute slots at the start of run r; a rejected row
    // appears in no range and so is never copied into the cache, which is the whole of the rollback.
    // The session holds one link set at a time, so a speculative turn switches the links to the
    // verification bucket and back to the decode bucket afterwards (readResident is name-keyed and
    // stays unambiguous that way).
    std::vector<IOTensor> specOutputs;
    std::vector<float>    specLogitsF32;                                      // fp32 copy of an fp16 verification logits block
    int64_t               specRounds = 0, specProposed = 0, specAccepted = 0; // acceptance instrument

    // Link every verification-bucket present output to its past input (ranges re-armed per round).
    auto linkSpecBucket = [&]() -> bool {
        for (int l = 0; l < L; ++l)
        {
            for (int part = 0; part < 2; ++part)
            {
                const std::string &pres = outs[(size_t) (part ? presVal[l] : presKey[l])].name;
                const std::string &past = ins[(size_t) (part ? pastVal[l] : pastKey[l])].name;
                if (sess->linkOutputToInput((size_t) specBucket, pres, past, {}) != Status::Ok)
                {
                    fprintf(stderr, "[chat] spec-verify link failed for %s -> %s (see log)\n", pres.c_str(), past.c_str());
                    return false;
                }
            }
        }
        return true;
    };

    // Materialize the verification bucket's resident cache into the host past buffers: the resident
    // past (every armed fold applied) plus the last round's accepted rows, still pending in the
    // present outputs. Requires the verification links to be the session's current set.
    auto syncSpecResidentToHost = [&](int lastSlot, int lastRows) -> bool {
        const int    newRowsAt    = specPresRows - (int) kSpecVerifyTokens;
        const size_t presentBytes = (size_t) kvHeads * specPresRows * headDim * kvElemBytes;
        for (int l = 0; l < L; ++l)
        {
            for (int part = 0; part < 2; ++part)
            {
                IOTensor &hostPast = inputs[(size_t) (part ? pastVal[l] : pastKey[l])];
                if (hostPast.data.empty())
                {
                    hostPast.data.assign((size_t) ins[(size_t) (part ? pastVal[l] : pastKey[l])].elems * kvElemBytes, 0);
                }
                IOTensor resident;
                if (sess->readResident(hostPast.name, resident) != Status::Ok || resident.data.size() != hostPast.data.size())
                {
                    fprintf(stderr, "[chat] spec-verify resident cache readback failed for %s\n", hostPast.name.c_str());
                    return false;
                }
                std::memcpy(hostPast.data.data(), resident.data.data(), resident.data.size());
                if (lastRows <= 0)
                {
                    continue;
                }
                IOTensor present;
                if (sess->readResident(outs[(size_t) (part ? presVal[l] : presKey[l])].name, present) != Status::Ok || present.data.size() != presentBytes)
                {
                    fprintf(stderr, "[chat] spec-verify resident present readback failed\n");
                    return false;
                }
                const uint8_t *src = present.data.data();
                uint8_t       *dst = hostPast.data.data();
                for (int h = 0; h < kvHeads; ++h)
                {
                    std::memcpy(dst + ((size_t) h * C + lastSlot) * headDim * kvElemBytes, src + ((size_t) h * specPresRows + newRowsAt) * headDim * kvElemBytes, (size_t) lastRows * headDim * kvElemBytes);
                }
            }
        }
        return true;
    };

    // One verification forward over `window` real tokens at absolute position p, with the previous
    // round's accepted rows armed as the fold ranges. Leaves the per-row argmax in `rowArgMax`.
    // `bindPast` re-seeds the resident cache from the host buffers (the turn's first round).
    auto runVerify = [&](const int64_t *windowTokens, int prevSlot, int prevRows, bool bindPast, std::vector<int64_t> &rowArgMax) -> bool {
        const std::vector<LinkRange> ranges = specVerifyFoldRanges(kvHeads, specPresRows, C, headDim, kSpecVerifyTokens, prevSlot, prevRows);
        for (int l = 0; l < L; ++l)
        {
            for (int part = 0; part < 2; ++part)
            {
                const std::string &pres = outs[(size_t) (part ? presVal[l] : presKey[l])].name;
                const std::string &past = ins[(size_t) (part ? pastVal[l] : pastKey[l])].name;
                if (sess->linkOutputToInput((size_t) specBucket, pres, past, ranges) != Status::Ok)
                {
                    fprintf(stderr, "[chat] spec-verify link update failed for %s -> %s (see log)\n", pres.c_str(), past.c_str());
                    return false;
                }
            }
        }
        IOTensor ids, mask, pos;
        ids.name  = inputs[(size_t) idIdx].name;
        ids.dtype = ins[(size_t) idIdx].dtype;
        ids.shape = {1, kSpecVerifyTokens};
        pos.name  = inputs[(size_t) posIdx].name;
        pos.dtype = ins[(size_t) posIdx].dtype;
        pos.shape = {1, kSpecVerifyTokens};
        {
            std::vector<int64_t> posVals((size_t) kSpecVerifyTokens);
            for (int64_t t = 0; t < kSpecVerifyTokens; ++t)
            {
                posVals[(size_t) t] = (int64_t) p + t;
            }
            ids.data.resize((size_t) kSpecVerifyTokens * 8);
            std::memcpy(ids.data.data(), windowTokens, ids.data.size());
            pos.data.resize((size_t) kSpecVerifyTokens * 8);
            std::memcpy(pos.data.data(), posVals.data(), pos.data.size());
        }
        mask.name  = inputs[(size_t) maskIdx].name;
        mask.dtype = ins[(size_t) maskIdx].dtype;
        mask.shape = {1, (int64_t) specMaskLen};
        {
            // Every window column is a real token (a round never pads), so the mask marks the valid
            // past slots plus the whole window; the graph's own causal mask orders the columns.
            std::vector<int64_t> maskVals((size_t) specMaskLen, 0);
            for (int j = 0; j < p && j < C; ++j)
            {
                maskVals[(size_t) j] = 1;
            }
            for (int64_t t = 0; t < kSpecVerifyTokens; ++t)
            {
                maskVals[(size_t) (C + t)] = 1;
            }
            mask.data.resize((size_t) specMaskLen * 8);
            std::memcpy(mask.data.data(), maskVals.data(), mask.data.size());
        }
        std::vector<IOTensor> bound {ids, mask, pos};
        if (bindPast)
        {
            for (int l = 0; l < L; ++l)
            {
                bound.push_back(inputs[(size_t) pastKey[l]]);
                bound.push_back(inputs[(size_t) pastVal[l]]);
            }
        }
        if (sess->run(bound, specOutputs) != Status::Ok)
        {
            fprintf(stderr, "[chat] spec-verify run failed\n");
            return false;
        }
        const IOTensor *logitsOut = nullptr;
        for (const IOTensor &o: specOutputs)
        {
            if (o.name == "logits")
            {
                logitsOut = &o;
            }
        }
        if (!logitsOut || logitsOut->data.size() < (size_t) (kSpecVerifyTokens * vocab) * (specLogitsFp16 ? sizeof(fp16_t) : sizeof(float)))
        {
            fprintf(stderr, "[chat] spec-verify logits missing or short (every window row feeds the acceptance test)\n");
            return false;
        }
        const float *rows = nullptr;
        if (specLogitsFp16)
        {
            specLogitsF32.resize((size_t) (kSpecVerifyTokens * vocab));
            halfToFloatBulk(reinterpret_cast<const fp16_t *>(logitsOut->data.data()), specLogitsF32.data(), kSpecVerifyTokens * vocab);
            rows = specLogitsF32.data();
        } else
        {
            rows = reinterpret_cast<const float *>(logitsOut->data.data());
        }
        rowArgMax.assign((size_t) kSpecVerifyTokens, 0);
        for (int64_t r = 0; r < kSpecVerifyTokens; ++r)
        {
            const float *row  = rows + r * vocab;
            int64_t      best = 0;
            for (int64_t i = 1; i < vocab; ++i)
            {
                if (row[i] > row[(size_t) best])
                {
                    best = i;
                }
            }
            rowArgMax[(size_t) r] = best;
        }
        return true;
    };

    // Decode one turn speculatively, starting from the pending token `next` at position p.
    // Returns 1 when the turn is complete (end-of-stream or the token budget), 0 when it handed the
    // rest of the turn back to the plain loop with `next` still pending and unprinted (the context
    // edge, or a round that could not run), and -1 on a fatal error. `generated` and `emitted` are
    // advanced for every printed token. The engine state on return is the plain loop's: the host
    // cache holds the conversation, the decode bucket's links are restored, and the next decode step
    // re-seeds the resident cache from the host buffers.
    auto specDecodeTurn = [&](int64_t &next, int &generated, int &emitted) -> int {
        if (residentDirty)
        {
            // A linked decode from an earlier turn (or this turn's token-by-token prompt tail)
            // leaves a pending fold at slot p-1; bring it and the resident rows to the host first.
            if (!syncResidentToHost(pendingResidentFold ? kvFoldSlot(p, C) : -1))
            {
                return -1;
            }
            residentDirty = false;
        }
        sess->clearLinks();
        auto restoreDecodeLinks = [&]() {
            sess->clearLinks();
            if (!relinkDecode())
            {
                sess->clearLinks();
                fprintf(stderr, "[chat] switching to the host KV loop (decode re-link failed after speculation)\n");
                kvLink = false; // the host cache holds the conversation state
            }
            reseedCache         = kvLink;
            residentDirty       = false;
            pendingResidentFold = true;
        };
        if (!linkSpecBucket())
        {
            restoreDecodeLinks();
            return 0; // the plain loop owns the turn; `next` was never fed
        }
        int                  prevSlot = -1, prevRows = 0;
        bool                 bindPast = true;
        bool                 ranRound = false;
        int                  outcome  = 0;
        std::vector<int64_t> window((size_t) kSpecVerifyTokens, 0), rowArgMax, proposals((size_t) kSpecDraftTokens, 0);
        while (generated < maxTokens && next != eos)
        {
            // A round writes kSpecVerifyTokens cache rows starting at p, so it needs a whole window
            // inside the compiled context; past that the turn's tail decodes token by token, where
            // the fold-slot clamp is the single-step loop's own.
            if (p + (int) kSpecVerifyTokens > C || !draft->propose(next, p, proposals.data(), (int) kSpecDraftTokens))
            {
                break;
            }
            window[0] = next;
            for (int64_t i = 0; i < kSpecDraftTokens; ++i)
            {
                window[(size_t) i + 1] = proposals[(size_t) i];
            }
            if (!runVerify(window.data(), prevSlot, prevRows, bindPast, rowArgMax))
            {
                break;
            }
            bindPast           = false;
            ranRound           = true;
            const int accepted = specAcceptedDrafts(proposals.data(), rowArgMax.data(), (int) kSpecDraftTokens);
            ++specRounds;
            specProposed += kSpecDraftTokens;
            specAccepted += accepted;
            // The round's committed tokens: the anchor plus the accepted proposals, then the plain
            // loop's own two truncations (the first end-of-stream id, the token budget).
            std::vector<int64_t> committed;
            committed.push_back(next);
            for (int i = 0; i < accepted; ++i)
            {
                committed.push_back(proposals[(size_t) i]);
            }
            const int commitCount = specEmittedCount(committed.data(), (int) committed.size(), eos, maxTokens - generated);
            for (int i = 0; i < commitCount; ++i)
            {
                printf("%lld\n", (long long) committed[(size_t) i]);
                fedTokens.push_back(committed[(size_t) i]);
                ++generated;
                ++emitted;
            }
            fflush(stdout);
            prevSlot = p;
            prevRows = commitCount;
            p += commitCount;
            tmSteps += commitCount; // a round counts as the tokens it committed
            // The draft fed every committed token in this round, so its cache holds the whole
            // conversation prefix again; the rows past it are the rejected proposals, which the next
            // round's rewind masks out and overwrites.
            draftValid = (int) fedTokens.size();
            if (commitCount < (int) committed.size())
            {
                outcome = 1; // end-of-stream or the budget ended the turn inside the round
                break;
            }
            next = rowArgMax[(size_t) accepted]; // the correction, or the bonus token at full acceptance
        }
        if (ranRound && !syncSpecResidentToHost(prevSlot, prevRows))
        {
            restoreDecodeLinks();
            return -1;
        }
        restoreDecodeLinks();
        if (generated >= maxTokens || next == eos)
        {
            outcome = 1;
        }
        return outcome;
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

        if (specActive)
        {
            fedTokens.insert(fedTokens.end(), prompt.begin(), prompt.end()); // every prefill path feeds the whole prompt
        }
        const double tTurnStart       = nowMs(); // turn walls for the --timing-summary line below
        size_t       consumed         = 0;
        bool         chunkRanThisTurn = false;
        // The whole-window bucket wins whenever it covers the prompt: one pass over the weights
        // beats ceil(T / chunkS) passes, and every chunk pass re-reads the whole weight set (on a
        // 0.5B decoder, time to first token is ~520 ms flat for the window pass against ~450 ms per
        // chunk). Chunking earns its keep only past that window, where the alternative is the
        // token-by-token tail — several times slower again. Both paths produce the same stream.
        const bool windowCoversPrompt = prefillBucket >= 0 && (int64_t) prompt.size() <= (int64_t) prefillS;
        if (chunkBucket >= 0 && kvLink && !windowCoversPrompt)
        {
            // Chunked-resident prefill: sync the decode-resident cache to the host once per turn
            // (a linked decode from the previous turn leaves a pending fold at slot p-1), then run
            // the prompt as resident chunk passes. A failure restores the turn-entry state and
            // falls through to the whole-window path below, permanently.
            if (residentDirty)
            {
                if (!syncResidentToHost(pendingResidentFold ? kvFoldSlot(p, C) : -1))
                {
                    return 3;
                }
                residentDirty = false;
            }
            const int chunkStatus = chunkedPrefillTurn(prompt, consumed);
            if (chunkStatus < 0)
            {
                fprintf(stderr, "[chat] chunked prefill unavailable; using the whole-window path\n");
                chunkBucket = -1;
                consumed    = 0;
            } else
            {
                chunkRanThisTurn = true;
                reseedCache      = kvLink && consumed > 0; // first decode step re-seeds the resident cache
            }
        }
        if (!chunkRanThisTurn && prefillBucket >= 0)
        {
            // Whole-window prefill: sync the engine-resident cache to the host once per turn (a
            // linked decode from the previous turn leaves a pending fold at slot p-1), then feed
            // the prompt in prefillS-sized forwards. A remainder past the compiled context falls
            // to the token-by-token loop below.
            if (kvLink && residentDirty)
            {
                if (!syncResidentToHost(pendingResidentFold ? kvFoldSlot(p, C) : -1))
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
        // The prompt is folded and the first token's logits are in hand: this instant is the
        // turn's time-to-first-token (the argmax that turns the logits into an id follows below
        // and is either already GPU-side or a single host vocab scan).
        const double tPromptDone = nowMs();
        int          emitted     = 0; // tokens actually printed this turn (the decode-rate denominator)
        int          generated   = 0;
        // Greedy speculative decode. The draft first catches up on every conversation token it has
        // not seen (this prompt, and anything an earlier turn generated while speculation was down),
        // then the rounds run; whatever they leave — the context edge, a link failure, a draft that
        // cannot reach this position — falls through to the single-step loop below with the pending
        // token still in hand. Speculation and --chain are mutually exclusive by construction.
        if (specActive && kvLink)
        {
            const double tSpec0 = nowMs();
            int64_t      next   = 0;
            if (gpuArgmax && lastFromDecode)
            {
                float bestValue;
                if (sess->readOutputArgMax("logits", next, bestValue) != Status::Ok)
                {
                    fprintf(stderr, "[chat] engine argmax readback failed\n");
                    return 3;
                }
            } else if (lastLogits)
            {
                next = sample(lastLogits); // the prefill pass's host row (greedy scan)
            } else
            {
                fprintf(stderr, "[chat] logits row unavailable for sampling\n");
                return 3;
            }
            tmSampleMs += nowMs() - tSpec0;
            const int caughtUp = (int) fedTokens.size();
            if (draftValid > caughtUp)
            {
                draftValid = caughtUp; // a trimmed turn left the draft ahead of the conversation
            }
            bool draftReady = draftValid == caughtUp;
            if (!draftReady)
            {
                draft->rewind(draftValid);
                draftReady = draft->consume(fedTokens.data() + draftValid, caughtUp - draftValid);
                draftValid = draftReady ? caughtUp : draftValid;
                if (!draftReady)
                {
                    fprintf(stderr, "[chat] draft model could not follow the conversation to %d tokens (its context is %d); decoding this turn plainly\n", caughtUp,
                            draft->cacheSlots());
                }
            }
            const int specOutcome = draftReady ? specDecodeTurn(next, generated, emitted) : 0;
            if (specOutcome < 0)
            {
                return 3;
            }
            if (specOutcome == 1)
            {
                generated = maxTokens; // the turn is complete; skip the single-step loop
            } else if (generated < maxTokens && next != eos)
            {
                // The pending token was never printed or fed: hand it to the single-step loop, which
                // then continues from the decode bucket's own argmax.
                printf("%lld\n", (long long) next);
                fflush(stdout);
                ++generated;
                ++emitted;
                if (specActive)
                {
                    fedTokens.push_back(next);
                }
                if (!step(next))
                {
                    return 3;
                }
                ++p;
            } else
            {
                generated = maxTokens;
            }
        }
        if (chainActive && kvLink)
        {
            // Device-chained greedy decode: the same stream as the single-step loop below, one
            // submit per chain. `next` is always the newest printed-or-pending token that has NOT
            // been fed yet; each chain feeds it plus the chain's own on-GPU intermediate ids.
            const double tSample0 = nowMs();
            int64_t      next     = 0;
            if (gpuArgmax && lastFromDecode)
            {
                float bestValue;
                if (sess->readOutputArgMax("logits", next, bestValue) != Status::Ok)
                {
                    fprintf(stderr, "[chat] engine argmax readback failed\n");
                    return 3;
                }
            } else if (lastLogits)
            {
                next = sample(lastLogits); // the prefill pass's host row (greedy scan)
            } else
            {
                fprintf(stderr, "[chat] logits row unavailable for sampling\n");
                return 3;
            }
            tmSampleMs += nowMs() - tSample0;
            bool fellBack = false;
            while (next != eos && generated < maxTokens)
            {
                printf("%lld\n", (long long) next);
                fflush(stdout);
                ++generated;
                ++emitted;
                if (generated >= maxTokens)
                {
                    // The single-step loop also feeds the last printed token (its argmax is never
                    // read), leaving the identical engine state for the next turn.
                    const int fedLast = chainStep(next, 1);
                    if (fedLast == 0 || (fedLast < 0 && !step(next)))
                    {
                        return 3;
                    }
                    ++p;
                    break;
                }
                // Clamp the chain at the context edge: iterations stay below the fold-slot clamp,
                // so a mid-chain EOS's discarded overshoot can never overwrite a live clamped
                // slot; past the edge each chain is one iteration (single-step semantics).
                const int64_t window   = p < C ? std::min<int64_t>((int64_t) chainSteps, (int64_t) C - p) : 1;
                const int     steps    = (int) std::min<int64_t>(window, (int64_t) (maxTokens - generated));
                const int     ranChain = chainStep(next, steps);
                if (ranChain == 0)
                {
                    return 3;
                }
                if (ranChain < 0)
                {
                    fellBack = true; // `next` is printed but was never fed
                    break;
                }
                const double         tIds0 = nowMs();
                std::vector<int64_t> chainIds((size_t) steps, 0);
                int                  firstEosAt = steps;
                for (int i = 0; i < steps; ++i)
                {
                    float bestValue;
                    if (sess->readOutputArgMax("logits", i, chainIds[(size_t) i], bestValue) != Status::Ok)
                    {
                        fprintf(stderr, "[chat] engine argmax readback failed\n");
                        return 3;
                    }
                    if (firstEosAt == steps && chainIds[(size_t) i] == eos)
                    {
                        firstEosAt = i;
                    }
                }
                tmSampleMs += nowMs() - tIds0;
                if (firstEosAt < steps)
                {
                    // EOS inside the chain: print the ids before it and roll the position back to
                    // the single-step loop's end state; the overshoot iterations never print or
                    // count. Iteration firstEosAt's pending row was already folded on-device by
                    // the first overshoot iteration — unless EOS was the chain's last id, which
                    // leaves the standard pending fold.
                    for (int i = 0; i < firstEosAt; ++i)
                    {
                        printf("%lld\n", (long long) chainIds[(size_t) i]);
                        ++generated;
                        ++emitted;
                    }
                    fflush(stdout);
                    p += firstEosAt + 1;
                    pendingResidentFold = firstEosAt == steps - 1;
                    break;
                }
                for (int i = 0; i + 1 < steps; ++i)
                {
                    printf("%lld\n", (long long) chainIds[(size_t) i]);
                    ++generated;
                    ++emitted;
                }
                fflush(stdout);
                p += steps;
                next = chainIds[(size_t) steps - 1]; // printed by the next pass, fed by the next chain
            }
            if (fellBack)
            {
                // Mid-stream link failure: feed the printed-but-unfed token on the host loop, then
                // continue single-step below.
                if (!step(next))
                {
                    return 3;
                }
                ++p;
            } else
            {
                generated = maxTokens; // the turn is complete; skip the single-step loop
            }
        }
        for (int n = generated; n < maxTokens; ++n)
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
            ++emitted;
            if (specActive)
            {
                fedTokens.push_back(next);
            }
            if (!step(next))
            {
                return 3;
            }
            ++p;
        }
        if (cfg.timing || cfg.timingSummary)
        {
            // Turn walls: time-to-first-token (prompt fold through the first token's logits) and
            // the mean decode step over the tokens generated after it.
            const double turnEndMs = nowMs();
            fprintf(stderr, "[chat] turn: prompt=%zu tok ttft=%.2fms generated=%d decode=%.3fms/tok\n", prompt.size(), tPromptDone - tTurnStart, emitted, emitted > 1 ? (turnEndMs - tPromptDone) / (emitted - 1) : 0.0);
        }
        if (specActive && specRounds > 0)
        {
            // The acceptance rate and the tokens committed per target forward: the two numbers that
            // decide whether speculation pays on this model pair, reported every turn.
            fprintf(stderr, "[chat] speculation: %lld rounds, %lld/%lld drafts accepted (%.1f%%), %.2f tokens per target forward\n", (long long) specRounds, (long long) specAccepted, (long long) specProposed, specProposed > 0 ? 100.0 * (double) specAccepted / (double) specProposed : 0.0, (double) (specRounds + specAccepted) / (double) specRounds);
            specRounds = specProposed = specAccepted = 0;
        }
        if ((cfg.timing || cfg.timingSummary) && tmSteps > 0)
        {
            fprintf(stderr, "[chat] step phases avg over %d step(s): prep=%.3fms links=%.3fms run=%.3fms sample=%.3fms\n", tmSteps, tmPrepMs / tmSteps, tmLinkMs / tmSteps, tmRunMs / tmSteps, tmSampleMs / tmSteps);
            tmPrepMs = tmLinkMs = tmRunMs = tmSampleMs = 0;
            tmSteps                                    = 0;
        }
        printf("END\n");
        fflush(stdout);
    }
    return 0;
}
