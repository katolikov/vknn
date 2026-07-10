// Device-side GPU loop for a vision-language decoder shipped as ONE multi-graph .vxm: a vision
// encoder bucket (pixel_values -> image embeddings), token-embedding buckets, and a text-decoder
// graph at a prefill shape (S tokens in one pass) and a decode shape (S=1), all dispatched by bound
// input names+shapes over one shared weight pool.
//
// Reads simple commands on stdin, one per line:
//   i <path>          load a raw fp32 [1,3,IMG,IMG] pixel file and run the vision bucket; the
//                     resulting image embeddings splice into the next prompt at its image tokens
//   <id id id ...>    prompt token ids for one turn: token rows come from the embedding bucket,
//                     each image-token row is replaced by the next image-embedding row, the whole
//                     prompt prefills in one pass, then greedy/sampled decode streams token ids to
//                     stdout (one per line) and "END" when the turn finishes
// The KV cache and the absolute position persist across turns; a host front-end (vlm_host.py)
// supplies token ids and detokenizes the stream.
//
//   vknn_vlm model.vxm [--backend vulkan|cpu] [--precision low|normal|high] [--max-tokens N]
//            [--temp T] [--top-k K] [--top-p P] [--eos ID] [--image-token ID] [--seed S]
//            [--no-kv-link]
//
// During decode the KV cache is ENGINE-RESIDENT on the S=1 decoder bucket by default
// (Session::linkOutputToInput): each step binds only embeds/mask/positions and the engine folds the
// new row in place. The prefill bucket keeps the host cache flow (once per turn), so at each turn
// boundary the device state materializes back into the host cache via readResident(). --no-kv-link
// keeps the host loop everywhere; both paths produce the same token stream.
//
// The decoder consumes a host-built 4-D ADDITIVE attention mask. Its fill value is -1e4, never
// -FLT_MAX/-65504: the mask crosses the fp16 boundary, where -FLT_MAX overflows to -inf and
// 0*inf = NaN poisons the softmax, and -65504 overflows as soon as a score adds to it. exp(-1e4)
// underflows to exactly 0 in fp16, so masked positions contribute nothing and no inf ever exists.
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

namespace {

    constexpr float kMaskFill = -1e4f; // fp16-safe additive mask fill (see file header)

    // Value after option `key` (scanned from arg 2, past the model path), or `fallback`.
    const char *argValue(int argc, char **argv, const char *key, const char *fallback) noexcept {
        for (int i = 2; i < argc - 1; ++i)
        {
            if (!strcmp(argv[i], key))
            {
                return argv[i + 1];
            }
        }
        return fallback;
    }

    int indexOfName(const std::vector<IOInfo> &infos, const std::string &name) {
        for (size_t i = 0; i < infos.size(); ++i)
        {
            if (infos[i].name == name)
            {
                return (int) i;
            }
        }
        return -1;
    }

    // Value statistics for --debug-stats: pinpoints WHICH pipeline stage first produces NaN/inf or
    // an implausible range when a model misbehaves, without any extra tooling on the device.
    void printStats(const char *label, const float *data, size_t count) {
        size_t nanCount = 0, infCount = 0;
        float  lo = 0, hi = 0;
        bool   seeded = false;
        for (size_t i = 0; i < count; ++i)
        {
            const float v = data[i];
            if (std::isnan(v))
            {
                ++nanCount;
                continue;
            }
            if (std::isinf(v))
            {
                ++infCount;
                continue;
            }
            if (!seeded)
            {
                lo = hi = v;
                seeded  = true;
            } else
            {
                lo = std::min(lo, v);
                hi = std::max(hi, v);
            }
        }
        fprintf(stderr, "[vlm][stats] %-16s n=%zu nan=%zu inf=%zu range=[%.4g, %.4g]\n", label, count, nanCount, infCount, lo, hi);
    }

} // namespace

int main(int argc, char **argv) {
    if (argc < 2)
    {
        fprintf(stderr,
                "usage: %s model.vxm [--backend vulkan|cpu] [--precision low|normal|high]\n"
                "        [--max-tokens N] [--temp T] [--top-k K] [--top-p P] [--eos ID]\n"
                "        [--image-token ID] [--seed S]\n",
                argv[0]);
        return 1;
    }
    Config cfg;
    cfg.backend                = backendFromStr(argValue(argc, argv, "--backend", "vulkan"));
    cfg.precision              = precisionFromStr(argValue(argc, argv, "--precision", "low"));
    cfg.freeWeightsAfterUpload = true;
    const int     maxTokens    = atoi(argValue(argc, argv, "--max-tokens", "256"));
    const float   temperature  = (float) atof(argValue(argc, argv, "--temp", "0"));
    const int     topK         = atoi(argValue(argc, argv, "--top-k", "0"));
    const float   topP         = (float) atof(argValue(argc, argv, "--top-p", "1"));
    const int64_t eosToken     = atoll(argValue(argc, argv, "--eos", "49279"));
    const int64_t imageToken   = atoll(argValue(argc, argv, "--image-token", "49190"));
    bool          debugStats   = false;
    bool          kvLink       = true;
    for (int i = 2; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--debug-stats"))
        {
            debugStats = true;
        }
        if (!strcmp(argv[i], "--no-kv-link"))
        {
            kvLink = false;
        }
        if (!strcmp(argv[i], "--timing"))
        {
            cfg.timing = true; // per-run pack/submit/unpack walls
        }
        if (!strcmp(argv[i], "--no-matmul-view-fold"))
        {
            cfg.setHint(Hint::MatMulViewFold, (int) Mode::Off);
        }
    }
    std::mt19937 rng((unsigned) atoi(argValue(argc, argv, "--seed", "1234")));

    auto session = Runtime::load(argv[1], cfg);
    if (!session)
    {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }

    // --- discover the bucket roles by input names/shapes ---------------------------------------
    // vision:  single input "pixel_values"  [1,3,IMG,IMG]
    // embed:   single input "input_ids"     [1,S] at S = prefill length and S = 1
    // decoder: "inputs_embeds" [1,S,H] + "attention_mask" [1,1,S,C+S] + "position_ids" [1,S] +
    //          past_key_values.<l>.{key,value} [1,KV,C,HD]; S = prefill length and S = 1
    int          prefillWindow = 0; // sequence length of the widest decoder bucket
    int          visionBucket = -1, embedPrefillBucket = -1, embedDecodeBucket = -1;
    int          decoderPrefillBucket = -1, decoderDecodeBucket = -1;
    const size_t bucketTotal = session->bucketCount();
    for (size_t bucket = 0; bucket < bucketTotal; ++bucket)
    {
        std::vector<IOInfo> inputs = session->inputInfo(bucket);
        if (indexOfName(inputs, "pixel_values") >= 0)
        {
            visionBucket = (int) bucket;
        } else if (inputs.size() == 1 && inputs[0].name == "input_ids")
        {
            (inputs[0].shape.back() == 1 ? embedDecodeBucket : embedPrefillBucket) = (int) bucket;
        } else if (indexOfName(inputs, "inputs_embeds") >= 0)
        {
            const int     embedsAt = indexOfName(inputs, "inputs_embeds");
            const int64_t seqLen   = inputs[(size_t) embedsAt].shape[1];
            if (seqLen == 1)
            {
                decoderDecodeBucket = (int) bucket;
            } else
            {
                decoderPrefillBucket = (int) bucket;
                prefillWindow        = (int) seqLen;
            }
        }
    }
    if (visionBucket < 0 || embedPrefillBucket < 0 || embedDecodeBucket < 0 || decoderPrefillBucket < 0 || decoderDecodeBucket < 0)
    {
        fprintf(stderr, "model is not a vision-decoder multi-graph .vxm (need pixel_values / input_ids[1,S]+[1,1] / inputs_embeds prefill+decode buckets; have %zu buckets)\n", bucketTotal);
        return 2;
    }

    const std::vector<IOInfo> visionInputInfo   = session->inputInfo((size_t) visionBucket);
    const std::vector<IOInfo> visionOutputInfo  = session->outputInfo((size_t) visionBucket);
    const std::vector<IOInfo> decoderInputInfo  = session->inputInfo((size_t) decoderPrefillBucket);
    const std::vector<IOInfo> decoderOutputInfo = session->outputInfo((size_t) decoderPrefillBucket);

    // Decoder geometry from the past/present tensors and logits.
    std::vector<int> pastKeyInputIdx, pastValueInputIdx;
    for (int layer = 0;; ++layer)
    {
        char keyName[64], valueName[64];
        snprintf(keyName, sizeof keyName, "past_key_values.%d.key", layer);
        snprintf(valueName, sizeof valueName, "past_key_values.%d.value", layer);
        const int keyAt   = indexOfName(decoderInputInfo, keyName);
        const int valueAt = indexOfName(decoderInputInfo, valueName);
        if (keyAt < 0 || valueAt < 0)
        {
            break;
        }
        pastKeyInputIdx.push_back(keyAt);
        pastValueInputIdx.push_back(valueAt);
    }
    const int numLayers        = (int) pastKeyInputIdx.size();
    const int embedsInputIdx   = indexOfName(decoderInputInfo, "inputs_embeds");
    const int maskInputIdx     = indexOfName(decoderInputInfo, "attention_mask");
    const int positionInputIdx = indexOfName(decoderInputInfo, "position_ids");
    const int logitsOutputIdx  = indexOfName(decoderOutputInfo, "logits");
    if (numLayers == 0 || embedsInputIdx < 0 || maskInputIdx < 0 || positionInputIdx < 0 || logitsOutputIdx < 0)
    {
        fprintf(stderr, "decoder bucket misses inputs_embeds/attention_mask/position_ids/logits/past\n");
        return 2;
    }
    const Shape  &pastShape      = decoderInputInfo[(size_t) pastKeyInputIdx[0]].shape; // [1, KV, C, HD]
    const int     kvHeads        = (int) pastShape[1];
    const int     cacheSlots     = (int) pastShape[2];
    const int     headDim        = (int) pastShape[3];
    const int     hiddenDim      = (int) decoderInputInfo[(size_t) embedsInputIdx].shape[2];
    const int64_t vocabSize      = decoderOutputInfo[(size_t) logitsOutputIdx].shape.back();
    const Shape  &visionOutShape = visionOutputInfo[0].shape; // [1, imageRowCount, hiddenDim]
    const int     imageRowCount  = (int) visionOutShape[1];
    // The vision and embedding buckets feed rows straight into the decoder's inputs_embeds buffer;
    // their hidden dim must equal the decoder's or the row copies would over-read their outputs.
    const std::vector<IOInfo> embedOutputInfo = session->outputInfo((size_t) embedPrefillBucket);
    if (visionOutShape.back() != hiddenDim || embedOutputInfo.empty() || embedOutputInfo[0].shape.back() != hiddenDim)
    {
        fprintf(stderr, "[vlm] hidden-dim mismatch: vision %lld / embed %lld / decoder %d\n", (long long) visionOutShape.back(),
                (long long) (embedOutputInfo.empty() ? -1 : embedOutputInfo[0].shape.back()), hiddenDim);
        return 2;
    }
    fprintf(stderr, "[vlm] %s: layers=%d kv_heads=%d C=%d head_dim=%d H=%d vocab=%lld prefillS=%d imageRows=%d\n", argv[1], numLayers, kvHeads, cacheSlots, headDim, hiddenDim, (long long) vocabSize, prefillWindow, imageRowCount);

    // --- persistent boundary tensors -------------------------------------------------------------
    // One input vector serves BOTH decoder buckets: the past entries hold the KV cache across the
    // whole conversation (fp32 host boundary), and the three small entries are re-shaped per call
    // (S=prefill for a turn's first pass, S=1 for every decode step); run() dispatches to the bucket
    // matching the bound shapes. Copying the cache into a per-call vector would move ~C*KV*HD*2L*4
    // bytes per token, so the cache entries ARE the run inputs.
    std::vector<IOTensor> decoderInputs(decoderInputInfo.size());
    for (size_t i = 0; i < decoderInputInfo.size(); ++i)
    {
        decoderInputs[i].name  = decoderInputInfo[i].name;
        decoderInputs[i].shape = decoderInputInfo[i].shape;
        decoderInputs[i].dtype = decoderInputInfo[i].dtype;
        decoderInputs[i].data.assign((size_t) decoderInputInfo[i].elems * dtypeSize(decoderInputInfo[i].dtype), 0);
    }
    auto resizeDecoderInput = [&](int inputIdx, const Shape &shape, DType dtype) {
        decoderInputs[(size_t) inputIdx].shape = shape;
        decoderInputs[(size_t) inputIdx].dtype = dtype;
        decoderInputs[(size_t) inputIdx].data.assign((size_t) numElements(shape) * dtypeSize(dtype), 0);
    };

    std::vector<IOTensor> embedInputs(1), visionInputs(1);
    embedInputs[0].name   = "input_ids";
    embedInputs[0].dtype  = DType::Int64;
    visionInputs[0].name  = "pixel_values";
    visionInputs[0].shape = visionInputInfo[0].shape;
    visionInputs[0].dtype = DType::Float32;

    std::vector<float> imageEmbeddings; // imageRowCount*hiddenDim floats once an image is loaded
    int                absolutePos = 0; // token position across the whole conversation

    // Engine-resident KV cache on the DECODE bucket: link every present output to its past input
    // (empty ranges; the per-step fold slot arrives before each run). The prefill bucket keeps the
    // host cache, so turn boundaries materialize the device state back into decoderInputs.
    auto pastNameOf = [](int layer, int part, char (&buf)[64]) {
        snprintf(buf, sizeof buf, part ? "past_key_values.%d.value" : "past_key_values.%d.key", layer);
    };
    auto presentNameOf = [](int layer, int part, char (&buf)[64]) {
        snprintf(buf, sizeof buf, part ? "present.%d.value" : "present.%d.key", layer);
    };
    // Present rows from the DECODE bucket's own present output shape (decoderOutputInfo describes
    // the PREFILL bucket, whose present carries prefillWindow rows). This decoder's present holds
    // only the produced rows — [1,KV,1,HD] at S=1 — unlike a cache-concat decoder's [1,KV,C+1,HD];
    // the fold source is always the LAST present row, so both conventions drive the same code.
    int presRowsDecode = 0;
    {
        const std::vector<IOInfo> decodeOut = session->outputInfo((size_t) decoderDecodeBucket);
        const int                 presAt    = indexOfName(decodeOut, "present.0.key");
        if (presAt >= 0 && decodeOut[(size_t) presAt].shape.size() == 4)
        {
            presRowsDecode = (int) decodeOut[(size_t) presAt].shape[2];
        }
    }
    bool decodeLinked = false;
    if (kvLink && presRowsDecode > 0)
    {
        decodeLinked = true;
        for (int layer = 0; layer < numLayers && decodeLinked; ++layer)
        {
            for (int part = 0; part < 2 && decodeLinked; ++part)
            {
                char pastBuf[64], presentBuf[64];
                pastNameOf(layer, part, pastBuf);
                presentNameOf(layer, part, presentBuf);
                if (session->linkOutputToInput((size_t) decoderDecodeBucket, presentBuf, pastBuf, {}) != Status::Ok)
                {
                    fprintf(stderr, "[vlm] KV link setup failed (layer %d); using the host cache loop\n", layer);
                    session->clearLinks();
                    decodeLinked = false;
                }
            }
        }
        if (decodeLinked)
        {
            fprintf(stderr, "[vlm] decode KV cache engine-resident (%d links, present rows %d)\n", 2 * numLayers, presRowsDecode);
        }
    } else if (kvLink)
    { fprintf(stderr, "[vlm] decode-bucket present outputs are missing or not [1,KV,rows,HD]; using the host cache loop\n"); }
    bool cacheOnDevice      = false; // the decode bucket's resident past is ahead of decoderInputs
    bool rebindPastNextStep = false; // next decode step re-seeds the device cache from decoderInputs
    int  pendingFoldSlot    = -1;    // cache slot of the fold the next decode run applies

    // Update every decode link: fold the previous run's present row (the last row of the decode
    // bucket's present) into cache slot `slot`, or clear the pending fold when `slot` < 0.
    auto setDecodeFoldSlot = [&](int64_t slot) {
        const std::vector<LinkRange> ranges = kvFoldRanges(kvHeads, presRowsDecode, cacheSlots, headDim, slot);
        for (int layer = 0; layer < numLayers; ++layer)
        {
            for (int part = 0; part < 2; ++part)
            {
                char pastBuf[64], presentBuf[64];
                pastNameOf(layer, part, pastBuf);
                presentNameOf(layer, part, presentBuf);
                const Status st = session->linkOutputToInput((size_t) decoderDecodeBucket, presentBuf, pastBuf, ranges);
                if (st != Status::Ok)
                {
                    fprintf(stderr, "[vlm] KV link update failed for %s -> %s at slot %lld: %s (engine log has the reason)\n", presentBuf, pastBuf, (long long) slot, statusStr(st));
                    return false;
                }
            }
        }
        return true;
    };

    // Copy the decode bucket's device-resident cache (plus the pending fold) back into the host
    // decoderInputs past buffers, so the prefill bucket sees the full conversation state.
    auto materializeDeviceCache = [&]() {
        for (int layer = 0; layer < numLayers; ++layer)
        {
            for (int part = 0; part < 2; ++part)
            {
                char pastBuf[64], presentBuf[64];
                pastNameOf(layer, part, pastBuf);
                presentNameOf(layer, part, presentBuf);
                IOTensor resident;
                if (session->readResident(pastBuf, resident) != Status::Ok)
                {
                    return false;
                }
                IOTensor &hostPast = decoderInputs[(size_t) (part ? pastValueInputIdx[layer] : pastKeyInputIdx[layer])];
                if (resident.data.size() != hostPast.data.size())
                {
                    fprintf(stderr, "[vlm] resident cache size mismatch for %s\n", pastBuf);
                    return false;
                }
                std::memcpy(hostPast.data.data(), resident.data.data(), resident.data.size());
                if (pendingFoldSlot >= 0)
                {
                    IOTensor present;
                    if (session->readResident(presentBuf, present) != Status::Ok)
                    {
                        return false;
                    }
                    const float *src   = reinterpret_cast<const float *>(present.data.data());
                    float       *cache = reinterpret_cast<float *>(hostPast.data.data());
                    for (int head = 0; head < kvHeads; ++head)
                    {
                        const float *srcRow   = src + ((size_t) head * presRowsDecode + (presRowsDecode - 1)) * headDim;
                        float       *cacheRow = cache + ((size_t) head * cacheSlots + pendingFoldSlot) * headDim;
                        std::memcpy(cacheRow, srcRow, (size_t) headDim * sizeof(float));
                    }
                }
            }
        }
        pendingFoldSlot = -1;
        return true;
    };

    std::vector<IOTensor> outputs;
    auto                  findOutput = [&](const std::string &name) -> const IOTensor                  *{
        for (const IOTensor &out: outputs)
        {
            if (out.name == name)
            {
                return &out;
            }
        }
        return nullptr;
    };

    // Copy present rows [1,KV,presentRows,HD] (rows 0..copyRows-1) into cache slots
    // firstSlot..firstSlot+copyRows-1 of every layer's persistent past tensor.
    auto foldPresentIntoCache = [&](int presentRows, int copyRows, int firstSlot) {
        for (int layer = 0; layer < numLayers; ++layer)
        {
            char keyName[64], valueName[64];
            snprintf(keyName, sizeof keyName, "present.%d.key", layer);
            snprintf(valueName, sizeof valueName, "present.%d.value", layer);
            for (int part = 0; part < 2; ++part)
            {
                const IOTensor *present = findOutput(part ? valueName : keyName);
                if (!present)
                {
                    return false;
                }
                const float *src   = present->f32();
                float       *cache = reinterpret_cast<float *>(decoderInputs[(size_t) (part ? pastValueInputIdx[layer] : pastKeyInputIdx[layer])].data.data());
                for (int head = 0; head < kvHeads; ++head)
                {
                    for (int row = 0; row < copyRows; ++row)
                    {
                        const float *srcRow   = src + ((size_t) head * presentRows + row) * headDim;
                        float       *cacheRow = cache + ((size_t) head * cacheSlots + firstSlot + row) * headDim;
                        std::memcpy(cacheRow, srcRow, (size_t) headDim * sizeof(float));
                    }
                }
            }
        }
        return true;
    };

    // Pick the next token id from a logits row (greedy at temperature<=0, else temp+top-k+top-p).
    auto sampleNextToken = [&](const float *logits) -> int64_t {
        if (temperature <= 0.0f)
        {
            int64_t best      = 0;
            float   bestValue = logits[0];
            for (int64_t i = 1; i < vocabSize; ++i)
            {
                if (logits[i] > bestValue)
                {
                    bestValue = logits[i];
                    best      = i;
                }
            }
            return best;
        }
        std::vector<std::pair<float, int64_t>> scored((size_t) vocabSize);
        for (int64_t i = 0; i < vocabSize; ++i)
        {
            scored[(size_t) i] = {logits[i] / temperature, i};
        }
        const int keepCount = topK > 0 && topK < (int) vocabSize ? topK : (int) vocabSize;
        std::partial_sort(scored.begin(), scored.begin() + keepCount, scored.end(), [](const auto &a, const auto &b) {
            return a.first > b.first;
        });
        scored.resize((size_t) keepCount);
        float maxScore = scored[0].first, expSum = 0.0f;
        for (auto &entry: scored)
        {
            entry.first = std::exp(entry.first - maxScore);
            expSum += entry.first;
        }
        float  cumulative = 0.0f;
        size_t nucleus    = scored.size();
        for (size_t i = 0; i < scored.size(); ++i)
        {
            cumulative += scored[i].first / expSum;
            if (cumulative >= topP)
            {
                nucleus = i + 1;
                break;
            }
        }
        float draw = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) * (expSum * (cumulative > 0 ? cumulative : 1.0f));
        float acc  = 0.0f;
        for (size_t i = 0; i < nucleus; ++i)
        {
            acc += scored[i].first;
            if (acc >= draw)
            {
                return scored[i].second;
            }
        }
        return scored[nucleus - 1].second;
    };

    // One decode step: token id -> embedding bucket -> decoder S=1 bucket -> logits row.
    auto decodeOneToken = [&](int64_t tokenId) -> const float * {
        embedInputs[0].shape = {1, 1};
        embedInputs[0].data.resize(sizeof(int64_t));
        std::memcpy(embedInputs[0].data.data(), &tokenId, sizeof(int64_t));
        if (session->run(embedInputs, outputs) != Status::Ok)
        {
            return nullptr;
        }
        const IOTensor *embeds = findOutput("inputs_embeds");
        if (!embeds)
        {
            return nullptr;
        }
        resizeDecoderInput(embedsInputIdx, {1, 1, hiddenDim}, DType::Float32);
        std::memcpy(decoderInputs[(size_t) embedsInputIdx].data.data(), embeds->data.data(), (size_t) hiddenDim * sizeof(float));

        // The token at absolutePos sees the populated cache slots plus itself (appended at column
        // cacheSlots by the in-graph concat).
        resizeDecoderInput(maskInputIdx, {1, 1, 1, (int64_t) cacheSlots + 1}, DType::Float32);
        float *mask = reinterpret_cast<float *>(decoderInputs[(size_t) maskInputIdx].data.data());
        for (int col = 0; col < cacheSlots + 1; ++col)
        {
            mask[col] = (col < absolutePos || col == cacheSlots) ? 0.0f : kMaskFill;
        }
        resizeDecoderInput(positionInputIdx, {1, 1}, DType::Int64);
        const int64_t position = absolutePos;
        std::memcpy(decoderInputs[(size_t) positionInputIdx].data.data(), &position, sizeof(int64_t));

        const int slot      = absolutePos < cacheSlots ? absolutePos : cacheSlots - 1;
        bool      ranLinked = false;
        if (decodeLinked)
        {
            // The engine folds the previous run's row into pendingFoldSlot at the start of this run.
            // The first decode step of a turn re-seeds the device cache from the host past buffers
            // (which prefill just updated) and clears any pending fold.
            const bool rebind = rebindPastNextStep || !cacheOnDevice;
            if (setDecodeFoldSlot(rebind ? -1 : pendingFoldSlot))
            {
                std::vector<IOTensor> bound {decoderInputs[(size_t) embedsInputIdx], decoderInputs[(size_t) maskInputIdx], decoderInputs[(size_t) positionInputIdx]};
                if (rebind)
                {
                    for (int layer = 0; layer < numLayers; ++layer)
                    {
                        bound.push_back(decoderInputs[(size_t) pastKeyInputIdx[layer]]);
                        bound.push_back(decoderInputs[(size_t) pastValueInputIdx[layer]]);
                    }
                }
                if (session->run(bound, outputs) != Status::Ok)
                {
                    return (const float *) nullptr;
                }
                rebindPastNextStep = false;
                cacheOnDevice      = true;
                pendingFoldSlot    = slot;
                ranLinked          = true;
            } else
            {
                // A mid-stream link failure: bring the engine-resident cache (device rows + the
                // pending fold) back into the host past buffers, drop the links, and continue THIS
                // and every later step on the host cache loop — same tokens, no lost state.
                fprintf(stderr, "[vlm] switching to the host KV loop at p=%d (resyncing the cache from the engine)\n", absolutePos);
                if (cacheOnDevice && !materializeDeviceCache())
                {
                    fprintf(stderr, "[vlm] host cache resync failed; cannot continue\n");
                    return (const float *) nullptr;
                }
                session->clearLinks();
                decodeLinked       = false;
                cacheOnDevice      = false;
                rebindPastNextStep = false;
            }
        }
        if (!ranLinked)
        {
            if (session->run(decoderInputs, outputs) != Status::Ok)
            {
                return nullptr;
            }
            if (!foldPresentIntoCache(/*presentRows=*/1, /*copyRows=*/1, slot))
            {
                return nullptr;
            }
        }
        const IOTensor *logitsOut = findOutput("logits");
        return logitsOut ? logitsOut->f32() : nullptr;
    };

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
        {
            continue;
        }
        if (line[0] == 'i' && line.size() > 2 && line[1] == ' ')
        {
            // Load raw fp32 pixels and run the vision bucket; embeddings wait for the next prompt.
            const std::string path = line.substr(2);
            FILE             *file = fopen(path.c_str(), "rb");
            if (!file)
            {
                fprintf(stderr, "[vlm] cannot open %s\n", path.c_str());
                printf("ERR\n");
                fflush(stdout);
                continue;
            }
            visionInputs[0].data.resize((size_t) numElements(visionInputs[0].shape) * sizeof(float));
            const size_t wantBytes = visionInputs[0].data.size();
            const size_t gotBytes  = fread(visionInputs[0].data.data(), 1, wantBytes, file);
            fclose(file);
            if (gotBytes != wantBytes)
            {
                fprintf(stderr, "[vlm] %s: %zu bytes, want %zu\n", path.c_str(), gotBytes, wantBytes);
                printf("ERR\n");
                fflush(stdout);
                continue;
            }
            if (session->run(visionInputs, outputs) != Status::Ok || !findOutput(visionOutputInfo[0].name))
            {
                fprintf(stderr, "[vlm] vision run failed\n");
                printf("ERR\n");
                fflush(stdout);
                continue;
            }
            const IOTensor *visionOut = findOutput(visionOutputInfo[0].name);
            imageEmbeddings.assign(visionOut->f32(), visionOut->f32() + (size_t) imageRowCount * hiddenDim);
            if (debugStats)
            {
                printStats("vision-out", imageEmbeddings.data(), imageEmbeddings.size());
            }
            printf("IMG_OK\n");
            fflush(stdout);
            continue;
        }

        // Prompt turn: parse ids, embed the whole (padded) prompt, splice image rows, prefill.
        std::istringstream   parser(line);
        std::vector<int64_t> prompt;
        int64_t              tokenId;
        while (parser >> tokenId)
        {
            prompt.push_back(tokenId);
        }
        if (prompt.empty())
        {
            continue;
        }
        const int promptLen = (int) prompt.size();
        if (promptLen > prefillWindow)
        {
            fprintf(stderr, "[vlm] prompt %d tokens > prefill window %d\n", promptLen, prefillWindow);
            printf("END\n");
            fflush(stdout);
            continue;
        }
        if (absolutePos + promptLen + 1 > cacheSlots)
        {
            fprintf(stderr, "[vlm] context full (%d + %d > %d)\n", absolutePos, promptLen, cacheSlots);
            printf("END\n");
            fflush(stdout);
            continue;
        }
        // A turn boundary: the decode loop left the cache device-resident; bring it back to the host
        // buffers (with the pending fold applied) for the prefill bucket, and mark the next decode
        // step to re-seed the device cache.
        if (decodeLinked && cacheOnDevice)
        {
            if (!materializeDeviceCache())
            {
                fprintf(stderr, "[vlm] device cache materialization failed\n");
                return 3;
            }
            cacheOnDevice      = false;
            rebindPastNextStep = true;
        }

        std::vector<int64_t> padded(prompt);
        padded.resize((size_t) prefillWindow, eosToken); // pad rows are masked out and never folded
        embedInputs[0].shape = {1, (int64_t) prefillWindow};
        embedInputs[0].data.resize((size_t) prefillWindow * sizeof(int64_t));
        std::memcpy(embedInputs[0].data.data(), padded.data(), embedInputs[0].data.size());
        if (session->run(embedInputs, outputs) != Status::Ok || !findOutput("inputs_embeds"))
        {
            fprintf(stderr, "[vlm] embed run failed\n");
            return 3;
        }
        const IOTensor *promptEmbeds = findOutput("inputs_embeds");
        resizeDecoderInput(embedsInputIdx, {1, (int64_t) prefillWindow, hiddenDim}, DType::Float32);
        std::memcpy(decoderInputs[(size_t) embedsInputIdx].data.data(), promptEmbeds->data.data(), (size_t) prefillWindow * hiddenDim * sizeof(float));
        if (debugStats)
        {
            printStats("embed-out", promptEmbeds->f32(), (size_t) prefillWindow * hiddenDim);
        }

        // Splice: each image-token row takes the next image-embedding row. A prompt whose image
        // tokens have no (or too few) image rows fails THIS TURN only — like every other turn-level
        // input error here, it must not kill the persistent process and the conversation's cache.
        float *embedRows     = reinterpret_cast<float *>(decoderInputs[(size_t) embedsInputIdx].data.data());
        int    imageRowsUsed = 0;
        bool   spliceFailed  = false;
        for (int row = 0; row < promptLen; ++row)
        {
            if (prompt[(size_t) row] == imageToken)
            {
                if (imageEmbeddings.empty() || imageRowsUsed >= imageRowCount)
                {
                    fprintf(stderr, "[vlm] prompt has image tokens but no (or too few) image rows (row %d of %d)\n", imageRowsUsed, imageRowCount);
                    spliceFailed = true;
                    break;
                }
                std::memcpy(embedRows + (size_t) row * hiddenDim, imageEmbeddings.data() + (size_t) imageRowsUsed * hiddenDim, (size_t) hiddenDim * sizeof(float));
                ++imageRowsUsed;
            }
        }
        if (spliceFailed)
        {
            imageEmbeddings.clear();
            printf("END\n");
            fflush(stdout);
            continue;
        }
        if (imageRowsUsed != 0 && imageRowsUsed != imageRowCount)
        {
            fprintf(stderr, "[vlm] prompt consumed %d of %d image rows -- id stream and tile disagree\n", imageRowsUsed, imageRowCount);
        }

        // Additive mask [1,1,S,C+S]: populated cache slots visible to every row, new tokens causal.
        resizeDecoderInput(maskInputIdx, {1, 1, (int64_t) prefillWindow, (int64_t) (cacheSlots + prefillWindow)}, DType::Float32);
        {
            float *mask = reinterpret_cast<float *>(decoderInputs[(size_t) maskInputIdx].data.data());
            for (int row = 0; row < prefillWindow; ++row)
            {
                float *maskRow = mask + (size_t) row * (cacheSlots + prefillWindow);
                for (int slot = 0; slot < cacheSlots; ++slot)
                {
                    maskRow[slot] = slot < absolutePos ? 0.0f : kMaskFill;
                }
                for (int col = 0; col < prefillWindow; ++col)
                {
                    maskRow[cacheSlots + col] = col <= row ? 0.0f : kMaskFill;
                }
            }
        }
        resizeDecoderInput(positionInputIdx, {1, (int64_t) prefillWindow}, DType::Int64);
        {
            int64_t *positions = reinterpret_cast<int64_t *>(decoderInputs[(size_t) positionInputIdx].data.data());
            for (int row = 0; row < prefillWindow; ++row)
            {
                positions[row] = absolutePos + (row < promptLen ? row : promptLen - 1); // pad rows clamp; never folded
            }
        }
        if (session->run(decoderInputs, outputs) != Status::Ok)
        {
            fprintf(stderr, "[vlm] prefill run failed\n");
            return 3;
        }
        if (!foldPresentIntoCache(/*presentRows=*/prefillWindow, /*copyRows=*/promptLen, absolutePos))
        {
            fprintf(stderr, "[vlm] prefill present outputs missing\n");
            return 3;
        }
        const IOTensor *logitsOut = findOutput("logits");
        if (!logitsOut)
        {
            return 3;
        }
        absolutePos += promptLen;
        const float *logits = logitsOut->f32() + (size_t) (promptLen - 1) * vocabSize;
        if (debugStats)
        {
            printStats("spliced-embeds", reinterpret_cast<const float *>(decoderInputs[(size_t) embedsInputIdx].data.data()), (size_t) prefillWindow * hiddenDim);
            printStats("prefill-logits", logitsOut->f32(), (size_t) prefillWindow * vocabSize);
            printStats("last-row-logits", logits, (size_t) vocabSize);
            const IOTensor *present0 = findOutput("present.0.key");
            if (present0)
            {
                printStats("present.0.key", present0->f32(), (size_t) kvHeads * prefillWindow * headDim);
            }
        }

        imageEmbeddings.clear(); // the image belongs to this turn only
        for (int generated = 0; generated < maxTokens; ++generated)
        {
            const int64_t next = sampleNextToken(logits);
            if (next == eosToken)
            {
                // Fold the terminator into the cache too (its logits are discarded): the next turn
                // prefills against a cache holding the COMPLETE assistant reply including the
                // end-of-utterance token, keeping multi-turn context well-formed.
                if (absolutePos + 1 <= cacheSlots - 1 && decodeOneToken(next))
                {
                    ++absolutePos;
                }
                break;
            }
            printf("%lld\n", (long long) next);
            fflush(stdout);
            if (absolutePos + 1 > cacheSlots - 1)
            {
                fprintf(stderr, "[vlm] context full mid-generation\n");
                break;
            }
            logits = decodeOneToken(next);
            if (!logits)
            {
                fprintf(stderr, "[vlm] decode step failed\n");
                return 3;
            }
            ++absolutePos;
        }
        printf("END\n");
        fflush(stdout);
    }
    return 0;
}
