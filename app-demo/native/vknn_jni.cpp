// JNI bridge for the on-device Qwen chat demo: wraps a VKNN Session as an autoregressive decoder,
// lifting the token loop from examples/chat.cpp into a handle the Kotlin app drives one step at a time.
//
// The model is a with-past Qwen2 decoder compiled at a fixed context length C (one plan serves prefill,
// token by token, and every decode step). The past key/value buffers ARE the KV cache: fp32 host
// boundary tensors retained across steps; each step appends the new token's key/value (present slot C)
// into cache slot p. Every tensor-compute op runs on the Vulkan backend; only argmax/sampling is here.
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
        std::vector<IOTensor>    inputs;  // persistent, in model input order; KV buffers survive across steps
        std::vector<IOTensor>    outputs; // last run()'s outputs
        std::vector<IOInfo>      inInfo, outInfo;

        int              idIdx = -1, maskIdx = -1, posIdx = -1, logitsIdx = -1;
        std::vector<int> pastKey, pastVal, presKey, presVal;
        std::vector<int> outIdxByInfo; // outInfo index -> index in outputs vector (stable across runs)
        bool             mapped = false;

        int     L = 0, kvHeads = 0, C = 0, headDim = 0;
        int64_t vocab = 0;
        int     p     = 0; // absolute position across the whole conversation

        std::vector<float> logits; // last step's logits row (vocab floats)
        std::mt19937       rng {1234};

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
            std::memcpy(inputs[(size_t) idx].data.data(), vals.data(), vals.size() * sizeof(int64_t));
        }

        // Feed one token at the current position: write id/pos/mask, run the plan on the GPU, append the
        // new key/value into the KV cache, and keep the logits row. Returns 0 on success, -1 on error.
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

            if (sess->run(inputs, outputs) != Status::Ok)
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
                        const float *s = src + ((size_t) h * (C + 1) + C) * headDim;
                        float       *d = dst + ((size_t) h * C + slot) * headDim;
                        std::memcpy(d, s, (size_t) headDim * sizeof(float));
                    }
                }
            }
            const IOTensor &lo = outputs[(size_t) outIdxByInfo[(size_t) logitsIdx]];
            const float    *lp = reinterpret_cast<const float *>(lo.data.data());
            logits.assign(lp, lp + vocab);
            return 0;
        }

        // Next token from the stored logits: greedy at temp<=0, else temperature + top-k + top-p.
        int64_t sample(float temp, int topK, float topP) {
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
        std::vector<IOTensor>    embedIn; // single "input_ids" tensor
        std::vector<IOTensor>    visInT;  // single "pixel_values" tensor
        std::vector<IOTensor>    outs;    // last run()'s outputs

        std::vector<int> pastKey, pastVal;
        int              embIdx = -1, maskIdx = -1, posIdx = -1;
        int              L = 0, kvHeads = 0, C = 0, headDim = 0, H = 0;
        int              prefillS = 0, imgRows = 0, imgSide = 0;
        int64_t          vocab = 0;
        int              p     = 0; // absolute token position across the conversation

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

        // Prefill one prompt: embed the padded ids, splice image rows over image-token rows, build the
        // additive mask + clamped positions, run the prefill bucket, and fold the real present rows.
        // Pad rows are masked causally anyway but are NEVER folded into the cache.
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
            std::vector<int64_t> padded(ids, ids + nReal);
            padded.resize((size_t) prefillS, padId);
            embedIn[0].shape = {1, (int64_t) prefillS};
            embedIn[0].data.resize((size_t) prefillS * sizeof(int64_t));
            std::memcpy(embedIn[0].data.data(), padded.data(), embedIn[0].data.size());
            if (sess->run(embedIn, outs) != Status::Ok || !outByName("inputs_embeds"))
            {
                LOGE("embed run failed");
                return -1;
            }
            const IOTensor *emb = outByName("inputs_embeds");
            setDecShape(embIdx, {1, (int64_t) prefillS, (int64_t) H}, DType::Float32);
            std::memcpy(dec[(size_t) embIdx].data.data(), emb->data.data(), (size_t) prefillS * H * sizeof(float));

            // Splice: each image-token row takes the next image-embedding row, in order.
            float *rows   = reinterpret_cast<float *>(dec[(size_t) embIdx].data.data());
            int    imgRow = 0;
            for (int q = 0; q < nReal; ++q)
            {
                if (ids[q] == imageToken)
                {
                    if (!imgEmb || imgRow >= imgRowsGiven)
                    {
                        LOGE("prompt has image tokens but no (or too few) image rows (row %d of %d)", imgRow, imgRowsGiven);
                        return -1;
                    }
                    std::memcpy(rows + (size_t) q * H, imgEmb + (size_t) imgRow * H, (size_t) H * sizeof(float));
                    ++imgRow;
                }
            }
            if (imgRow != 0 && imgRow != imgRowsGiven)
            {
                LOGE("prompt consumed %d of %d image rows -- id stream and tile disagree", imgRow, imgRowsGiven);
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

            if (sess->run(dec, outs) != Status::Ok)
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

        // One decode step: token id -> embedding bucket at S=1 -> decoder S=1 bucket -> logits row.
        int step(int64_t tok) {
            if (p + 1 > C - 1)
            {
                LOGE("context full at p=%d", p);
                return -2;
            }
            embedIn[0].shape = {1, 1};
            embedIn[0].data.resize(sizeof(int64_t));
            std::memcpy(embedIn[0].data.data(), &tok, sizeof(int64_t));
            if (sess->run(embedIn, outs) != Status::Ok)
            {
                return -1;
            }
            const IOTensor *emb = outByName("inputs_embeds");
            if (!emb)
            {
                return -1;
            }
            setDecShape(embIdx, {1, 1, (int64_t) H}, DType::Float32);
            std::memcpy(dec[(size_t) embIdx].data.data(), emb->data.data(), (size_t) H * sizeof(float));
            setDecShape(maskIdx, {1, 1, 1, (int64_t) C + 1}, DType::Float32);
            float *m = reinterpret_cast<float *>(dec[(size_t) maskIdx].data.data());
            for (int c = 0; c < C + 1; ++c)
            {
                m[c] = (c < p || c == C) ? 0.0f : kMaskFill;
            }
            setDecShape(posIdx, {1, 1}, DType::Int64);
            int64_t pos = p;
            std::memcpy(dec[(size_t) posIdx].data.data(), &pos, sizeof(int64_t));
            if (sess->run(dec, outs) != Status::Ok)
            {
                return -1;
            }
            const int slot = p < C ? p : C - 1;
            if (!foldPresent(1, 1, slot))
            {
                return -1;
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

// Load the .vxm and build a Decoder. Returns a native handle (0 on failure).
JNIEXPORT jlong JNICALL Java_com_vknn_chat_NativeLib_nativeInit(JNIEnv *env, jobject, jstring jvxm, jstring jcache, jstring jprec, jstring jbackend) {
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
        d->inInfo  = d->sess->inputInfo();
        d->outInfo = d->sess->outputInfo();

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
        if (d->idIdx < 0 || d->maskIdx < 0 || d->posIdx < 0 || d->logitsIdx < 0 || d->L == 0)
        {
            LOGE("model is not a qwen2 with-past decoder");
            delete d;
            return 0;
        }
        const Shape &ks = d->inInfo[(size_t) d->pastKey[0]].shape; // [1, kv_heads, C, head_dim]
        d->kvHeads      = (int) ks[1];
        d->C            = (int) ks[2];
        d->headDim      = (int) ks[3];
        d->vocab        = d->outInfo[(size_t) d->logitsIdx].shape.back();
        d->logits.assign((size_t) d->vocab, 0.0f);

        d->inputs.resize(d->inInfo.size());
        for (size_t i = 0; i < d->inInfo.size(); ++i)
        {
            d->inputs[i].name  = d->inInfo[i].name;
            d->inputs[i].shape = d->inInfo[i].shape;
            d->inputs[i].dtype = d->inInfo[i].dtype;
            d->inputs[i].data.assign((size_t) d->inInfo[i].elems * dtypeSize(d->inInfo[i].dtype), 0);
        }
        LOGI("loaded: L=%d kv_heads=%d C=%d head_dim=%d vocab=%lld", d->L, d->kvHeads, d->C, d->headDim, (long long) d->vocab);
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

// int[5] = {L, kv_heads, C, head_dim, vocab}.
JNIEXPORT jintArray JNICALL Java_com_vknn_chat_NativeLib_nativeInfo(JNIEnv *env, jobject, jlong ptr) {
    auto     *d    = reinterpret_cast<Decoder *>(ptr);
    jint      v[5] = {d->L, d->kvHeads, d->C, d->headDim, (jint) d->vocab};
    jintArray a    = env->NewIntArray(5);
    env->SetIntArrayRegion(a, 0, 5, v);
    return a;
}

// Reset the conversation: position 0 and a cleared KV cache. Reseeds the sampler RNG.
JNIEXPORT void JNICALL Java_com_vknn_chat_NativeLib_nativeReset(JNIEnv *, jobject, jlong ptr, jint seed) {
    auto *d = reinterpret_cast<Decoder *>(ptr);
    d->p    = 0;
    d->rng.seed((unsigned) seed);
    for (int l = 0; l < d->L; ++l)
    {
        std::fill(d->inputs[(size_t) d->pastKey[l]].data.begin(), d->inputs[(size_t) d->pastKey[l]].data.end(), (uint8_t) 0);
        std::fill(d->inputs[(size_t) d->pastVal[l]].data.begin(), d->inputs[(size_t) d->pastVal[l]].data.end(), (uint8_t) 0);
    }
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

        // Discover the bucket roles by input names/shapes (see examples/llm/vlm.cpp):
        // vision "pixel_values" [1,3,IMG,IMG]; embed "input_ids" [1,S] at S=prefill and S=1;
        // decoder "inputs_embeds" [1,S,H] + mask/positions/past at the same two S values.
        int          visionB = -1, embedPreB = -1, embedDecB = -1, decPreB = -1, decDecB = -1;
        const size_t nb = m->sess->bucketCount();
        for (size_t b = 0; b < nb; ++b)
        {
            std::vector<IOInfo> in = m->sess->inputInfo(b);
            if (findByName(in, "pixel_values") >= 0)
            {
                visionB = (int) b;
            } else if (in.size() == 1 && in[0].name == "input_ids")
            {
                (in[0].shape.back() == 1 ? embedDecB : embedPreB) = (int) b;
            } else if (findByName(in, "inputs_embeds") >= 0)
            {
                const int     ie = findByName(in, "inputs_embeds");
                const int64_t S  = in[(size_t) ie].shape[1];
                if (S == 1)
                {
                    decDecB = (int) b;
                } else
                {
                    decPreB     = (int) b;
                    m->prefillS = (int) S;
                }
            }
        }
        if (visionB < 0 || embedPreB < 0 || embedDecB < 0 || decPreB < 0 || decDecB < 0)
        {
            LOGE("not a vision-decoder multi-graph .vxm (%zu buckets)", nb);
            delete m;
            return 0;
        }
        m->visIn  = m->sess->inputInfo((size_t) visionB);
        m->visOut = m->sess->outputInfo((size_t) visionB);
        m->decIn  = m->sess->inputInfo((size_t) decPreB);
        m->decOut = m->sess->outputInfo((size_t) decPreB);

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
        m->embIdx           = findByName(m->decIn, "inputs_embeds");
        m->maskIdx          = findByName(m->decIn, "attention_mask");
        m->posIdx           = findByName(m->decIn, "position_ids");
        const int logitsIdx = findByName(m->decOut, "logits");
        if (m->L == 0 || m->embIdx < 0 || m->maskIdx < 0 || m->posIdx < 0 || logitsIdx < 0)
        {
            LOGE("decoder bucket misses inputs_embeds/attention_mask/position_ids/logits/past");
            delete m;
            return 0;
        }
        const Shape &ks = m->decIn[(size_t) m->pastKey[0]].shape; // [1, KV, C, HD]
        m->kvHeads      = (int) ks[1];
        m->C            = (int) ks[2];
        m->headDim      = (int) ks[3];
        m->H            = (int) m->decIn[(size_t) m->embIdx].shape[2];
        m->vocab        = m->decOut[(size_t) logitsIdx].shape.back();
        const Shape &vs = m->visOut[0].shape; // [1, imageRows, H]
        m->imgRows      = (int) vs[1];
        m->imgSide      = (int) m->visIn[0].shape.back();
        m->logits.assign((size_t) m->vocab, 0.0f);

        m->dec.resize(m->decIn.size());
        for (size_t i = 0; i < m->decIn.size(); ++i)
        {
            m->dec[i].name  = m->decIn[i].name;
            m->dec[i].shape = m->decIn[i].shape;
            m->dec[i].dtype = m->decIn[i].dtype;
            m->dec[i].data.assign((size_t) m->decIn[i].elems * dtypeSize(m->decIn[i].dtype), 0);
        }
        m->embedIn.resize(1);
        m->embedIn[0].name  = "input_ids";
        m->embedIn[0].dtype = DType::Int64;
        m->visInT.resize(1);
        m->visInT[0].name  = "pixel_values";
        m->visInT[0].shape = m->visIn[0].shape;
        m->visInT[0].dtype = DType::Float32;

        LOGI("vlm loaded: L=%d kv_heads=%d C=%d head_dim=%d H=%d vocab=%lld prefillS=%d imgRows=%d img=%d", m->L, m->kvHeads, m->C, m->headDim, m->H,
             (long long) m->vocab, m->prefillS, m->imgRows, m->imgSide);
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

// Reset the conversation: position 0 and a cleared KV cache. Reseeds the sampler RNG.
JNIEXPORT void JNICALL Java_com_vknn_chat_NativeLib_nativeVlmReset(JNIEnv *, jobject, jlong ptr, jint seed) {
    auto *m = reinterpret_cast<Vlm *>(ptr);
    m->p    = 0;
    m->rng.seed((unsigned) seed);
    for (int l = 0; l < m->L; ++l)
    {
        std::fill(m->dec[(size_t) m->pastKey[l]].data.begin(), m->dec[(size_t) m->pastKey[l]].data.end(), (uint8_t) 0);
        std::fill(m->dec[(size_t) m->pastVal[l]].data.begin(), m->dec[(size_t) m->pastVal[l]].data.end(), (uint8_t) 0);
    }
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

// int[4] = {gaussians, views, height, width}. gaussians is 0 until an encode succeeds.
JNIEXPORT jintArray JNICALL Java_com_vknn_chat_NativeLib_nativeSplatInfo(JNIEnv *env, jobject, jlong ptr) {
    auto     *splat   = reinterpret_cast<Splat *>(ptr);
    jint      info[4] = {splat->rasterizer ? splat->rasterizer->gaussians() : 0, splat->views, splat->height, splat->width};
    jintArray out     = env->NewIntArray(4);
    env->SetIntArrayRegion(out, 0, 4, info);
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
