// JNI bridge for the on-device Qwen chat demo: wraps a VKNN Session as an autoregressive decoder,
// lifting the token loop from examples/chat.cpp into a handle the Kotlin app drives one step at a time.
//
// The model is a with-past Qwen2 decoder compiled at a fixed context length C (one plan serves prefill,
// token by token, and every decode step). The past key/value buffers ARE the KV cache: fp32 host
// boundary tensors retained across steps; each step appends the new token's key/value (present slot C)
// into cache slot p. Every tensor-compute op runs on the Vulkan backend; only argmax/sampling is here.
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

    // One loaded decoder: the Session plus the persistent boundary tensors, the layer index maps, the
    // running absolute position, and the last logits row (so sampling is a separate call from stepping).
    struct Decoder {
        std::unique_ptr<Session> sess;
        std::vector<IOTensor>    inputs;  // persistent, in model input order; KV buffers survive across steps
        std::vector<IOTensor>    outputs; // last run()'s outputs
        std::vector<IOInfo>      inInfo, outInfo;

        int idIdx = -1, maskIdx = -1, posIdx = -1, logitsIdx = -1;
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
                if (inInfo[i].name == n)
                    return (int) i;
            return -1;
        }
        int findOut(const std::string &n) const {
            for (size_t i = 0; i < outInfo.size(); ++i)
                if (outInfo[i].name == n)
                    return (int) i;
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
                am[(size_t) j] = 1; // valid past slots
            am[(size_t) C] = 1;      // the current token (present slot C)
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
                    for (size_t k = 0; k < outInfo.size(); ++k)
                        if (outputs[j].name == outInfo[k].name)
                            outIdxByInfo[k] = (int) j;
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
            const float *lg = logits.data();
            if (temp <= 0.0f)
            {
                int64_t best = 0;
                float   bv   = lg[0];
                for (int64_t i = 1; i < vocab; ++i)
                    if (lg[i] > bv)
                    {
                        bv   = lg[i];
                        best = i;
                    }
                return best;
            }
            std::vector<std::pair<float, int64_t>> v((size_t) vocab);
            for (int64_t i = 0; i < vocab; ++i)
                v[(size_t) i] = {lg[i] / temp, i};
            const int keep = topK > 0 && topK < (int) vocab ? topK : (int) vocab;
            std::partial_sort(v.begin(), v.begin() + keep, v.end(),
                              [](const auto &a, const auto &b) { return a.first > b.first; });
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
                    return v[i].second;
            }
            return v[n - 1].second;
        }
    };

    Precision precFromStr(const std::string &s) {
        if (s == "normal")
            return Precision::Normal;
        if (s == "high")
            return Precision::High;
        return Precision::Low;
    }

    std::string jstr(JNIEnv *env, jstring s) {
        if (!s)
            return {};
        const char *c = env->GetStringUTFChars(s, nullptr);
        std::string out(c ? c : "");
        if (c)
            env->ReleaseStringUTFChars(s, c);
        return out;
    }

} // namespace

extern "C" {

// Load the .vxm and build a Decoder. Returns a native handle (0 on failure).
JNIEXPORT jlong JNICALL Java_com_vknn_chat_NativeLib_nativeInit(JNIEnv *env, jobject, jstring jvxm, jstring jcache, jstring jprec) {
    auto *d = new Decoder();
    Config cfg;
    cfg.backend                = BackendKind::Vulkan;
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
            break;
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
}

// int[5] = {L, kv_heads, C, head_dim, vocab}.
JNIEXPORT jintArray JNICALL Java_com_vknn_chat_NativeLib_nativeInfo(JNIEnv *env, jobject, jlong ptr) {
    auto     *d   = reinterpret_cast<Decoder *>(ptr);
    jint      v[5] = {d->L, d->kvHeads, d->C, d->headDim, (jint) d->vocab};
    jintArray a   = env->NewIntArray(5);
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
        return -1;
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

} // extern "C"
