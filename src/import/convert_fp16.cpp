// Post-pass fp16 weight conversion: rewrite every Float32 initializer payload as Float16 and stamp
// the tensor desc, halving the serialized model. Non-fp32 payloads (int64 shape tensors, ...) are
// kept as-is, and so is every COORDINATE-CLASS initializer (see coordinateKeepSet): a sampling
// coordinate consumed at fp16 loses ~0.5 px at a 2k-wide axis (fp16 resolves ~2^-11 of the [-1,1]
// range), a loss no runtime precision mode can recover once the payload is narrowed. Runs after
// the standard passes, immediately before saveGraphBin — the runtime decodes the fp16 payloads at
// load (initFloats / the session's CPU pool load).
#include "import/passes.h"
#include "vknn/dtype.h"
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace vknn {

    // Largest finite fp16 magnitude. A finite fp32 value beyond this range narrows to +/-inf, and an
    // fp32 +/-inf stays +/-inf. Either way an fp16 inf is an operand hazard: the attention-mask idiom
    // (1 - mask) * -3.4028235e38 multiplies the non-pad 0 by that constant, and 0 * -inf is NaN, which
    // poisons the softmax and the whole output. Clamping the constant to the finite extreme keeps
    // 0 * -65504 = 0, while a -65504 additive bias still drives exp() to 0 for pad tokens.
    static constexpr float kFp16MaxFinite = 65504.0f;

    // Saturate an out-of-fp16-range or infinite input to +/-kFp16MaxFinite so the conversion produces
    // a finite fp16 value. NaN passes through unchanged.
    static inline float clampToFp16Range(float v) {
        if (std::isnan(v))
        {
            return v;
        }
        if (v > kFp16MaxFinite)
        {
            return kFp16MaxFinite;
        }
        if (v < -kFp16MaxFinite)
        {
            return -kFp16MaxFinite;
        }
        return v;
    }

    // Initializers whose stored bits reach a sampling-coordinate input (GridSample's grid, or the
    // fused warp variant's flow/base operands) through coordinate-transparent ops only. These keep
    // their fp32 payloads: their precision IS the sample position.
    static std::unordered_set<TensorId> coordinateKeepSet(const Graph &g) {
        std::unordered_map<TensorId, const Node *> producer;
        for (const Node &n: g.nodes)
        {
            for (TensorId out: n.outputs)
            {
                if (out != kNoTensor)
                {
                    producer[out] = &n;
                }
            }
        }
        std::unordered_set<TensorId> keep, seen;
        std::vector<TensorId>        walk;
        for (const Node &n: g.nodes)
        {
            if (n.type != OpType::GridSample)
            {
                continue;
            }
            for (size_t i = 1; i < n.inputs.size(); ++i) // every coordinate operand; 0 is the image
            {
                if (n.inputs[i] != kNoTensor)
                {
                    walk.push_back(n.inputs[i]);
                }
            }
        }
        while (!walk.empty())
        {
            TensorId t = walk.back();
            walk.pop_back();
            if (!seen.insert(t).second)
            {
                continue;
            }
            if (g.isInitializer(t))
            {
                keep.insert(t);
                continue;
            }
            auto it = producer.find(t);
            if (it == producer.end() || !coordinateTransparentOp(it->second->type))
            {
                continue; // a graph input, or real computation: fp16 activations dominate past here
            }
            for (TensorId in: it->second->inputs)
            {
                if (in != kNoTensor)
                {
                    walk.push_back(in);
                }
            }
        }
        return keep;
    }

    // A per-SPATIAL-axis coordinate ramp: rank >= 3 with exactly one non-1 dim, that dim on one of
    // the two trailing (H/W) axes and at least this long. Such a constant is a position vector
    // (meshgrid axis, feather/threshold ramp), and its fp16 error (2^-11 of the value range)
    // multiplies by whatever slope the downstream algebra applies - a steep mask edge turns
    // half-ULP coordinate noise into whole-pixel flips. The payload is one axis long (a few KB),
    // so keeping it fp32 is free next to the image-sized tensors it broadcasts over. The trailing-
    // axis requirement keeps biases [C] and channel scales [1,C,1,1] on the fp16 path: their
    // values are magnitudes, not positions, and the runtime converts them to compute precision at
    // load either way.
    constexpr int64_t kCoordRampMinExtent = 32;

    static bool isAxisRamp(const Shape &s) {
        if (s.size() < 3)
        {
            return false;
        }
        int64_t nonUnit = 0;
        size_t  where   = 0;
        int64_t extent  = 0;
        for (size_t i = 0; i < s.size(); ++i)
        {
            if (s[i] != 1)
            {
                ++nonUnit;
                where  = i;
                extent = s[i];
            }
        }
        return nonUnit == 1 && where >= s.size() - 2 && extent >= kCoordRampMinExtent;
    }

    Fp16ConvertStats convertInitializersFp16(Graph &g) {
        Fp16ConvertStats                   st;
        const std::unordered_set<TensorId> coordKeep = coordinateKeepSet(g);
        for (auto &kv: g.initializers)
        {
            TensorDesc &d = g.tensors[kv.first];
            st.bytesBefore += (int64_t) kv.second.bytes.size();
            if (d.dtype != DType::Float32)
            {
                st.bytesAfter += (int64_t) kv.second.bytes.size();
                ++st.kept;
                continue;
            }
            if (coordKeep.count(kv.first) != 0 || isAxisRamp(d.shape))
            {
                st.bytesAfter += (int64_t) kv.second.bytes.size();
                ++st.kept;
                ++st.keptCoord;
                continue;
            }
            // Element count from the payload size, not numElements(): the two agree for rank >= 1,
            // but numElements() is 0 for a rank-0 scalar (shape []), which would emit an empty
            // payload and destroy the value. The stored fp32 payload is tightly packed, so its byte
            // size is exact.
            int64_t              n   = (int64_t) (kv.second.bytes.size() / sizeof(float));
            const float         *src = kv.second.f32();
            std::vector<uint8_t> half((size_t) n * 2);
            fp16_t              *h = reinterpret_cast<fp16_t *>(half.data());
            for (int64_t i = 0; i < n; ++i)
            {
                h[i] = floatToHalf(clampToFp16Range(src[i]));
            }
            kv.second.bytes = std::move(half);
            d.dtype         = DType::Float16;
            st.bytesAfter += (int64_t) kv.second.bytes.size();
            ++st.converted;
        }
        return st;
    }

} // namespace vknn
