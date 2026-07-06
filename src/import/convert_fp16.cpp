// Post-pass fp16 weight conversion: rewrite every Float32 initializer payload as Float16 and stamp
// the tensor desc, halving the serialized model. Non-fp32 payloads (int64 shape tensors, ...) are
// kept as-is. Runs after the standard passes, immediately before saveGraphBin — the runtime decodes
// the fp16 payloads at load (initFloats / the session's CPU pool load).
#include "import/passes.h"
#include "vknn/dtype.h"
#include <cmath>

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

    Fp16ConvertStats convertInitializersFp16(Graph &g) {
        Fp16ConvertStats st;
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
