// Post-pass fp16 weight conversion: rewrite every Float32 initializer payload as Float16 and stamp
// the tensor desc, halving the serialized model. Non-fp32 payloads (int64 shape tensors, ...) are
// kept as-is. Runs after the standard passes, immediately before saveGraphBin — the runtime decodes
// the fp16 payloads at load (initFloats / the session's CPU pool load).
#include "import/passes.h"
#include "vknn/dtype.h"

namespace vknn {

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
                h[i] = floatToHalf(src[i]);
            }
            kv.second.bytes = std::move(half);
            d.dtype         = DType::Float16;
            st.bytesAfter += (int64_t) kv.second.bytes.size();
            ++st.converted;
        }
        return st;
    }

} // namespace vknn
