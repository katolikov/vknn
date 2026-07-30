// Packed-int4 payload encode/decode (layout contract in quant_int4.h). The pack side runs in the
// compile-time quantization pass; the dequant side runs in the session's load-time materialization
// and in host tests, so both directions live beside the layout definition.
#include "core/quant_int4.h"
#include "core/quant_weights.h"
#include <cstring>

namespace vknn {

    std::vector<uint8_t> int4Pack(const std::vector<int8_t> &q, int64_t K, int64_t N) {
        const int64_t        rowBytes = int4RowBytes(N);
        std::vector<uint8_t> packed((size_t) (K * rowBytes), 0);
        for (int64_t k = 0; k < K; ++k)
        {
            uint8_t *row = packed.data() + k * rowBytes;
            for (int64_t n = 0; n < N; ++n)
            {
                const uint8_t nibble = (uint8_t) (q[(size_t) (k * N + n)] & 0xF);
                if (n & 1)
                {
                    row[n / 2] |= (uint8_t) (nibble << 4);
                } else
                {
                    row[n / 2] |= nibble;
                }
            }
        }
        return packed;
    }

    std::vector<float> int4Dequant(const uint8_t *packed, const uint16_t *scales, const int32_t *oidx, const uint16_t *oval, int64_t K, int64_t N, int64_t group, int64_t nOut) {
        const int64_t      rowBytes = int4RowBytes(N);
        std::vector<float> w((size_t) (K * N));
        for (int64_t k = 0; k < K; ++k)
        {
            const int64_t g = k / group;
            for (int64_t n = 0; n < N; ++n)
            {
                const float s           = halfToFloat(scales[(size_t) (g * N + n)]);
                w[(size_t) (k * N + n)] = (float) int4At(packed, rowBytes, k, n) * s;
            }
        }
        // Outlier columns overwrite their k rows with the kept fp16 values (their nibbles are 0, but
        // an overwrite is the contract — the packed value at an outlier column is unspecified).
        for (int64_t j = 0; j < nOut; ++j)
        {
            const int64_t k = oidx[j];
            if (k < 0 || k >= K)
            {
                continue; // outlier row index from an untrusted .vxm: w[k*N+n] would be an OOB write
            }
            for (int64_t n = 0; n < N; ++n)
            {
                w[(size_t) (k * N + n)] = halfToFloat(oval[(size_t) (j * N + n)]);
            }
        }
        return w;
    }

    int64_t materializeInt4Weights(Graph &g, const std::function<bool(size_t nodeIdx, const Node &)> &keepPacked) {
        // The format-dispatching materialization (quant_weights.cpp) subsumes the int4-only loop;
        // this entry point keeps the quant_int4.h contract name.
        return materializeQuantWeights(g, keepPacked);
    }

} // namespace vknn
