// Packed-int4 payload encode/decode (layout contract in quant_int4.h). The pack side runs in the
// compile-time quantization pass; the dequant side runs in the session's load-time materialization
// and in host tests, so both directions live beside the layout definition.
#include "core/quant_int4.h"
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

    std::vector<float> int4Dequant(const uint8_t *packed, const uint16_t *scales, const int32_t *oidx,
                                   const uint16_t *oval, int64_t K, int64_t N, int64_t group, int64_t nOut) {
        const int64_t      rowBytes = int4RowBytes(N);
        std::vector<float> w((size_t) (K * N));
        for (int64_t k = 0; k < K; ++k)
        {
            const int64_t g = k / group;
            for (int64_t n = 0; n < N; ++n)
            {
                const float s      = halfToFloat(scales[(size_t) (g * N + n)]);
                w[(size_t) (k * N + n)] = (float) int4At(packed, rowBytes, k, n) * s;
            }
        }
        // Outlier columns overwrite their k rows with the kept fp16 values (their nibbles are 0, but
        // an overwrite is the contract — the packed value at an outlier column is unspecified).
        for (int64_t j = 0; j < nOut; ++j)
        {
            const int64_t k = oidx[j];
            for (int64_t n = 0; n < N; ++n)
            {
                w[(size_t) (k * N + n)] = halfToFloat(oval[(size_t) (j * N + n)]);
            }
        }
        return w;
    }

    int64_t materializeInt4Weights(Graph &g, const std::function<bool(size_t nodeIdx, const Node &)> &keepPacked) {
        int64_t materialized = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &nd = g.nodes[i];
            if (!nd.attr.has(kWq))
            {
                continue;
            }
            if (keepPacked && keepPacked(i, nd))
            {
                continue;
            }
            const TensorId weight  = nd.inputs[1];
            const int64_t  K       = nd.attr.geti(kWqK, 0);
            const int64_t  N       = nd.attr.geti(kWqN, 0);
            const int64_t  group   = nd.attr.geti(kWqGroup, 1);
            const int64_t  nOut    = nd.attr.geti(kWqNOut, 0);
            const int64_t  layout  = nd.attr.geti(kWqLayout, 0);
            const TensorId scaleId = (TensorId) nd.attr.geti(kWqScales, kNoTensor);
            const TensorId oidxId  = (TensorId) nd.attr.geti(kWqOidx, kNoTensor);
            const TensorId ovalId  = (TensorId) nd.attr.geti(kWqOval, kNoTensor);
            const HostBuffer &packedHb = g.initializers.at(weight);
            const HostBuffer &scaleHb  = g.initializers.at(scaleId);
            // Payloads may be unaligned .vxm mmap views; copy the typed side tensors out before the
            // element reads (int4Dequant reads the packed payload byte-wise, which needs no copy).
            std::vector<uint16_t> scales((size_t) (int4GroupCount(K, group) * N));
            std::memcpy(scales.data(), scaleHb.bytes.data(), scales.size() * 2);
            std::vector<int32_t>  oidx((size_t) nOut);
            std::vector<uint16_t> oval((size_t) (nOut * N));
            if (nOut > 0)
            {
                std::memcpy(oidx.data(), g.initializers.at(oidxId).bytes.data(), oidx.size() * 4);
                std::memcpy(oval.data(), g.initializers.at(ovalId).bytes.data(), oval.size() * 2);
            }
            const std::vector<float> w = int4Dequant(packedHb.bytes.data(), scales.data(),
                                                     nOut > 0 ? oidx.data() : nullptr,
                                                     nOut > 0 ? oval.data() : nullptr, K, N, group, nOut);
            // Back to the original tensor layout, narrowed to the fp16 the desc already declares.
            // Saturating keeps the reconstruction inside the same finite-fp16 contract every other
            // weight entry point applies (a pathological scale never materializes an infinity).
            std::vector<uint8_t> half((size_t) (K * N) * 2);
            fp16_t              *h = reinterpret_cast<fp16_t *>(half.data());
            if (layout == 0)
            {
                for (int64_t idx = 0; idx < K * N; ++idx)
                {
                    h[idx] = floatToHalfSat(w[(size_t) idx]);
                }
            } else
            {
                for (int64_t n = 0; n < N; ++n)
                {
                    for (int64_t k = 0; k < K; ++k)
                    {
                        h[n * K + k] = floatToHalfSat(w[(size_t) (k * N + n)]);
                    }
                }
            }
            HostBuffer hb;
            hb.bytes               = std::move(half);
            g.initializers[weight] = std::move(hb);
            g.initializers.erase(scaleId);
            if (nOut > 0)
            {
                g.initializers.erase(oidxId);
                g.initializers.erase(ovalId);
            }
            for (const char *key: {kWq, kWqK, kWqN, kWqGroup, kWqNOut, kWqLayout, kWqScales, kWqOidx, kWqOval})
            {
                nd.attr.map.erase(key);
            }
            ++materialized;
        }
        return materialized;
    }

} // namespace vknn
