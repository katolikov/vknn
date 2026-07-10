// Bit-width-agnostic packed-weight encode/decode and the format-dispatching load-time
// materialization (scheme in quant_weights.h; format 1's layout authority is quant_int4.h). The
// pack side runs in the compile-time quantization pass; the dequant side runs in the session's
// materialization and in host tests.
#include "core/quant_weights.h"
#include "vknn/error.h"
#include <cstring>
#include <string>

namespace vknn {

    std::vector<uint8_t> int8Pack(const std::vector<int8_t> &q, int64_t K, int64_t N) {
        const int64_t        rowBytes = int8RowBytes(N);
        std::vector<uint8_t> packed((size_t) (K * rowBytes), 0);
        for (int64_t k = 0; k < K; ++k)
        {
            uint8_t *row = packed.data() + k * rowBytes;
            for (int64_t n = 0; n < N; ++n)
            {
                row[n] = (uint8_t) q[(size_t) (k * N + n)];
            }
        }
        return packed;
    }

    std::vector<float> int8Dequant(const uint8_t *packed, const uint16_t *scales, const int32_t *oidx,
                                   const uint16_t *oval, int64_t K, int64_t N, int64_t group, int64_t nOut) {
        const int64_t      rowBytes = int8RowBytes(N);
        std::vector<float> w((size_t) (K * N));
        for (int64_t k = 0; k < K; ++k)
        {
            const int64_t g = k / group;
            for (int64_t n = 0; n < N; ++n)
            {
                const float s           = halfToFloat(scales[(size_t) (g * N + n)]);
                w[(size_t) (k * N + n)] = (float) int8At(packed, rowBytes, k, n) * s;
            }
        }
        // Outlier columns overwrite their k rows with the kept fp16 values (their packed bytes are
        // 0, but an overwrite is the contract — the packed value at an outlier column is
        // unspecified).
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

    std::vector<float> lut4Dequant(const uint8_t *packed, const uint16_t *codebook, const uint16_t *scales,
                                   const int32_t *oidx, const uint16_t *oval, int64_t K, int64_t N,
                                   int64_t group, int64_t nOut) {
        const int64_t rowBytes = int4RowBytes(N);
        float         cb[16];
        for (int i = 0; i < 16; ++i)
        {
            cb[i] = halfToFloat(codebook[i]);
        }
        std::vector<float> w((size_t) (K * N));
        for (int64_t k = 0; k < K; ++k)
        {
            const int64_t g = k / group;
            for (int64_t n = 0; n < N; ++n)
            {
                // Unsigned nibble index (int4At sign-extends, so mask back to 0..15).
                const int   idx         = int4At(packed, rowBytes, k, n) & 0xF;
                const float s           = halfToFloat(scales[(size_t) (g * N + n)]);
                w[(size_t) (k * N + n)] = cb[idx] * s;
            }
        }
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

    int weightQuantFormat(const Node &node) {
        return node.attr.has(kWq) ? (int) node.attr.geti(kWq, 0) : 0;
    }

    bool weightQuantHasNativeMatMulKernel(int format) {
        return format == kWqFormatInt4 || format == kWqFormatInt8 || format == kWqFormatLut4;
    }

    int64_t materializeQuantWeights(Graph &g, const std::function<bool(size_t nodeIdx, const Node &)> &keepPacked) {
        int64_t materialized = 0;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            Node &nd = g.nodes[i];
            if (!nd.attr.has(kWq))
            {
                continue;
            }
            const int format = weightQuantFormat(nd);
            if (format != kWqFormatInt4 && format != kWqFormatInt8 && format != kWqFormatLut4)
            {
                // A format this build does not implement must fail the load: defaulting to any
                // implemented decode would dequantize garbage.
                throw Error(Status::Unsupported, "weight-quantization format " + std::to_string(format) +
                                                     " on node " + nd.name +
                                                     " is not supported by this build — reconvert the model or upgrade vknn");
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
            const TensorId lutId   = (TensorId) nd.attr.geti(kWqLut, kNoTensor);
            const HostBuffer &packedHb = g.initializers.at(weight);
            const HostBuffer &scaleHb  = g.initializers.at(scaleId);
            // Payloads may be unaligned .vxm mmap views; copy the typed side tensors out before the
            // element reads (the packed payload is read byte-wise, which needs no copy).
            std::vector<uint16_t> scales((size_t) (int4GroupCount(K, group) * N));
            std::memcpy(scales.data(), scaleHb.bytes.data(), scales.size() * 2);
            std::vector<int32_t>  oidx((size_t) nOut);
            std::vector<uint16_t> oval((size_t) (nOut * N));
            if (nOut > 0)
            {
                std::memcpy(oidx.data(), g.initializers.at(oidxId).bytes.data(), oidx.size() * 4);
                std::memcpy(oval.data(), g.initializers.at(ovalId).bytes.data(), oval.size() * 2);
            }
            std::vector<float> w;
            if (format == kWqFormatInt4)
            {
                w = int4Dequant(packedHb.bytes.data(), scales.data(), nOut > 0 ? oidx.data() : nullptr,
                                nOut > 0 ? oval.data() : nullptr, K, N, group, nOut);
            } else if (format == kWqFormatInt8)
            {
                w = int8Dequant(packedHb.bytes.data(), scales.data(), nOut > 0 ? oidx.data() : nullptr,
                                nOut > 0 ? oval.data() : nullptr, K, N, group, nOut);
            } else // kWqFormatLut4
            {
                std::vector<uint16_t> codebook(16);
                std::memcpy(codebook.data(), g.initializers.at(lutId).bytes.data(), codebook.size() * 2);
                w = lut4Dequant(packedHb.bytes.data(), codebook.data(), scales.data(),
                                nOut > 0 ? oidx.data() : nullptr, nOut > 0 ? oval.data() : nullptr, K, N,
                                group, nOut);
            }
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
            if (lutId != kNoTensor)
            {
                g.initializers.erase(lutId);
            }
            for (const char *key: {kWq, kWqK, kWqN, kWqGroup, kWqNOut, kWqLayout, kWqScales, kWqOidx, kWqOval, kWqLut})
            {
                nd.attr.map.erase(key);
            }
            ++materialized;
        }
        return materialized;
    }

} // namespace vknn
