// The parallel boundary pack/unpack (src/core/boundary_pack.h) is byte-identical to the serial
// nest at every thread count: the partitions write disjoint rows and evaluate exactly the serial
// loop's expressions, so the CPU stays the byte oracle for the GPU boundary conversion. The
// reference here is an independent copy of the serial NC4HW4 nest plus the scalar fp16 converts —
// not the function under test at threads=1 — so a restructuring bug in the shared code cannot hide.
#include "core/boundary_pack.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>
#if defined(VKNN_ENABLE_NEON) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

using namespace vknn;

namespace {

    // Deterministic value mix: sign changes, magnitudes across the fp16 normal/subnormal/overflow
    // ranges, and exact-tie mantissas that expose a rounding-rule mismatch.
    float testValue(int64_t i) {
        switch (i % 7)
        {
            case 0:
                return (float) (i % 1023) * 0.125f;
            case 1:
                return -(float) (i % 255) * 3.0517578125e-05f; // fp16 subnormal range
            case 2:
                return (float) (i % 97) * 1024.0f; // large magnitudes
            case 3:
                return 1.0009765625f + (float) (i % 11); // exact half-ULP ties at fp16
            case 4:
                return -70000.0f; // beyond fp16 max -> saturation
            case 5:
                return (float) (i % 13) * 1e-9f; // flushes to zero in fp16
            default:
                return (float) ((i * 2654435761u) % 8191) - 4095.0f;
        }
    }

    std::vector<float> makeSource(int64_t elems) {
        std::vector<float> v((size_t) elems);
        for (int64_t i = 0; i < elems; ++i)
        {
            v[(size_t) i] = testValue(i);
        }
        return v;
    }

    // Independent serial reference of the NC4HW4 pack: the pre-parallel loop nest verbatim,
    // including the platform's exact fp16 conversion (the NEON 4-lane vcvt where compiled in), so
    // "byte-identical to the serial path" is asserted against the same per-element instructions.
    void referencePackNc4(const float *src, void *dst, const NCHW &x, bool fp16) {
        int64_t Cb = cBlocks(x.c);
        for (int64_t n = 0; n < x.n; ++n)
        {
            for (int64_t cb = 0; cb < Cb; ++cb)
            {
                for (int64_t h = 0; h < x.h; ++h)
                {
                    for (int64_t w = 0; w < x.w; ++w)
                    {
                        int64_t base = ((((n * Cb + cb) * x.h + h) * x.w) + w) * 4;
                        float   t[4] = {0, 0, 0, 0};
                        for (int l = 0; l < 4; ++l)
                        {
                            int64_t c = cb * 4 + l;
                            if (c < x.c)
                            {
                                t[l] = src[((n * x.c + c) * x.h + h) * x.w + w];
                            }
                        }
                        if (fp16)
                        {
#if defined(VKNN_ENABLE_NEON) && defined(__ARM_NEON)
                            // Mirrors packNc4's saturating NEON convert (clamp to +/-65504; FMIN/FMAX
                            // propagate a NaN lane, matching the scalar path).
                            float32x4_t gathered = vld1q_f32(t);
                            gathered             = vmaxq_f32(vminq_f32(gathered, vdupq_n_f32(65504.0f)), vdupq_n_f32(-65504.0f));
                            vst1_f16(reinterpret_cast<__fp16 *>(static_cast<fp16_t *>(dst) + base), vcvt_f16_f32(gathered));
#else
                            for (int l = 0; l < 4; ++l)
                            {
                                static_cast<fp16_t *>(dst)[base + l] = floatToHalfSat(t[l]);
                            }
#endif
                        } else
                        {
                            for (int l = 0; l < 4; ++l)
                            {
                                static_cast<float *>(dst)[base + l] = t[l];
                            }
                        }
                    }
                }
            }
        }
    }

    void referenceUnpackNc4(const void *src, float *dst, const NCHW &x, bool fp16) {
        int64_t Cb = cBlocks(x.c);
        for (int64_t n = 0; n < x.n; ++n)
        {
            for (int64_t c = 0; c < x.c; ++c)
            {
                for (int64_t h = 0; h < x.h; ++h)
                {
                    for (int64_t w = 0; w < x.w; ++w)
                    {
                        int64_t cb = c / 4, l = c % 4;
                        int64_t sidx  = ((((n * Cb + cb) * x.h + h) * x.w) + w) * 4 + l;
                        float   value = fp16 ? halfToFloat(static_cast<const fp16_t *>(src)[sidx]) : static_cast<const float *>(src)[sidx];
                        dst[((n * x.c + c) * x.h + h) * x.w + w] = value;
                    }
                }
            }
        }
    }

    const Shape kShapes[] = {
        {1, 2, 1024, 64},  // LLM KV cache boundary
        {1, 3, 224, 224},  // classifier image input
        {1, 1, 720, 1920}, // frame-sized single channel
        {1, 733},          // rank-2 (flat-style geometry through NCHW::from)
        {1, 5, 17, 9},     // channel count not a multiple of 4 (padding lanes)
    };

} // namespace

// packNc4 at several thread counts is byte-identical to the independent serial reference, fp16 and
// fp32 alike, including the zeroed channel-padding lanes.
TEST(BoundaryPack, PackNc4MatchesSerialReferenceAtAnyThreadCount) {
    for (const Shape &shape: kShapes)
    {
        NCHW          x      = NCHW::from(shape);
        const int64_t elems  = x.elems();
        const int64_t packed = cBlocks(x.c) * 4 * x.n * x.h * x.w;
        auto          src    = makeSource(elems);
        for (bool fp16: {false, true})
        {
            const size_t         bytes = (size_t) packed * (fp16 ? 2 : 4);
            std::vector<uint8_t> want(bytes, 0xAB);
            referencePackNc4(src.data(), want.data(), x, fp16);
            for (int threads: {1, 2, 4, 7})
            {
                std::vector<uint8_t> got(bytes, 0xAB);
                boundary::packNc4(src.data(), got.data(), x, fp16, threads);
                EXPECT_EQ(got, want) << "shape [" << x.n << "," << x.c << "," << x.h << "," << x.w << "] fp16=" << fp16 << " threads=" << threads;
            }
        }
    }
}

// unpackNc4 at several thread counts is byte-identical to the independent serial reference.
TEST(BoundaryPack, UnpackNc4MatchesSerialReferenceAtAnyThreadCount) {
    for (const Shape &shape: kShapes)
    {
        NCHW          x      = NCHW::from(shape);
        const int64_t elems  = x.elems();
        const int64_t packed = cBlocks(x.c) * 4 * x.n * x.h * x.w;
        auto          canon  = makeSource(elems);
        for (bool fp16: {false, true})
        {
            // Build a device-side buffer via the reference pack, then unpack it back.
            std::vector<uint8_t> device((size_t) packed * (fp16 ? 2 : 4));
            referencePackNc4(canon.data(), device.data(), x, fp16);
            std::vector<float> want((size_t) elems, -1.f);
            referenceUnpackNc4(device.data(), want.data(), x, fp16);
            for (int threads: {1, 2, 4, 7})
            {
                std::vector<float> got((size_t) elems, -1.f);
                boundary::unpackNc4(device.data(), got.data(), x, fp16, threads);
                EXPECT_EQ(0, std::memcmp(got.data(), want.data(), got.size() * 4)) << "shape [" << x.n << "," << x.c << "," << x.h << "," << x.w << "] fp16=" << fp16 << " threads=" << threads;
            }
        }
    }
}

// The flat fp16 converts are byte-identical to the scalar per-element converts at any thread count
// (odd element counts exercise uneven chunk boundaries and the vector tail).
TEST(BoundaryPack, FlatFp16ConvertsMatchScalarAtAnyThreadCount) {
    for (int64_t elems: {(int64_t) 1, (int64_t) 16383, (int64_t) 131072, (int64_t) 1000003})
    {
        auto                src = makeSource(elems);
        std::vector<fp16_t> want((size_t) elems);
        for (int64_t i = 0; i < elems; ++i)
        {
            want[(size_t) i] = floatToHalfSat(src[(size_t) i]);
        }
        std::vector<float> wantBack((size_t) elems);
        for (int64_t i = 0; i < elems; ++i)
        {
            wantBack[(size_t) i] = halfToFloat(want[(size_t) i]);
        }
        for (int threads: {1, 2, 4, 7})
        {
            std::vector<fp16_t> got((size_t) elems, 0x7777);
            boundary::packFlatFp16(src.data(), got.data(), elems, threads);
            EXPECT_EQ(0, std::memcmp(got.data(), want.data(), got.size() * 2)) << "elems=" << elems << " threads=" << threads;
            std::vector<float> back((size_t) elems, -1.f);
            boundary::unpackFlatFp16(want.data(), back.data(), elems, threads);
            EXPECT_EQ(0, std::memcmp(back.data(), wantBack.data(), back.size() * 4)) << "elems=" << elems << " threads=" << threads;
        }
    }
}
