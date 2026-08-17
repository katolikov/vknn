// See boundary_pack.h. The NC4HW4 loops partition over the (n, channel-block/channel, h) row
// product and keep the innermost w loop serial per row, so each parallel chunk executes the same
// per-element expressions, in the same order, as the serial nest — the byte-oracle contract the
// host test pins. The flat converts chunk the contiguous range; floatToHalfSatBulk/halfToFloatBulk
// are bit-identical to their scalar per-element forms, so chunk boundaries cannot change any value.
// fp16 packs saturate out-of-range values to +/-65504 (matching GPU activation stores and imported
// constants) instead of overflowing to +/-inf.
#include "core/boundary_pack.h"
#include "backend/cpu/parallel.h"
#if defined(VKNN_ENABLE_NEON) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace vknn { namespace boundary {

    // The flat converts are contiguous NEON sweeps: one core already saturates the memory bus, and
    // splitting the range costs more in pool wake-up and little-core stragglers than the work itself
    // (2-5x slower at 1.4M elements). They run on the calling thread; `threads` stays in the signature
    // for symmetry with the NC4 siblings, which gather strided channel planes and are not bus-bound.
    void packFlatFp16(const float *src, fp16_t *dst, int64_t elems, int threads) {
        (void) threads;
        // Saturating: a value beyond +/-65504 packs as the max finite fp16, never as +/-inf.
        floatToHalfSatBulk(src, dst, elems);
    }

    void unpackFlatFp16(const fp16_t *src, float *dst, int64_t elems, int threads) {
        (void) threads;
        halfToFloatBulk(src, dst, elems);
    }

    void packNc4(const float *src, void *dst, const NCHW &shape, bool fp16, int threads) {
        const NCHW    x        = shape;
        const int64_t Cb       = cBlocks(x.c);
        const int64_t rows     = x.n * Cb * x.h;                // one row = the w sweep of one (n, cb, h)
        const int64_t minChunk = cpu::minChunkForWork(x.w * 4); // 4 lanes gathered+converted per w
        if (fp16)
        {
            fp16_t *out = static_cast<fp16_t *>(dst);
            cpu::parallelFor(threads, 0, rows, minChunk, [&](int64_t rowBegin, int64_t rowEnd) {
                for (int64_t row = rowBegin; row < rowEnd; ++row)
                {
                    const int64_t h  = row % x.h;
                    const int64_t cb = (row / x.h) % Cb;
                    const int64_t n  = row / (x.h * Cb);
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
#if defined(VKNN_ENABLE_NEON) && defined(__ARM_NEON)
                        // Saturate to +/-65504 (FMIN/FMAX propagate a NaN lane, so NaN passes like
                        // the scalar path), then convert the 4 gathered channels to fp16 in one
                        // instruction.
                        float32x4_t gathered = vld1q_f32(t);
                        gathered             = vmaxq_f32(vminq_f32(gathered, vdupq_n_f32(65504.0f)), vdupq_n_f32(-65504.0f));
                        vst1_f16(reinterpret_cast<__fp16 *>(out + base), vcvt_f16_f32(gathered));
#else
                        for (int l = 0; l < 4; ++l)
                        {
                            out[base + l] = floatToHalfSat(t[l]);
                        }
#endif
                    }
                }
            });
        } else
        {
            float *out = static_cast<float *>(dst);
            cpu::parallelFor(threads, 0, rows, minChunk, [&](int64_t rowBegin, int64_t rowEnd) {
                for (int64_t row = rowBegin; row < rowEnd; ++row)
                {
                    const int64_t h  = row % x.h;
                    const int64_t cb = (row / x.h) % Cb;
                    const int64_t n  = row / (x.h * Cb);
                    for (int64_t w = 0; w < x.w; ++w)
                    {
                        int64_t base = ((((n * Cb + cb) * x.h + h) * x.w) + w) * 4;
                        for (int l = 0; l < 4; ++l)
                        {
                            int64_t c     = cb * 4 + l;
                            out[base + l] = (c < x.c) ? src[((n * x.c + c) * x.h + h) * x.w + w] : 0.f;
                        }
                    }
                }
            });
        }
    }

    void unpackNc4(const void *src, float *dst, const NCHW &shape, bool fp16, int threads) {
        const NCHW    x        = shape;
        const int64_t Cb       = cBlocks(x.c);
        const int64_t rows     = x.n * x.c * x.h; // one row = the w sweep of one (n, c, h)
        const int64_t minChunk = cpu::minChunkForWork(x.w);
        if (fp16)
        {
            const fp16_t *in = static_cast<const fp16_t *>(src);
            cpu::parallelFor(threads, 0, rows, minChunk, [&](int64_t rowBegin, int64_t rowEnd) {
                for (int64_t row = rowBegin; row < rowEnd; ++row)
                {
                    const int64_t h  = row % x.h;
                    const int64_t c  = (row / x.h) % x.c;
                    const int64_t n  = row / (x.h * x.c);
                    const int64_t cb = c / 4, l = c % 4;
                    for (int64_t w = 0; w < x.w; ++w)
                    {
                        int64_t sidx                             = ((((n * Cb + cb) * x.h + h) * x.w) + w) * 4 + l;
                        dst[((n * x.c + c) * x.h + h) * x.w + w] = halfToFloat(in[sidx]);
                    }
                }
            });
        } else
        {
            const float *in = static_cast<const float *>(src);
            cpu::parallelFor(threads, 0, rows, minChunk, [&](int64_t rowBegin, int64_t rowEnd) {
                for (int64_t row = rowBegin; row < rowEnd; ++row)
                {
                    const int64_t h  = row % x.h;
                    const int64_t c  = (row / x.h) % x.c;
                    const int64_t n  = row / (x.h * x.c);
                    const int64_t cb = c / 4, l = c % 4;
                    for (int64_t w = 0; w < x.w; ++w)
                    {
                        int64_t sidx                             = ((((n * Cb + cb) * x.h + h) * x.w) + w) * 4 + l;
                        dst[((n * x.c + c) * x.h + h) * x.w + w] = in[sidx];
                    }
                }
            });
        }
    }

}} // namespace vknn::boundary
