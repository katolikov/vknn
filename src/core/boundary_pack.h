// Host-side conversion between canonical fp32 NCHW tensors and the GPU boundary buffers (flat
// row-major or NC4HW4, fp16 or fp32), parallelized over cpu::parallelFor. Each element is computed
// by exactly the expression the serial loop uses, and the partitions write disjoint output rows, so
// every thread count produces byte-identical buffers (threads == 1 runs the plain serial loop).
// Lives in core (not the Vulkan backend) so the host test suite covers the byte-equality contract.
#pragma once
#include "vknn/dtype.h"
#include "vknn/nchw.h"

namespace vknn { namespace boundary {

    /// Canonical fp32 -> flat fp16 boundary (row-major; the layout is identity, only the precision
    /// converts). Bit-identical to floatToHalf() per element at any thread count.
    void packFlatFp16(const float *src, fp16_t *dst, int64_t elems, int threads);
    /// Flat fp16 boundary -> canonical fp32 (exact widening).
    void unpackFlatFp16(const fp16_t *src, float *dst, int64_t elems, int threads);
    /// Canonical fp32 NCHW -> NC4HW4 boundary. `dst` holds formatElems(NC4HW4, shape) elements of
    /// fp16 (2 bytes) or fp32 (4 bytes); channel padding lanes are written as zero.
    void packNc4(const float *src, void *dst, const NCHW &shape, bool fp16, int threads);
    /// NC4HW4 boundary -> canonical fp32 NCHW (`shape.elems()` destination elements).
    void unpackNc4(const void *src, float *dst, const NCHW &shape, bool fp16, int threads);

}} // namespace vknn::boundary
