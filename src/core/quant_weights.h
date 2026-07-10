// Bit-width-agnostic packed-weight scheme layered over the frozen int4 contract (quant_int4.h).
// One attribute set (the kWq* names from quant_int4.h) describes every format; the VALUE of the kWq
// attr is the format id. quant_int4.h stays the authority for format 1's serialized meaning; this
// header adds the other formats and the shared dispatch every reader/writer of packed bytes uses.
//
// Format ids (the kWq attr value):
//   1 = int4  (kWqFormatInt4, == kWqVersion) — the frozen quant_int4.h layout: nibble payload,
//       rowBytes = 4*ceil(N/8), two's-complement [-7,7], value = nibble * scale.
//   2 = int8  (kWqFormatInt8) — 1 byte per element, k-major/n-minor:
//         rowBytes  = 8 * ceil(N/8)            (each k row padded to a whole 8-byte word PAIR: the
//                                               device reads the payload as a uint array, and the
//                                               GPU kernels load an 8-adjacent-n block as two
//                                               words — row granularity == block granularity, so a
//                                               tail block never reads past its row, exactly as
//                                               int4's 8-nibble word never does)
//         byte(k,n) = payload[k*rowBytes + n]  (two's-complement signed int8 in [-127,127];
//                                               symmetric, -128 never produced)
//         value(k,n) = q * scale[(k/group)*N + n]
//       Padding bytes (n >= N) and outlier columns (k in the oidx list) are 0.
//   3 = LUT4  (kWqFormatLut4) — the int4 nibble payload reinterpreted as UNSIGNED indices (0..15)
//       into one 16-entry fp16 codebook per tensor (side initializer kWqLut, Float16 [16]):
//         value(k,n) = codebook[index] * scale[(k/group)*N + n]
//       codebook[0] is EXACTLY 0.0: padding nibbles and outlier columns pack index 0, so they
//       contribute zero through the packed grid — the same role int4's zero nibble plays (the GPU
//       kernels never skip outlier k rows; they rely on the grid value being zero there). Entries
//       1..15 are fitted, fp16 values in ascending order.
//       The codebook is per tensor, not per group: a per-group table would multiply the side bank
//       by 16x (32 bytes vs 2 per group-column) while the per-group dynamic range is already
//       carried by the scale — measured on 1024x1024 near-normal weight populations (group 128,
//       absmax scales), a per-group fitted table matches a per-tensor fitted one to three decimals
//       of weight-space relative L2 (0.0919 vs 0.0918 Gaussian, 0.121 vs 0.119 heavy-tailed t(4))
//       while plain int4 sits at 0.117 / 0.193 — the shape win is tensor-wide, not per-group.
//
// Shared across every format (identical to quant_int4.h):
//   scales : Float16 [nGroups * N], nGroups = ceil(K/group) — n-minor, so the adjacent-n outputs a
//            subgroup's lanes own read their group scales as CONSECUTIVE fp16 halfwords of one
//            cache line (coalesced; an n-major bank would stride the simultaneous reads nGroups
//            elements apart)
//   oidx   : Int32   [nOut]     ascending k indices of the fp16-kept outlier columns
//   oval   : Float16 [nOut * N] row j = W[oidx[j], :] saturated to fp16
//   kWqK/kWqN/kWqGroup/kWqNOut/kWqLayout attrs, side tensors referenced ONLY from node attrs.
//
// Container versioning (model_io.cpp): a .vxm whose graphs carry ONLY format-1 weights keeps the
// VXM5 magic (readable by every engine with the int4 kernels); a .vxm carrying ANY other format
// stamps VXM6, whose body is the same [subtag u32: 3|4][VXM3/VXM4 body] wrapper. A pre-VXM6 engine
// rejects VXM6 with its "incompatible version — reconvert" message (the magic-prefix diagnostic),
// so a new format is never silently dequantized as int4. Within a loadable container, every reader
// dispatches on the kWq VALUE and fails loudly on an id it does not implement.
#pragma once
#include "core/quant_int4.h"
#include <cstdint>
#include <functional>
#include <vector>

namespace vknn {

    // Format ids carried as the kWq attr value. kWqFormatInt4 aliases kWqVersion: existing int4
    // containers already store 1.
    inline constexpr int kWqFormatInt4 = kWqVersion; // 1
    inline constexpr int kWqFormatInt8 = 2;
    inline constexpr int kWqFormatLut4 = 3;

    // Side-initializer attr for the LUT4 codebook (Float16 [16]); the other side attrs are shared
    // with int4 (quant_int4.h).
    inline constexpr const char *kWqLut = "wq_lut"; // int: TensorId of the codebook initializer

    // Bytes per packed int8 k row: N bytes, padded up to a whole 8-byte word pair (the kernels'
    // 8-column block width).
    inline int64_t int8RowBytes(int64_t n) noexcept {
        return 8 * ((n + 7) / 8);
    }

    // Pack quantized values q[k*N+n] (each in [-127,127]) into the int8 payload layout above.
    std::vector<uint8_t> int8Pack(const std::vector<int8_t> &q, int64_t K, int64_t N);

    // Signed byte at (k, n) of a packed int8 payload. Byte-addressed, so it is safe on unaligned
    // (mmap-view) payloads.
    inline int int8At(const uint8_t *packed, int64_t rowBytes, int64_t k, int64_t n) noexcept {
        return (int) (int8_t) packed[k * rowBytes + n];
    }

    // Reconstruct the logical [K, N] fp32 matrix from an int8 payload + scales + outliers, mirroring
    // int4Dequant's contract: exact fp32 dequantization (q * halfToFloat(s)), outlier rows
    // overwritten from oval.
    std::vector<float> int8Dequant(const uint8_t *packed, const uint16_t *scales, const int32_t *oidx,
                                   const uint16_t *oval, int64_t K, int64_t N, int64_t group, int64_t nOut);

    // Reconstruct the logical [K, N] fp32 matrix from a LUT4 payload (unsigned nibble indices) + the
    // 16-entry fp16 codebook + scales + outliers: value = codebook[index] * scale.
    std::vector<float> lut4Dequant(const uint8_t *packed, const uint16_t *codebook, const uint16_t *scales,
                                   const int32_t *oidx, const uint16_t *oval, int64_t K, int64_t N,
                                   int64_t group, int64_t nOut);

    // The format id of a node's packed weight: 0 when the node carries no kWq attr, the raw attr
    // value otherwise (a reader must reject values it does not implement — never default to int4).
    int weightQuantFormat(const Node &node);

    // True when the Vulkan MatMul has a native packed kernel for the format (matmul_gemv_*/
    // matmul_tiled_* wrappers); a format without one always materializes.
    bool weightQuantHasNativeMatMulKernel(int format);

    // Load-time materialization across every format: for each node carrying kWq whose
    // `keepPacked(nodeIdx, node)` is false, reconstruct the weight's fp16 payload in its original
    // tensor layout, drop the side initializers, and strip the attributes (the int4-format contract
    // in quant_int4.h, applied per format id). Throws Error(Unsupported) on a format id this build
    // does not implement — an unknown format must fail the load, never dequantize as another format.
    int64_t materializeQuantWeights(Graph &g, const std::function<bool(size_t nodeIdx, const Node &)> &keepPacked);

} // namespace vknn
