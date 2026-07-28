// Packed-int4 weight representation shared by the compile-time quantization pass
// (src/import/quantize_weights.cpp), the session's load-time materialization, and the Vulkan MatMul
// int4 kernels. One canonical layout; every reader/writer of packed bytes goes through here.
//
// A quantized weight keeps its LOGICAL TensorDesc (original shape, dtype Float16) so shape-reading
// gates and planners never see the packing; only the initializer payload holds packed bytes. The
// consumer node carries the quantization attributes (kWq* below) that describe the payload and
// reference the side initializers (scales / outlier columns) by tensor id.
//
// Logical view: a [K, N] matrix W[k][n] — K the reduction axis, N the output channels — matching
// MatMul's B orientation. kWqLayout says how the ORIGINAL tensor maps onto that view:
//   0 = the tensor is [K, N] row-major (MatMul B, Gemm transB=0)
//   1 = the tensor is [N, K] row-major (Gemm transB=1; Conv with K = Cin/g*kH*kW flattened per
//       output channel)
//
// Packed payload (k-major, n-minor — adjacent n share bytes, so GPU lanes spread over n read
// coalesced words):
//   rowBytes  = 4 * ceil(N/8)              (each k row padded to a whole 4-byte word)
//   byte(k,n) = payload[k*rowBytes + n/2]  (even n = low nibble, odd n = high nibble)
//   nibble    = two's-complement signed int4 in [-7, 7] (symmetric; -8 never produced)
//   value(k,n) = nibble * scale[(k/group)*N + n]
// Padding nibbles (n >= N) and outlier columns (k in the oidx list) are 0.
//
// Side initializers (side tensors are referenced ONLY from the node attrs, never node.inputs, so
// operand-arity gates and the pointwise-chain machinery are untouched; the standard passes — and
// pruneDeadInitializers — have already run when they are added, and a compiled .vxm never re-runs
// them):
//   scales : Float16 [nGroups * N], nGroups = ceil(K/group)
//   oidx   : Int32   [nOut]          ascending k indices of the fp16-kept outlier columns
//   oval   : Float16 [nOut * N]      row j = W[oidx[j], :] saturated to fp16
#pragma once
#include "vknn/dtype.h"
#include "vknn/graph.h"
#include <cstdint>
#include <functional>
#include <vector>

namespace vknn {

    // Node attribute names for a quantized weight. kWq is the scheme version (1); its presence marks
    // the node's weight input (MatMul/Gemm/Conv input[1]) as packed.
    inline constexpr const char *kWq       = "wq";        // int: scheme version, 1
    inline constexpr const char *kWqK      = "wq_k";      // int: K, the reduction extent
    inline constexpr const char *kWqN      = "wq_n";      // int: N, the output-channel extent
    inline constexpr const char *kWqGroup  = "wq_group";  // int: scale group size along k
    inline constexpr const char *kWqNOut   = "wq_nout";   // int: outlier column count (may be 0)
    inline constexpr const char *kWqLayout = "wq_layout"; // int: 0 = tensor is [K,N], 1 = [N,K]
    inline constexpr const char *kWqScales = "wq_scales"; // int: TensorId of the scales initializer
    inline constexpr const char *kWqOidx   = "wq_oidx";   // int: TensorId of oidx (absent when nOut=0)
    inline constexpr const char *kWqOval   = "wq_oval";   // int: TensorId of oval (absent when nOut=0)

    inline constexpr int kWqVersion = 1;

    // Bytes per packed k row: N nibbles, padded up to a whole 4-byte word so the device reads the
    // payload as a uint array.
    inline int64_t int4RowBytes(int64_t n) noexcept {
        return 4 * ((n + 7) / 8);
    }
    // Scale group count along k.
    inline int64_t int4GroupCount(int64_t k, int64_t group) noexcept {
        return (k + group - 1) / group;
    }

    // Pack quantized values q[k*N+n] (each in [-7,7]) into the payload layout above.
    std::vector<uint8_t> int4Pack(const std::vector<int8_t> &q, int64_t K, int64_t N);

    // Signed nibble at (k, n) of a packed payload. Byte-addressed, so it is safe on unaligned
    // (mmap-view) payloads.
    inline int int4At(const uint8_t *packed, int64_t rowBytes, int64_t k, int64_t n) noexcept {
        const uint8_t byte   = packed[k * rowBytes + n / 2];
        const int     nibble = (n & 1) ? (byte >> 4) : (byte & 0xF);
        return (int) ((int8_t) (nibble << 4)) >> 4; // sign-extend the low 4 bits
    }

    // Reconstruct the logical [K, N] fp32 matrix from packed payload + scales + outliers. scales are
    // raw fp16 words; oidx/oval may be null when nOut == 0. Dequantization is exact in fp32
    // (q * halfToFloat(s)), so a CPU consumer of the materialized weight computes on the same values
    // the int4 GPU kernel dequantizes in-register.
    std::vector<float> int4Dequant(const uint8_t *packed, const uint16_t *scales, const int32_t *oidx, const uint16_t *oval, int64_t K, int64_t N, int64_t group, int64_t nOut);

    // Load-time materialization: for every node carrying the kWq attributes whose `keepPacked(nodeIdx,
    // node)` is false, reconstruct the weight's fp16 payload in its original tensor layout, drop the
    // side initializers, and strip the attributes — the consumer then sees a plain fp16 weight, so
    // quantization is invisible to every backend without a native packed kernel (all CPU nodes, and
    // every GPU op except the int4 MatMul). Runs in the session after backend assignment, before the
    // CPU pool load and any GPU op prepare. Returns the number of weights materialized.
    int64_t materializeInt4Weights(Graph &g, const std::function<bool(size_t nodeIdx, const Node &)> &keepPacked);

} // namespace vknn
