// Int8 KV-cache quantization core (Hint::KvCacheQuant): constants, buffer layouts, the host
// reference codec, and the eligibility rules shared by the Vulkan segment, the Vulkan
// FusedAttention op, and the CPU oracle. Pure host code (no Vulkan types), so every rule and
// codec path unit-tests on the CPU-only host build.
//
// Scheme: per-(token,head) symmetric int8 for BOTH K and V. The engine-resident cache keeps its
// [1, kvHeads, cacheSlots, headDim] geometry with a 1-byte int8 payload per element plus one fp16
// scale per (head, token) row. The scale buffer is laid out [kvHeads, cacheSlots] token-minor
// within head, so scaleIndex == payloadRowElementBase / headDim: the attention K-dot walking
// ascending tokens reads adjacent scale words (coalesced), and the fold addresses a row's scale
// straight from the row's destination offset. Cached K is post-RoPE, so a pre-RoPE per-channel
// scheme is structurally unavailable; at 8 bits the per-(token,head) scale is near-lossless and
// fold-local — a new token's row quantizes independently and old tokens are never rescanned.
//
// Quantize happens on cache WRITE (the resident-link fold of the present rows on the GPU;
// the host prefill re-seed of a rebound past input), dequantize is fused into the attention
// kernels' K-dot and V-apply loops: the per-token scale multiplies into the existing fp32 math
// (K-dot: dot * kScale[s]; V-apply: score * vScale[s] * v). The current step's rows (the
// split-KV new-rows sources, inputs[4]/[5]) stay fp16 — only the cached past is int8.
//
// Codec contract (bit-exact between the host helpers here, link_copy_kvq.comp, and the CPU
// oracle): absmax = max |x| over the row's headDim elements (NaN-skipping on the host, clamped
// to the fp16 finite ceiling so the scale is always finite); scale = absmax / 127 rounded to
// nearest-even fp16 (the STORED value — quantization divides by the rounded scale, exactly what
// the attention kernels multiply back); code = roundEven(x / scale) clamped to [-127, 127], zero
// when the scale is zero; dequant = float(code) * scale in fp32.
#pragma once
#include "vknn/config_struct.h"
#include "vknn/dtype.h"
#include "vknn/graph.h"
#include "core/fused_attention.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <set>
#include <vector>

namespace vknn {

    inline constexpr int   kKvQuantBits        = 8;      // payload bits per cached element
    inline constexpr float kKvQuantMaxCode     = 127.f;  // symmetric: codes span [-127, 127]
    inline constexpr int   kKvQuantScaleBytes  = 2;      // one fp16 scale per (head, token) row
    // The cache holds fp16-stored values, so no legitimate row magnitude exceeds the fp16 finite
    // ceiling; clamping absmax there keeps the scale finite even against a hostile host mirror.
    inline constexpr float kKvQuantAbsMaxCeil  = 65504.f;

    /// absmax over one cache row: NaN elements are skipped (std::fmax ignores a NaN operand) and
    /// the result saturates at the fp16 finite ceiling, so the derived scale is always finite.
    inline float kvQuantRowAbsMax(const float *row, int64_t headDim) {
        float absMax = 0.f;
        for (int64_t i = 0; i < headDim; ++i)
        {
            absMax = std::fmax(absMax, std::fabs(row[i]));
        }
        return std::fmin(absMax, kKvQuantAbsMaxCeil);
    }

    /// The row scale as STORED: absmax / 127 rounded to nearest-even fp16. Quantization must
    /// divide by this rounded value — the attention kernels multiply the stored fp16 scale back,
    /// so an unrounded divisor would bake a mismatch into every code.
    inline fp16_t kvQuantRowScaleBits(float absMax) {
        return floatToHalf(absMax / kKvQuantMaxCode);
    }

    /// int8 code for one element against the (fp16-rounded, fp32-widened) scale: roundEven with
    /// clamp to [-127, 127], matching the GPU shaders' roundEven + clamp. A zero scale (all-zero
    /// row) and any non-finite quotient encode 0, so the codec never emits an undefined narrow.
    inline int8_t kvQuantEncode(float value, float scale) {
        if (!(scale > 0.f))
        {
            return 0;
        }
        const float code = std::nearbyint(value / scale);
        if (!(code == code)) // NaN element against a valid scale
        {
            return 0;
        }
        return (int8_t) std::max(-kKvQuantMaxCode, std::min(kKvQuantMaxCode, code));
    }

    inline float kvQuantDecode(int8_t code, float scale) {
        return (float) code * scale;
    }

    /// Quantize `rows` consecutive dense rows of `headDim` fp32 values into the int8 payload and
    /// the per-row fp16 scale array (both indexed from row 0). The reference for the GPU fold and
    /// the shared implementation of the host seed path and the CPU oracle.
    inline void kvQuantRows(const float *src, int64_t rows, int64_t headDim, int8_t *payload, fp16_t *scaleBits) {
        for (int64_t r = 0; r < rows; ++r)
        {
            const float *row   = src + r * headDim;
            const fp16_t bits  = kvQuantRowScaleBits(kvQuantRowAbsMax(row, headDim));
            const float  scale = halfToFloat(bits);
            scaleBits[r]       = bits;
            for (int64_t i = 0; i < headDim; ++i)
            {
                payload[r * headDim + i] = kvQuantEncode(row[i], scale);
            }
        }
    }

    /// Dequantize `rows` consecutive rows back to fp32 — the readback inverse of kvQuantRows
    /// (resident-cache download, host mirror refresh).
    inline void kvDequantRows(const int8_t *payload, const fp16_t *scaleBits, int64_t rows, int64_t headDim, float *dst) {
        for (int64_t r = 0; r < rows; ++r)
        {
            const float scale = halfToFloat(scaleBits[r]);
            for (int64_t i = 0; i < headDim; ++i)
            {
                dst[r * headDim + i] = kvQuantDecode(payload[r * headDim + i], scale);
            }
        }
    }

    /// Seed-path variant: rounds every source value through fp16 storage FIRST, then quantizes.
    /// The host seed replaces a pack that would have stored fp16, and the GPU fold quantizes
    /// fp16-stored present rows — pre-rounding makes the seeded payload byte-identical to a fold
    /// of the same values. (A mirror downloaded from an fp16 device buffer is already exactly
    /// fp16-representable, so the pre-round is the identity on the production path.)
    inline void kvQuantRowsFromFp32ViaFp16(const float *src, int64_t rows, int64_t headDim, int8_t *payload, fp16_t *scaleBits) {
        std::vector<float> rounded((size_t) headDim);
        for (int64_t r = 0; r < rows; ++r)
        {
            for (int64_t i = 0; i < headDim; ++i)
            {
                rounded[(size_t) i] = halfToFloat(floatToHalf(src[r * headDim + i]));
            }
            kvQuantRows(rounded.data(), 1, headDim, payload + r * headDim, scaleBits + r);
        }
    }

    /// Structural eligibility of one FusedAttention node for the int8 past sources: the split-KV
    /// form whose past caches are dense row-contiguous [.., token, headDim] reads — K token stride
    /// == headDim with element stride 1, V token stride == headDim with output stride 1, and every
    /// per-row-dim stride a headDim multiple — so payloadRowElementBase / headDim indexes the
    /// scale buffer directly in both kernels. fp32-pinned operands are refused: the scheme is
    /// defined on the fp16 storage path only.
    /// Why one node is NOT eligible — nullptr when it is. The named reason is what the backends
    /// log when the hint asks for the scheme and nothing qualifies, so a refusal is always
    /// attributable to a specific structural fact rather than a bare "nothing qualified".
    inline const char *kvQuantNodeRefusal(const Graph &g, const Node &node) {
        if (node.type != OpType::FusedAttention)
        {
            return "not a FusedAttention node";
        }
        if (node.attr.geti(kFaSplit, 0) == 0)
        {
            return "attention is not the split-KV form (no folded KV concat)";
        }
        if (node.inputs.size() < 6)
        {
            return "split-KV attention without the new-rows inputs";
        }
        const int64_t headDim = node.attr.geti(kFaHd);
        if (headDim <= 0)
        {
            return "head dim is not static";
        }
        if (node.attr.geti(kFaPastLen) <= 0)
        {
            return "past length is zero (prefill-shaped bucket)";
        }
        if (node.attr.geti(kFaKK) != 1 || node.attr.geti(kFaKN) != headDim || node.attr.geti(kFaVN) != 1 || node.attr.geti(kFaVK) != headDim)
        {
            return "past K/V are not dense row-contiguous [.., token, headDim] reads";
        }
        for (const char *key: {kFaKStride, kFaVStride})
        {
            for (int64_t stride: node.attr.getints(key))
            {
                if (stride % headDim != 0)
                {
                    return "a past K/V row-dim stride is not a headDim multiple";
                }
            }
        }
        const TensorId kPast = node.inputs[1], vPast = node.inputs[2];
        if (kPast == kNoTensor || vPast == kNoTensor || kPast == vPast || g.isInitializer(kPast) || g.isInitializer(vPast))
        {
            return "past K/V are missing, aliased, or initializers";
        }
        // The past sources must be the resident cache itself: graph inputs, not mid-graph tensors.
        for (TensorId past: {kPast, vPast})
        {
            if (std::find(g.inputs.begin(), g.inputs.end(), past) == g.inputs.end())
            {
                return "past K/V are mid-graph tensors, not the resident cache inputs";
            }
        }
        for (TensorId t: node.inputs)
        {
            if (t != kNoTensor && g.desc(t).storeFp32)
            {
                return "an attention operand is pinned to fp32 storage";
            }
        }
        for (TensorId t: node.outputs)
        {
            if (t != kNoTensor && g.desc(t).storeFp32)
            {
                return "an attention output is pinned to fp32 storage";
            }
        }
        return nullptr;
    }

    inline bool kvQuantNodeEligible(const Graph &g, const Node &node) {
        return kvQuantNodeRefusal(g, node) == nullptr;
    }

    /// The refusal reason for a whole graph: the first FusedAttention node's reason, or a note that
    /// the graph has no fused attention at all. Only meaningful when kvQuantCacheTensors came back
    /// empty with the hint On.
    inline std::string kvQuantGraphRefusal(const Graph &g) {
        for (const Node &node: g.nodes)
        {
            if (node.type == OpType::FusedAttention)
            {
                const char *reason = kvQuantNodeRefusal(g, node);
                if (!reason)
                {
                    return "every attention node qualifies but no cache tensor passed the consumer/layout check";
                }
                std::string detail = reason;
                if (node.inputs.size() > 2)
                {
                    detail += " (past K '" + g.desc(node.inputs[1]).name + "', past V '" + g.desc(node.inputs[2]).name + "')";
                }
                return detail;
            }
        }
        return "the graph has no FusedAttention node";
    }

    /// Auto-mode threshold: the smallest resident cache (fp16 bytes, summed over every eligible
    /// cache tensor of one segment) at which halving the past-KV read traffic outweighs the
    /// in-kernel dequantize. A decode step streams the WHOLE compiled cache — the attention kernel
    /// walks all C slots — so the cache's compiled byte size, not the momentary fill, is what the
    /// saving scales with, and the rule is a static property of the plan rather than a timing race.
    /// Device-measured on a mobile Vulkan GPU with an int4 0.5B decoder: at 12.6 MB (24 layers x 2 kv
    /// heads x 1024 slots x 64) int8 costs about 4% of the decode step, at 25.2 MB (the same decoder
    /// at 2048 slots) it saves about 9%. The threshold sits at the upper measured point so Auto only
    /// engages where a win was actually observed.
    inline constexpr int64_t kKvQuantAutoMinCacheBytes = 24 * 1024 * 1024;

    /// Bytes one cache tensor occupies in the fp16 layout — the traffic the int8 payload halves.
    inline int64_t kvQuantCacheTensorFp16Bytes(const Graph &g, TensorId t) {
        return numElements(g.desc(t).shape) * (int64_t) sizeof(fp16_t);
    }

    /// Resolve Hint::KvCacheQuant for one segment's candidate set. On/Off are literal; Auto engages
    /// exactly when the eligible cache is at least kKvQuantAutoMinCacheBytes — a pure function of
    /// the compiled shapes, so every load of the same plan on the same device decides identically.
    inline bool kvQuantEnabled(const Config &cfg, int64_t candidateFp16Bytes) {
        const int mode = cfg.kvCacheQuantMode();
        if (mode == (int) Mode::Off)
        {
            return false;
        }
        if (mode == (int) Mode::On)
        {
            return true;
        }
        return candidateFp16Bytes >= kKvQuantAutoMinCacheBytes;
    }

    /// The cache tensors the int8 scheme applies to: the structural rules below, `backendEligible`
    /// (the Vulkan segment passes its device 8-bit-storage capability AND fp16 storage; the CPU
    /// oracle passes true — it models the quantized cache whenever the hint asks for it), every
    /// consumer of the tensor being an eligible FusedAttention reading it in a past slot (1 or 2),
    /// and finally kvQuantEnabled() over the qualifying set's total fp16 bytes. `requireFlat` adds
    /// the Vulkan device-layout constraint (gpuFlat cache tensors); the CPU caller passes false.
    /// Both the segment (buffer allocation) and the FusedAttention backends derive from this ONE
    /// rule, so payload buffers and kernel variants can never disagree.
    inline std::set<TensorId> kvQuantCacheTensors(const Graph &g, const Config &cfg, bool backendEligible, bool requireFlat) {
        std::set<TensorId> cacheTensors;
        if (!backendEligible || cfg.kvCacheQuantMode() == (int) Mode::Off)
        {
            return cacheTensors;
        }
        std::set<TensorId> candidates;
        for (const Node &node: g.nodes)
        {
            if (kvQuantNodeEligible(g, node))
            {
                candidates.insert(node.inputs[1]);
                candidates.insert(node.inputs[2]);
            }
        }
        int64_t qualifyingBytes = 0;
        for (TensorId t: candidates)
        {
            if (requireFlat && !g.desc(t).gpuFlat)
            {
                continue;
            }
            bool allConsumersEligible = true;
            for (const Node &node: g.nodes)
            {
                bool reads = node.fusedResidual == t || node.fusedBias == t;
                for (size_t i = 0; i < node.inputs.size() && !reads; ++i)
                {
                    reads = node.inputs[i] == t && !(kvQuantNodeEligible(g, node) && (i == 1 || i == 2));
                }
                if (reads)
                {
                    allConsumersEligible = false;
                    break;
                }
            }
            if (allConsumersEligible)
            {
                cacheTensors.insert(t);
                qualifyingBytes += kvQuantCacheTensorFp16Bytes(g, t);
            }
        }
        if (!kvQuantEnabled(cfg, qualifyingBytes))
        {
            cacheTensors.clear();
        }
        return cacheTensors;
    }

} // namespace vknn
