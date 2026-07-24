// Shared matmul_tiled kernel-selection facts, consumed by the Vulkan MatMul op (kernel choice +
// specialization constants + tune-table decode), the pointwise-fusion pass (epilogue attach
// refusal for tiled-shaped MatMuls), and the vec4-load twin routing (matmulVec4Route + the
// zero-padded weight repack). One definition keeps the op, the importer predicate, and the host
// tests in lock-step.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace vknn {

    /// The register-blocked tiled GEMM (shaders/matmul_tiled.comp) serves MatMuls with M, N, K all
    /// >= this; smaller and mat-vec shapes keep the naive 1-thread/output kernel. The pointwise
    /// fusion pass mirrors the same predicate when it refuses to attach VM units to tiled-shaped
    /// MatMuls (the register-blocked kernel has no register headroom for the VM).
    constexpr int64_t kTiledMatMulMin = 32;

    /// A mat-vec (M == 1) launches only N threads under matmul.comp's one-thread-per-output grid, so
    /// its time scales with K once N stops filling the GPU. shaders/matmul_gemv.comp splits the k
    /// reduction across GEMV_KS lanes per output instead. It takes over when the naive grid is too
    /// narrow to saturate (N < kGemvMaxN) and K is long enough for the split to pay for its shared-
    /// memory reduction (K >= kGemvMinK). Above kGemvMaxN the naive grid already reaches peak
    /// bandwidth (an LLM's [896,151936] lm_head projection does), so it keeps the simpler kernel.
    constexpr int64_t kGemvMinK = 512;
    constexpr int64_t kGemvMaxN = 16384;

    /// Output elements one mat-vec workgroup covers along the flat grid; the dispatch splits the
    /// output grid by it. Both mat-vec kernels cover exactly this many: matmul_gemv.comp as
    /// GEMV_NX == 64 scalar lanes, matmul_gemv4.comp as GEMV_NX == 16 lanes of GEMV_VEC == 4.
    constexpr int64_t kGemvNx = 64;

    /// Lanes each mat-vec kernel spends reducing over k (both shaders' GEMV_KS). With kGemvNx it sets the
    /// workgroup size the kernel needs the device to host: kGemvNx * kGemvKs invocations for the scalar
    /// kernel, kGemvNx / kGemvVec * kGemvKs for the vector one. Vulkan guarantees only 128, so MatMul
    /// checks maxComputeWorkGroupInvocations before selecting either.
    constexpr int64_t kGemvKs = 16;

    /// Adjacent n a single lane of shaders/matmul_gemv4.comp owns, loaded as one vector element of B.
    /// That vector index is exact only when B's n axis is kGemvVec-aligned at every k, which N %
    /// kGemvVec == 0 gives (it also aligns every batch stride, a multiple of K*N, and every
    /// workgroup's first n, a multiple of kGemvNx). An indivisible N keeps the scalar kernel; the two
    /// share a k partition and a per-output accumulation order, so the choice never affects bits.
    constexpr int64_t kGemvVec = 4;

    /// One matmul_tiled tile geometry: specialization constants 0/1/2 (TM/TN/TK). The workgroup
    /// stays 16x16 = 256 threads; each thread computes a (tm/16)x(tn/16) register micro-tile.
    struct MatMulTile {
        int tm, tn, tk;
    };

    /// The default {128,128,16} tile dispatches the compile-time matmul_tiled_fast.comp kernel
    /// (literal #defines -> fully-unrolled, register-resident micro-tile) instead of the
    /// spec-constant matmul_tiled.comp: the target mobile driver loses the inner-loop unroll and spills
    /// registers when the accumulator bounds are spec constants (~26% slower at this geometry), so
    /// the common case — every --tuning none run, and the race's usual winner — keeps main's fast
    /// literal kernel while non-default raced tiles use the flexible spec-constant kernel (ADR-0011:
    /// prefer compile-time shader variants over runtime-parameterized shaders on this driver). The
    /// two kernels are byte-identical at {128,128,16}.
    constexpr bool isDefaultMatMulTile(const MatMulTile &t) {
        return t.tm == 128 && t.tn == 128 && t.tk == 16;
    }

    /// Raced tile candidates. Index 0 is the Tuning::None default and stays {128,128,16} (the
    /// geometry the fixed #defines used); the tune table persists a winning INDEX into this array,
    /// so adding or reordering entries requires renaming the tune-signature stem ("mm_") to keep
    /// stale cached indices from decoding as a different tile. Every candidate is bit-neutral: the
    /// per-output fp32 K accumulation is one ascending-k chain for any TM/TN/TK — the tiles only
    /// remap threads to outputs and move the LDS barriers. Bounds: tm/tn <= 128 and tk <= 16 (the
    /// shader's TM_MAX/TN_MAX/TK_MAX shared-array sizing); tm*tk and tk*tn are multiples of 256
    /// (the cooperative load loops assign one element per thread per iteration).
    constexpr MatMulTile kMatMulTiles[] = {
        {128, 128, 16}, // the default: 8x8 micro-tile, 8 KB LDS at fp16
        {128, 64, 16},  // half the register pressure, more N-parallel workgroups
        {64, 128, 16},  // mirror, for M-heavy asymmetric shapes
        {64, 64, 16},   // small/medium matrices where a 128 tile is one mostly-padded workgroup
        {128, 128, 8},  // half the LDS of the default -> potentially two concurrent workgroups/CU
    };
    constexpr int kMatMulTileCount = (int) (sizeof(kMatMulTiles) / sizeof(kMatMulTiles[0]));

    /// Element alignment of the vec4-load GEMM twins (shaders/matmul_tiled_fast_v4*_fp16.comp).
    /// Their cooperative panel loads fetch f16vec4 — four contiguous fp16 as one 64-bit load — so
    /// every global element index they form must be a multiple of four: K % kVec4Align == 0 keeps
    /// each A vec4 inside one A row, a kVec4Align-multiple physical B row stride keeps each B vec4
    /// inside one B row, and kVec4Align-multiple batch strides keep both properties across the
    /// batch decode.
    constexpr int64_t kVec4Align = 4;

    /// n rounded up to the next kVec4Align multiple: the physical row stride (bNp) a zero-padded B
    /// repack presents to the vec4 kernels.
    constexpr int64_t roundUpVec4(int64_t n) {
        return (n + kVec4Align - 1) / kVec4Align * kVec4Align;
    }

    /// matmulVec4Route's verdict. bNp is the physical B row stride the v4 kernels consume
    /// (MatMulPC::bNp): N for a naturally aligned packed B, roundUpVec4(N) for a repacked weight.
    struct MatMulVec4Route {
        bool    eligible = false; ///< the default {128,128,16} tile dispatches the v4 kernel
        bool    padB     = false; ///< B repacks to a zero-padded copy with row stride bNp ("#wv4")
        int64_t bNp      = 0;     ///< physical B row stride for the v4 kernel (== N unless padB)
    };

    /// Deterministic routing rule for the vec4-load GEMM twins — a pure shape/layout function so
    /// the Vulkan MatMul op and the host tests share one definition. The twins are byte-identical
    /// to the scalar fast kernels (same LDS layout, same ascending-k fp32 accumulation, same
    /// rounding); this rule trades global-load width only, never bits, so it needs no race and no
    /// accuracy gate. It holds when every global element index the kernel forms is 4-aligned:
    ///   - useFp16: the twins are hand-written fp16 kernels; a precision-high (fp32) node keeps
    ///     the scalar kernels.
    ///   - K % kVec4Align == 0: bounds each A vec4 inside its row (A is the runtime activation and
    ///     K a model dim, so the rule gates on it — no A repack, no tail path).
    ///   - every A/B batch stride a kVec4Align multiple: keeps the per-batch base offsets aligned.
    ///   - N % kVec4Align == 0, or B is a constant initializer whose rows can repack to a
    ///     zero-padded bNp = roundUpVec4(N). The repack is restricted to an all-zero B batch
    ///     stride (the broadcast Linear-weight case): a nonzero batch stride would need rescaling
    ///     to the padded layout in the geometry SSBO. A runtime-activation B cannot repack, so it
    ///     routes only when N is naturally aligned.
    /// `bPayloadResident` reports whether the initializer's host payload is still readable: a cold
    /// start computes the repack from it, and gating here (rather than on weight-cache warmth)
    /// keeps the routing identical between cold and warm starts.
    inline MatMulVec4Route matmulVec4Route(bool useFp16, int64_t N, int64_t K, const std::vector<int32_t> &aBatchStride, const std::vector<int32_t> &bBatchStride, bool bIsInitializer, bool bPayloadResident) {
        MatMulVec4Route route;
        route.bNp       = N;
        auto allAligned = [](const std::vector<int32_t> &strides) {
            for (int32_t stride: strides)
            {
                if (stride % kVec4Align != 0)
                {
                    return false;
                }
            }
            return true;
        };
        if (!useFp16 || N <= 0 || K <= 0 || K % kVec4Align != 0 || !allAligned(aBatchStride))
        {
            return route;
        }
        if (N % kVec4Align == 0)
        {
            route.eligible = allAligned(bBatchStride);
            return route;
        }
        if (!bIsInitializer || !bPayloadResident)
        {
            return route;
        }
        for (int32_t stride: bBatchStride)
        {
            if (stride != 0)
            {
                return route;
            }
        }
        route.eligible = true;
        route.padB     = true;
        route.bNp      = roundUpVec4(N);
        return route;
    }

    /// Zero-padded row repack for the v4 route's "#wv4" weight upload: `rows` rows of `n` source
    /// elements each copy into rows of `np` elements (np >= n, a kVec4Align multiple from
    /// roundUpVec4); the tail np - n elements of every row are zero. The pad zeros contribute
    /// nothing to the accumulator and mirror the value the scalar kernel's column guard yields, so
    /// the repacked operand is output-byte-neutral. `src` holds at least rows * n elements.
    inline std::vector<float> padMatMulRowsVec4(const std::vector<float> &src, int64_t rows, int64_t n, int64_t np) {
        std::vector<float> padded((size_t) (rows * np), 0.0f);
        for (int64_t row = 0; row < rows; ++row)
        {
            std::copy_n(src.data() + row * n, (size_t) n, padded.data() + row * np);
        }
        return padded;
    }

} // namespace vknn
