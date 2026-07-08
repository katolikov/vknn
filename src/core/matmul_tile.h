// Shared matmul_tiled kernel-selection facts, consumed by the Vulkan MatMul op (kernel choice +
// specialization constants + tune-table decode) and the pointwise-fusion pass (epilogue attach
// refusal for tiled-shaped MatMuls). One definition keeps the op and the importer predicate in
// lock-step.
#pragma once
#include <cstdint>

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
    /// spec-constant matmul_tiled.comp: the Xclipse driver loses the inner-loop unroll and spills
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

} // namespace vknn
