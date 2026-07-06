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

    /// One matmul_tiled tile geometry: specialization constants 0/1/2 (TM/TN/TK). The workgroup
    /// stays 16x16 = 256 threads; each thread computes a (tm/16)x(tn/16) register micro-tile.
    struct MatMulTile {
        int tm, tn, tk;
    };

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
