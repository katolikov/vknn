// Conv kernel-selection + GPU-pass knobs: the Hint selector, the Mode value set, and the parser.
#pragma once
#include <string>

namespace vknn {

    /// Which knob a Mode value applies to (MNN-style Config::setHint). Autotune EFFORT is a separate
    /// top-level Config::tuning field (see tuning.h), not a Hint.
    enum class Hint {
        Winograd        = 0, ///< 3x3 Winograd selection (Auto / On / Off).
        WinogradVariant = 1, ///< Winograd matmul impl (TiledGemm / Fused / FusedSplit / FullyFused / SubgroupGemm).
        WinogradUnit    = 2, ///< Winograd output tile (F23 / F43).
        DirectConv3x3   = 3, ///< Direct 3x3 kernel (DirectAuto / RegisterTiled / LdsHalo).
        FlatLayout      = 4, ///< Flat row-major GPU layout pass — keeps generic head ops on the GPU (On / Off, default On).
        GpuIslandFold   = 5, ///< Fold tiny GPU op-islands onto the CPU (On / Off, default On).
        MatMulViewFold  = 6, ///< Fold Transpose/Expand chains into MatMul operand views at load (On / Off, default On).
        RopeFusion      = 7, ///< Fuse rotate-half RoPE chains into one Rope dispatch at load (On / Off, default On).
        FusedAttention  = 8, ///< Fuse the M=1 decode-attention chain into one FusedAttention kernel at load (On / Off, default On).
        KvConcatFold    = 9, ///< Fold the per-token KV-cache Concat into split-source FusedAttention reads at load (On / Off, default On). The rows-only present output it produces drives the engine-resident KV link like the cache-concat present (the link fold source comes from the present shape; see io_link.h).
    };

    /// Every kernel/pass selection value, set uniformly via setHint(Hint, Mode). The value sets by
    /// Hint: Auto/On/Off serve Winograd, FlatLayout, GpuIslandFold, MatMulViewFold, RopeFusion,
    /// FusedAttention and KvConcatFold;
    /// TiledGemm..SubgroupGemm serve WinogradVariant; F23/F43 serve WinogradUnit;
    /// DirectAuto..LdsHalo serve DirectConv3x3. The same underlying int recurs across groups
    /// (legal — the Hint picks the knob, the Mode the value). Forcing Winograd On/Off skips
    /// per-shape timing, making the choice deterministic run-to-run.
    enum class Mode {
        Auto          = 0,
        On            = 1,
        Off           = 2,
        TiledGemm     = 0,
        Fused         = 1,
        FusedSplit    = 2,
        FullyFused    = 3,
        SubgroupGemm  = 4,
        F23           = 0,
        F43           = 4,
        DirectAuto    = 0,
        RegisterTiled = 1,
        LdsHalo       = 2,
    };

    /// Parse the Winograd knob from a string. Forcing on or off makes the 3x3-conv kernel choice
    /// deterministic run-to-run (no per-shape timing).
    /// @param s One of "auto" (measure per shape), "on" (force Winograd), "off" (force the direct kernel).
    /// @returns Mode::Auto, Mode::On, or Mode::Off respectively.
    Mode winogradFromStr(const std::string &s);

} // namespace vknn
