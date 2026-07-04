// Conv kernel-selection + GPU-pass knobs: the Hint selector, the Mode value set, and the parser.
#pragma once
#include <string>

namespace vknn {

    // Which knob a Mode value applies to (MNN-style Config::setHint). Autotune EFFORT is a separate
    // top-level Config::tuning field (see tuning.h), not a Hint.
    enum class Hint {
        Winograd        = 0, // 3x3 Winograd selection (Auto / On / Off)
        WinogradVariant = 1, // Winograd matmul impl (TiledGemm / Fused / FusedSplit / FullyFused / SubgroupGemm)
        WinogradUnit    = 2, // Winograd output tile (F23 / F43)
        DirectConv3x3   = 3, // direct 3x3 kernel (DirectAuto / RegisterTiled / LdsHalo)
        FlatLayout      = 4, // flat row-major GPU layout pass — keeps generic head ops on the GPU (On / Off, default On)
        GpuIslandFold   = 5, // fold tiny GPU op-islands onto the CPU (On / Off, default On)
    };

    // Every kernel/pass selection value, set uniformly via setHint(Hint, Mode). Each line is the set of
    // values valid for one Hint; the same underlying int recurs across groups (legal — the Hint picks the
    // knob, the Mode the value). Forcing Winograd On/Off skips per-shape timing, making the choice
    // deterministic run-to-run. FlatLayout/GpuIslandFold use On (default) / Off.
    enum class Mode {
        Auto = 0, On = 1, Off = 2,                                                  // Hint::Winograd, FlatLayout, GpuIslandFold
        TiledGemm = 0, Fused = 1, FusedSplit = 2, FullyFused = 3, SubgroupGemm = 4, // Hint::WinogradVariant
        F23 = 0, F43 = 4,                                                           // Hint::WinogradUnit
        DirectAuto = 0, RegisterTiled = 1, LdsHalo = 2,                             // Hint::DirectConv3x3
    };

    // Parse the Winograd knob from a string: "auto" (measure per shape), "on" (force Winograd), "off"
    // (force the direct kernel). Forcing on or off makes the 3x3-conv kernel choice deterministic.
    Mode winogradFromStr(const std::string &s); // "auto"/"on"/"off" -> Mode::Auto/On/Off

} // namespace vknn
