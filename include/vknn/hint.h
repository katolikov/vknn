// Conv kernel-selection knobs: the Hint selector, the Mode value set, and their string parsers.
#pragma once
#include <string>

namespace vknn {

    // Which conv kernel knob a Mode value applies to (MNN-style Config::setHint).
    enum class Hint {
        Winograd        = 0, // 3x3 Winograd selection (Auto / On / Off)
        Tuning          = 1, // autotune effort (NoTune / Fast / Thorough)
        WinogradVariant = 2, // Winograd matmul impl (TiledGemm / Fused / FusedSplit / FullyFused / SubgroupGemm)
        WinogradUnit    = 3, // Winograd output tile (F23 / F43)
        DirectConv3x3   = 4, // direct 3x3 kernel (DirectAuto / RegisterTiled / LdsHalo)
    };

    // Every conv kernel-selection value, set uniformly via setHint(Hint, Mode). Each line is the set of
    // values valid for one Hint; the same underlying int recurs across groups (legal — the Hint picks the
    // knob, the Mode the value). Forcing Winograd On/Off skips per-shape timing, making the choice
    // deterministic run-to-run.
    enum class Mode {
        Auto = 0, On = 1, Off = 2,                                                  // Hint::Winograd
        NoTune = 0, Fast = 1, Thorough = 2,                                         // Hint::Tuning
        TiledGemm = 0, Fused = 1, FusedSplit = 2, FullyFused = 3, SubgroupGemm = 4, // Hint::WinogradVariant
        F23 = 0, F43 = 4,                                                           // Hint::WinogradUnit
        DirectAuto = 0, RegisterTiled = 1, LdsHalo = 2,                             // Hint::DirectConv3x3
    };

    // Parse the conv autotune knobs from a string. winograd: "auto" (measure per shape), "on" (force
    // Winograd), "off" (force the direct kernel). tuning: "off" / "fast" / "thorough". Forcing winograd
    // on or off makes the 3x3-conv kernel choice deterministic (no per-run timing measurement).
    Mode winogradFromStr(const std::string &s); // "auto"/"on"/"off" -> Mode::Auto/On/Off
    Mode tuningFromStr(const std::string &s);   // "off"/"fast"/"thorough" -> Mode::NoTune/Fast/Thorough

} // namespace vknn
