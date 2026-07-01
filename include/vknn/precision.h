// Precision tier enum, the selective-fp32 preset, and the string parser.
#pragma once
#include <string>

namespace vknn {

    // Quality tiers (string tokens "low" / "normal" / "high"; "fp16" / "fp32" kept as aliases):
    //   Low    fp16 storage + fp32 accumulation everywhere.
    //   Normal fp16 storage, but a built-in geometry-tail set is kept fp32 (selective fp32).
    //   High   full fp32 storage.
    enum class Precision { Low = 0, Normal = 1, High = 2 };

    // The default selective-fp32 set used by Precision::Normal ("normal") when Config::fp32Tensors is empty:
    // the comma-separated tensor-name substrings of the geometry tail that benefit from fp32 storage
    // without the NaN-fragile camera-pose SVD. A no-op for models without these names (e.g. CNNs).
    const char *mixedPrecisionFp32Tensors();

    // Precision tier from a string: "low"/"fp16", "normal"/"mixed", "high"/"fp32" (unknown -> low).
    Precision precisionFromStr(const std::string &s);

} // namespace vknn
