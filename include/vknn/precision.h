/// Precision tier enum, the selective-fp32 preset, and the string parser.
#pragma once
#include <string>

namespace vknn {

    /// Storage-precision tier for inference, selecting how activations and weights are held on the GPU.
    /// Every tier accumulates in fp32 inside kernels; the tier governs only what is stored between them.
    /// The enumerator order is ascending precision, so the values double as a comparable rank.
    enum class Precision {
        Low    = 0, ///< fp16 storage everywhere, fp32 accumulation. Fastest and lowest memory.
        Normal = 1, ///< fp16 storage, except a built-in geometry-tail set kept fp32 (see mixedPrecisionFp32Tensors()).
        High   = 2, ///< Full fp32 storage. Highest fidelity, largest footprint.
    };

    /// The built-in selective-fp32 tensor set applied by Precision::Normal when Config::fp32Tensors is empty.
    ///
    /// @returns A comma-separated list of tensor-name substrings for the geometry tail that benefits from
    ///          fp32 storage, deliberately excluding the NaN-fragile camera-pose SVD. Matching is by name
    ///          substring, so the preset is a no-op for models without these names (e.g. CNNs). The returned
    ///          pointer refers to a static string literal and is valid for the program lifetime.
    const char *mixedPrecisionFp32Tensors();

    /// Parse a precision tier from a string token.
    /// @param s Tier token: "low"/"fp16", "normal"/"mixed", or "high"/"fp32" (the fp16/fp32 forms also
    ///          match their uppercase spelling).
    /// @returns The matching Precision, or Precision::Low for any unrecognized token.
    Precision precisionFromStr(const std::string &s);

} // namespace vknn
