// Load-time autotune-effort tier enum and its string parser.
#pragma once
#include <string>

namespace vknn {

    /// How much load-time conv-kernel autotuning to do. Autotuning happens once at load; the chosen
    /// kernels are stored in the model cache and reused on a warm start, so run() never tunes. Fast
    /// and Heavy may select a kernel whose fp32 summation order differs from the default's (Winograd,
    /// implicit-GEMM, split-K) — fp16-floor equivalent output, not byte-identical; None always keeps
    /// the deterministic default kernels (the byte-gate configuration).
    enum class Tuning {
        None  = 0, ///< No per-shape measurement — the default kernel is used (fastest load, no tuning).
        Fast  = 1, ///< A quick candidate sweep per conv shape (the production default).
        Heavy = 2, ///< An exhaustive sweep (best kernel, slowest load).
    };

    /// Parse a Tuning tier from its string token: "none" / "fast" / "heavy". Legacy aliases from the
    /// former --tuning knob are accepted: "off" -> None, "thorough" -> Heavy.
    /// @param s Tier token; any unrecognized value maps to Fast.
    /// @returns The matching Tuning tier.
    Tuning tuningFromStr(const std::string &s);

} // namespace vknn
