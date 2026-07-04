// Load-time autotune-effort tier enum and its string parser.
#pragma once
#include <string>

namespace vknn {

    // How much load-time conv-kernel autotuning to do (string tokens "none" / "fast" / "heavy"):
    //   None  no per-shape measurement — the default kernel is used (fastest load, no tuning).
    //   Fast  a quick candidate sweep per conv shape (the production default).
    //   Heavy an exhaustive sweep (best kernel, slowest load).
    // Autotuning happens once at load; the chosen kernels are stored in the model cache and reused on a
    // warm start, so run() never tunes. This is effort only — it never changes numerical output.
    enum class Tuning { None = 0, Fast = 1, Heavy = 2 };

    // Tuning tier from a string: "none"/"fast"/"heavy" (unknown -> fast). Legacy aliases from the former
    // --tuning knob are accepted: "off" -> None, "thorough" -> Heavy.
    Tuning tuningFromStr(const std::string &s);

} // namespace vknn
