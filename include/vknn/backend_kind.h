// Backend selection enum plus its string helpers.
//
// A BackendKind identifies which compute backend runs an op. It is the currency of Config::backend
// (the preferred backend) and Config::fallback (the ordered chain tried when the preferred backend
// cannot support a node), and it tags every registered backend and every planned node.
#pragma once
#include <string>

namespace vknn {

    /// Compute backend that executes ops. Vulkan is the GPU path; Cpu is the reference/fallback path.
    enum class BackendKind {
        Vulkan = 0, ///< GPU backend (Vulkan compute).
        Cpu    = 1, ///< CPU reference backend; also the terminal fallback.
    };

    /// Upper-case, stable name for `k` ("VULKAN" or "CPU"), suitable for logs and config dumps.
    /// @returns A static string literal; "?" if `k` is not a known enumerator. Never nullptr.
    const char *backendName(BackendKind k);

    /// Parse a backend name (case-insensitive for the Vulkan spelling only): "VULKAN"/"vulkan" map to
    /// BackendKind::Vulkan; every other string — including the empty string and unrecognized names —
    /// maps to BackendKind::Cpu.
    BackendKind backendFromStr(const std::string &s);

} // namespace vknn
