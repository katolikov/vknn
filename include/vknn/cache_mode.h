// Warm-start cache scope enum plus its string helpers.
#pragma once
#include <string>

namespace vknn {

    // What a warm start reloads from the unified cache file, in increasing order. Off recomputes
    // everything every load; Tune keeps the cheap, deterministic blobs (compiled pipelines + the
    // conv autotune table) but recomputes/re-uploads weights; Full also keeps the prepacked-weight
    // blob for the fastest warm load (and the largest cache file).
    enum class CacheMode { Off = 0, Tune = 1, Full = 2 };

    // Cache scope: "off" / "tune" / "full" -> CacheMode::Off/Tune/Full (and back).
    CacheMode   cacheModeFromStr(const std::string &s);
    const char *cacheModeStr(CacheMode m);

} // namespace vknn
