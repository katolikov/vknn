/// Engine version, compiled in from the repo-root VERSION file.
#pragma once

// Defaults for a translation unit built outside the vknn CMake target (a header-only consumer, an
// IDE index pass). The target always defines these.
#ifndef VKNN_VERSION_STRING
#define VKNN_VERSION_STRING "0.0.0"
#define VKNN_VERSION_MAJOR  0
#define VKNN_VERSION_MINOR  0
#define VKNN_VERSION_PATCH  0
#endif

namespace vknn {

    constexpr int kVknnVersionMajor = VKNN_VERSION_MAJOR;
    constexpr int kVknnVersionMinor = VKNN_VERSION_MINOR;
    constexpr int kVknnVersionPatch = VKNN_VERSION_PATCH;

    /// Engine version as "major.minor.patch". Compiled into the library, so it reports the engine a
    /// caller has linked rather than the header it was built against; a host that links a prebuilt
    /// library carries its own separate version, and the two are only equal when both are current.
    const char *vknnVersion();

} // namespace vknn
