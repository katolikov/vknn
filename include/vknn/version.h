// The engine's own version.
//
// The single source is the VERSION file at the repo root; CMake reads it, stamps it into
// project(vknn VERSION ...) and compiles it in as VKNN_VERSION_*. Bumping a release is therefore a
// one-line edit to that file -- the Android app reads the same file for its versionName, so the two
// cannot drift apart.
//
// This is the version of the ENGINE, not of whatever embeds it. A host application carries its own
// version stamped at ITS build time, and the two come from separate build steps: an app links a
// prebuilt libvknn/libvknnchat, so a stale native library pairs an old engine with a new app and
// nothing about the app's version reveals it. Reporting both side by side is what makes that
// visible -- they should read the same, and a mismatch names the stale half.
#pragma once

// Fallbacks for a translation unit compiled outside the CMake target (a header-only consumer, an
// IDE index pass). The build always defines these, so a mismatch here never reaches a binary.
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

    /// The engine version as "major.minor.patch". Compiled into the binary that runs, so it reports
    /// the engine actually loaded rather than the one a caller was built against.
    const char *vknnVersion();

} // namespace vknn
