#!/usr/bin/env bash
# Build vxrt for Android arm64-v8a using the NDK CMake toolchain.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${ANDROID_NDK:=$HOME/Library/Android/sdk/ndk/27.0.12077973}"
ABI="arm64-v8a"
API="${ANDROID_API:-33}"
BUILD_DIR="${BUILD_DIR:-build-android}"

if [[ ! -d "$ANDROID_NDK" ]]; then
  echo "ERROR: NDK not found at $ANDROID_NDK (set ANDROID_NDK)" >&2
  exit 1
fi

echo ">> Configuring ($ABI, API $API, NDK $ANDROID_NDK)"
cmake -S . -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="$ABI" \
  -DANDROID_PLATFORM="android-$API" \
  -DCMAKE_BUILD_TYPE=Release \
  -DVXRT_ENABLE_VULKAN=ON \
  -DVXRT_ENABLE_NEON=ON \
  "$@"

echo ">> Building"
cmake --build "$BUILD_DIR" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
echo ">> Build artifacts in $BUILD_DIR:"
find "$BUILD_DIR" -maxdepth 1 -type f \( -name 'vx_*' -o -name '*.a' \) -print
