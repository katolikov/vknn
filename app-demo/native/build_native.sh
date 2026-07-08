#!/usr/bin/env bash
# Build libvknnchat.so (JNI decoder) for arm64-v8a and drop it into the app's jniLibs. Uses the NDK
# toolchain and a modern cmake (homebrew) so the WHOLE_ARCHIVE generator expression is available.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"                       # repo root
NDK="${ANDROID_NDK:-$HOME/Library/Android/sdk/ndk/27.0.12077973}"
CMAKE="${CMAKE:-cmake}"                                  # homebrew cmake >= 3.24
ABI=arm64-v8a
API=33
BUILD="$HERE/build"
JNILIBS="$HERE/../app/src/main/jniLibs/$ABI"

[[ -f "$NDK/build/cmake/android.toolchain.cmake" ]] || { echo "NDK not found at $NDK (set ANDROID_NDK)"; exit 1; }

"$CMAKE" -S "$HERE" -B "$BUILD" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="$ABI" -DANDROID_PLATFORM="android-$API" -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE=Release -DVKNN_ROOT="$ROOT"
"$CMAKE" --build "$BUILD" -j

mkdir -p "$JNILIBS"
cp "$BUILD/libvknnchat.so" "$JNILIBS/"
echo "OK -> $JNILIBS/libvknnchat.so ($(du -h "$JNILIBS/libvknnchat.so" | cut -f1))"
