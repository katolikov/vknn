#!/usr/bin/env bash
# Unified build entry point for VKNN.
#
#   ./build.sh                  host build  — CPU backend + IR + ONNX import + tools + tests
#   ./build.sh --android        Android arm64-v8a build (Vulkan backend, NDK toolchain)
#   ./build.sh --clear          wipe the build directory first (clean build)
#   ./build.sh --convert        build only the model compiler (vknn_compile) for the chosen target
#   ./build.sh --docs           build the static documentation site (open docs/site/index.html)
#
# Flags combine, e.g.  ./build.sh --android --clear   or   ./build.sh --clear --convert
# Override the NDK with ANDROID_NDK=..., the API level with ANDROID_API=...
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

android=0 clear=0 convert_only=0 docs=0
for a in "$@"; do
  case "$a" in
    --android) android=1 ;;
    --clear)   clear=1 ;;
    --convert) convert_only=1 ;;
    --docs)    docs=1 ;;
    -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "build.sh: unknown flag '$a' (try --help)" >&2; exit 1 ;;
  esac
done

# --docs: build the static documentation site into docs/site (entry point: docs/site/index.html).
# Doxygen is optional/secondary — if installed, also emit the C++ API reference, linked from the site.
if [[ $docs -eq 1 ]]; then
  if command -v doxygen >/dev/null; then
    echo ">> generating API reference -> docs/api/html (doxygen)"
    doxygen docs/Doxyfile >/dev/null
  else
    echo ">> doxygen not found; skipping the optional API reference"
  fi
  echo ">> building documentation site -> docs/site"
  python3 "$ROOT/scripts/gen_site.py"
  echo ">> open docs/site/index.html"
  exit 0
fi

jobs="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

if [[ $android -eq 1 ]]; then
  : "${ANDROID_NDK:=$HOME/Library/Android/sdk/ndk/27.0.12077973}"
  [[ -d "$ANDROID_NDK" ]] || { echo "ERROR: NDK not found at $ANDROID_NDK (set ANDROID_NDK)" >&2; exit 1; }
  build_dir=build-android
  config=(-G Ninja
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"
    -DANDROID_ABI=arm64-v8a
    -DANDROID_PLATFORM="android-${ANDROID_API:-33}"
    -DCMAKE_BUILD_TYPE=Release
    -DVKNN_ENABLE_VULKAN=ON
    -DVKNN_ENABLE_NEON=ON)
  echo ">> VKNN Android build (arm64-v8a, NDK $ANDROID_NDK)"
else
  build_dir=build-host
  config=(-DCMAKE_BUILD_TYPE=Release)
  echo ">> VKNN host build"
fi

[[ $clear -eq 1 ]] && { echo ">> clean: removing $build_dir"; rm -rf "$build_dir"; }

cmake -S . -B "$build_dir" "${config[@]}" >/dev/null
if [[ $convert_only -eq 1 ]]; then
  echo ">> building model compiler (vknn_compile)"
  cmake --build "$build_dir" -j"$jobs" --target vknn_compile
else
  cmake --build "$build_dir" -j"$jobs"
fi
echo ">> artifacts in $build_dir/"
