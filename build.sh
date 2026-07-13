#!/usr/bin/env bash
# Unified build entry point for VKNN.
#
#   ./build.sh                  host build  — CPU backend + IR + ONNX import + tools + tests
#   ./build.sh --android        Android arm64-v8a build (Vulkan backend, NDK toolchain)
#   ./build.sh --clean          remove build-host + build-android and stop (clean only, no build)
#   ./build.sh --convert        build only the model compiler (vknn_compile) for the chosen target
#   ./build.sh --docs           build the static documentation site (open docs/site/index.html)
#   ./build.sh --leakcheck      instrument the whole codebase and run the tests under memory-leak
#                               detection (Linux: ASan+LeakSanitizer+UBSan; macOS: the `leaks` tool)
#
# --clean alone just cleans and exits. Combined with a build flag it cleans that target's dir first,
# then builds — e.g.  ./build.sh --android --clean   or   ./build.sh --clean --convert
# Override the NDK with ANDROID_NDK=..., the API level with ANDROID_API=...
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

android=0 clean=0 convert_only=0 docs=0 leakcheck=0
for a in "$@"; do
  case "$a" in
    --android)   android=1 ;;
    --clean)     clean=1 ;;
    --convert)   convert_only=1 ;;
    --docs)      docs=1 ;;
    --leakcheck) leakcheck=1 ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "build.sh: unknown flag '$a' (try --help)" >&2; exit 1 ;;
  esac
done

if [[ $clean -eq 1 && $android -eq 0 && $convert_only -eq 0 && $docs -eq 0 ]]; then
  echo ">> clean: removing build-host and build-android"
  rm -rf build-host build-android
  exit 0
fi

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

# --leakcheck: instrument the whole host codebase and run the unit tests under memory-leak detection.
# Linux: AddressSanitizer + LeakSanitizer + UBSan (leaks AND use-after-free / overflow / UB). macOS has
# no LeakSanitizer, so build with debug info and use the built-in `leaks` tool for leak detection; for
# use-after-free / UB on macOS, configure a build with -DVKNN_SANITIZE=ON. `leaks` reports only
# unreachable allocations (genuine leaks), not intentionally-retained singletons.
if [[ $leakcheck -eq 1 ]]; then
  build_dir=build-host-leakcheck
  [[ $clean -eq 1 ]] && { echo ">> clean: removing $build_dir"; rm -rf "$build_dir"; }
  if [[ "$(uname)" == "Darwin" ]]; then
    command -v leaks >/dev/null || { echo "ERROR: the 'leaks' tool is not installed (Xcode command-line tools)" >&2; exit 1; }
    echo ">> VKNN leak-check (macOS): host build + the 'leaks' tool over vknn_tests"
    cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
    cmake --build "$build_dir" -j"$jobs" --target vknn_tests
    echo ">> running vknn_tests under 'leaks --atExit' (leaks across all unit tests)"
    if MallocStackLogging=1 leaks --atExit -- "$build_dir/vknn_tests" >leakcheck.log 2>&1; then
      echo ">> NO LEAKS (full log: leakcheck.log)"
    else
      echo ">> LEAKS FOUND — summary (full report + call stacks in leakcheck.log):"
      grep -E 'leaks for [0-9]|[0-9]+ leaks? for|total leaked|Leak: ' leakcheck.log | head -40 || true
    fi
    echo ">> (for use-after-free / UB on macOS, add -DVKNN_SANITIZE=ON to a host cmake configure)"
  else
    echo ">> VKNN leak-check (Linux): AddressSanitizer + LeakSanitizer + UndefinedBehaviorSanitizer"
    cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVKNN_SANITIZE=ON >/dev/null
    cmake --build "$build_dir" -j"$jobs"
    echo ">> running vknn_tests under ASan/LSan/UBSan (leaks + memory errors reported at exit)"
    ASAN_OPTIONS=detect_leaks=1:print_stats=1 UBSAN_OPTIONS=print_stacktrace=1 "$build_dir/vknn_tests" || true
  fi
  echo ">> leak-check done"
  exit 0
fi

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

[[ $clean -eq 1 ]] && { echo ">> clean: removing $build_dir"; rm -rf "$build_dir"; }

# Vendored shader compiler: build glslang (third_party/glslang) for the host once. It is a build-time
# tool (GLSL -> SPIR-V on this machine, not the device), so it is always built natively even for the
# Android target; CMake then prefers it over a system glslc. Only the Vulkan (Android) build compiles
# shaders, so skip it for the CPU-only host build. If the submodule is absent, CMake falls back to glslc.
glslang_dir="$ROOT/third_party/glslang"
glslang_bin="$glslang_dir/build-host/StandAlone/glslang"
if [[ $android -eq 1 && -f "$glslang_dir/CMakeLists.txt" && ! -x "$glslang_bin" ]]; then
  echo ">> building vendored shader compiler (glslang, host)"
  cmake -S "$glslang_dir" -B "$glslang_dir/build-host" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_OPT=OFF -DGLSLANG_TESTS=OFF -DBUILD_EXTERNAL=OFF >/dev/null
  cmake --build "$glslang_dir/build-host" -j"$jobs" --target glslang-standalone >/dev/null
fi

cmake -S . -B "$build_dir" "${config[@]}" >/dev/null
if [[ $convert_only -eq 1 ]]; then
  echo ">> building model compiler (vknn_compile)"
  cmake --build "$build_dir" -j"$jobs" --target vknn_compile
else
  cmake --build "$build_dir" -j"$jobs"
fi
echo ">> artifacts in $build_dir/"
