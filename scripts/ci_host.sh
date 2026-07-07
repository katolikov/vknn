#!/usr/bin/env bash
# ci_host.sh — host-only local CI. The "before you push" gate: everything checkable without a
# device or a GPU, in one command, exiting non-zero on the first hard failure.
#
# Runs, in order:
#   1. ./build.sh                       host build (CPU backend + IR + import + tools + tests)
#   2. ./build-host/vknn_tests          the GoogleTest host unit + integration suite
#   3. ./build.sh --android             arm64 Vulkan build (compiles shaders; catches shader + host
#                                       glue breaks). SKIPPED (non-fatal) if no NDK is installed.
#   4. ./build.sh --docs                the static documentation site
#   5. tools/check_support_consistency.py  op.cpp <-> support-tool self-consistency (no model)
#   6. tools/check_epi_sync.py          epilogue-shader sync check, if present (else skipped)
#   7. clang-format --dry-run -Werror   format drift over include/src/examples/tests (never edits).
#                                       ADVISORY by default — clang-format's include-sort/comment
#                                       behavior changes between versions, so a newer local
#                                       clang-format flags files formatted by an older one. Reports
#                                       drift but does not fail the gate unless --strict-format is
#                                       given (pair that with the clang-format version the tree was
#                                       formatted with). scripts/format.sh fixes drift in place.
#   8. scripts/check_determinism.sh     CPU steady-state byte-identity (skips if the asset is absent)
#
# The on-device gates (gate_op.sh, gate_pw_probes.sh byte gate; dev_perfab.sh perf A/B; the GPU
# liveCount / command-buffer re-record counters) need a device + debug build and are NOT part of this
# host gate — run them separately once the change is device-ready.
#
# Usage:
#   scripts/ci_host.sh [--no-android] [--no-docs] [--clean] [--strict-format]
#
# Options:
#   --no-android     skip the Android build step (e.g. no NDK, or host-logic-only change)
#   --no-docs        skip the docs-site build
#   --clean          pass --clean to the build steps (clean build)
#   --strict-format  make clang-format drift a hard failure (else advisory; see step 7)
#   -h, --help       this text
set -uo pipefail

cd "$(dirname "$0")/.." || { echo "ci_host: cannot cd to repo root" >&2; exit 1; }
ROOT="$PWD"

DO_ANDROID=1 DO_DOCS=1 CLEAN="" STRICT_FORMAT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --no-android)    DO_ANDROID=0; shift ;;
    --no-docs)       DO_DOCS=0; shift ;;
    --clean)         CLEAN="--clean"; shift ;;
    --strict-format) STRICT_FORMAT=1; shift ;;
    -h|--help)       sed -n '2,34p' "$0"; exit 0 ;;
    *) echo "ci_host: unknown arg '$1'" >&2; exit 1 ;;
  esac
done

FAILED=""
step() { # <label> <command...>
  local label="$1"; shift
  printf "\n>>>> %s\n" "$label"
  if "$@"; then
    echo "PASS  $label"
  else
    echo "FAIL  $label"
    FAILED="$FAILED\n  - $label"
  fi
}
skip() { printf "\n>>>> %s\n---- SKIP: %s\n" "$1" "$2"; }

# 1. host build
step "build.sh (host)" ./build.sh $CLEAN

# 2. host tests — only meaningful if the build produced the binary
if [ -x "$ROOT/build-host/vknn_tests" ]; then
  step "vknn_tests" "$ROOT/build-host/vknn_tests"
else
  echo "FAIL  vknn_tests (binary missing — host build failed)"; FAILED="$FAILED\n  - vknn_tests (no binary)"
fi

# 3. android build (needs the NDK; non-fatal skip otherwise)
if [ "$DO_ANDROID" -eq 1 ]; then
  ndk="${ANDROID_NDK:-$HOME/Library/Android/sdk/ndk/27.0.12077973}"
  if [ -d "$ndk" ] && command -v ninja >/dev/null 2>&1; then
    step "build.sh --android" ./build.sh --android $CLEAN
  else
    skip "build.sh --android" "no NDK at $ndk (set ANDROID_NDK) or ninja missing"
  fi
else
  skip "build.sh --android" "--no-android"
fi

# 4. docs site
if [ "$DO_DOCS" -eq 1 ]; then
  step "build.sh --docs" ./build.sh --docs
else
  skip "build.sh --docs" "--no-docs"
fi

# 5. support-tool self-consistency (no model, no device)
step "check_support_consistency.py" python3 "$ROOT/tools/check_support_consistency.py"

# 6. epilogue-shader sync check, if it exists (audit fix #5; not present yet -> skip cleanly)
if [ -f "$ROOT/tools/check_epi_sync.py" ]; then
  step "check_epi_sync.py" python3 "$ROOT/tools/check_epi_sync.py"
else
  skip "check_epi_sync.py" "tool not present in this tree"
fi

# 7. clang-format drift (dry-run, never edits). Advisory unless --strict-format (step-7 note).
if [ -f "$ROOT/.clang-format" ] && command -v clang-format >/dev/null 2>&1; then
  fmt_check() {
    local files bad=0
    files=$(find include src examples tests -type f \
      \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' \) -not -path '*/third_party/*')
    for f in $files; do
      if ! clang-format --dry-run -Werror "$f" >/dev/null 2>&1; then
        echo "  unformatted: $f"; bad=$((bad+1))
      fi
    done
    [ "$bad" -eq 0 ] || { echo "  $bad file(s) differ from clang-format $(clang-format --version | grep -oE '[0-9]+' | head -1) — scripts/format.sh fixes in place"; return 1; }
    return 0
  }
  printf "\n>>>> clang-format drift (advisory)\n"
  if fmt_check; then
    echo "PASS  clang-format drift"
  elif [ "$STRICT_FORMAT" -eq 1 ]; then
    echo "FAIL  clang-format drift (--strict-format)"; FAILED="$FAILED\n  - clang-format drift"
  else
    echo "WARN  clang-format drift (advisory; --strict-format to enforce)"
  fi
else
  skip "clang-format drift" ".clang-format or clang-format missing"
fi

# 8. host determinism (CPU byte-identity; skips if the model asset is gitignored/absent)
step "check_determinism.sh" "$ROOT/scripts/check_determinism.sh"

# ---- verdict ----
printf "\n========================================\n"
if [ -n "$FAILED" ]; then
  printf "ci_host: FAILED steps:%b\n" "$FAILED"
  exit 1
fi
echo "ci_host: ALL PASS"
