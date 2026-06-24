#!/usr/bin/env bash
# Format all vxrt C/C++ sources with clang-format (.clang-format at repo root).
# Skips third_party and generated code.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v clang-format >/dev/null || { echo "clang-format not found (brew install clang-format)" >&2; exit 1; }
files=$(find include src examples tests -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' \) \
  -not -path '*/third_party/*')
echo "$files" | xargs clang-format -i
echo ">> formatted $(echo "$files" | wc -l | tr -d ' ') files"
