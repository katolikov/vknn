#!/usr/bin/env bash
# Generate API documentation with Doxygen (HTML in docs/api/html).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if ! command -v doxygen >/dev/null; then
  echo "doxygen not found. Install with: brew install doxygen" >&2
  exit 1
fi
doxygen docs/Doxyfile
echo ">> API docs generated at docs/api/html/index.html"
