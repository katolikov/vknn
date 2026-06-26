#!/usr/bin/env bash
# Back-compat shim — the build entry point is now ./build.sh (see ./build.sh --help).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/build.sh" --android "$@"
