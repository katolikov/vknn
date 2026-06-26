#!/usr/bin/env bash
# Build the VKNN documentation site. Thin wrapper around `./build.sh --docs`.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/build.sh" --docs
