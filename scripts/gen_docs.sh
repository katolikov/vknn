#!/usr/bin/env bash
# Generate the themed VKNN API docs. Thin wrapper around `./build.sh --docs`.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/build.sh" --docs
