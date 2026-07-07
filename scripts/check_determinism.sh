#!/usr/bin/env bash
# check_determinism.sh — host-only steady-state determinism assertion.
#
# Runs a fixed-shape model (default assets/mnasnet1_0.onnx) through the CPU backend TWICE with
# identical inputs and asserts the two runs are byte-identical. A CPU inference is fully
# deterministic, so any drift between runs signals an uninitialized read, a non-deterministic
# reduction order, or state leaking across runs — the cheap host analogue of the on-device
# byte gate. No device needed.
#
# The GPU-side steady-state check (pipeline liveCount / command-buffer re-record counters) needs a
# device and a debug build; it is the on-device extension of this check, not run here.
#
# Usage:
#   scripts/check_determinism.sh [--model M.onnx] [--input IN.bin] [--backend cpu|vulkan]
#
# Options:
#   --model FILE    model to run (default: first found among the candidate asset paths)
#   --input FILE    input .bin  (default: the model's *_in.bin next to it)
#   --backend B     cpu|vulkan  (default cpu; vulkan needs a device and is not the host gate)
#   --run FILE      vknn_run_io binary (default ./build-host/vknn_run_io)
#
# Exit codes: 0 = byte-identical (pass) OR model asset absent (skipped, non-fatal for CI in a
# worktree without the large asset); 1 = runs differ or a run failed.
set -euo pipefail

cd "$(dirname "$0")/.." || { echo "check_determinism: cannot cd to repo root" >&2; exit 1; }
ROOT="$PWD"

MODEL="" INPUT="" BACKEND="cpu" RUN="./build-host/vknn_run_io"
while [ $# -gt 0 ]; do
  case "$1" in
    --model)   MODEL="$2"; shift 2 ;;
    --input)   INPUT="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --run)     RUN="$2"; shift 2 ;;
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
    *) echo "check_determinism: unknown arg '$1'" >&2; exit 1 ;;
  esac
done

# Locate the model: --model, then this tree's assets, then the main worktree's assets (the large
# .onnx assets are gitignored, so a linked worktree usually lacks them — reuse the main checkout's
# copy when running from one). Skip, do not fail, if the asset is nowhere.
if [ -z "$MODEL" ]; then
  cands="$ROOT/assets/mnasnet1_0.onnx"
  # git common dir's parent is the main checkout root; in a plain clone it is $ROOT itself.
  if main_git="$(git rev-parse --git-common-dir 2>/dev/null)"; then
    main_root="$(cd "$(dirname "$main_git")" && pwd)"
    cands="$cands $main_root/assets/mnasnet1_0.onnx"
  fi
  for c in $cands; do
    if [ -f "$c" ]; then MODEL="$c"; break; fi
  done
fi
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
  echo "check_determinism: SKIP — mnasnet1_0.onnx asset not found (gitignored; not in this tree)"
  exit 0
fi

# Default input: the model's *_in.bin next to it.
if [ -z "$INPUT" ]; then
  INPUT="${MODEL%.onnx}_in.bin"
fi
[ -f "$INPUT" ] || { echo "check_determinism: input not found: $INPUT" >&2; exit 1; }
[ -x "$RUN" ] || { echo "check_determinism: runner not found/executable: $RUN (run ./build.sh)" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "check_determinism: $BACKEND x2 on $(basename "$MODEL")"
"$RUN" "$MODEL" "$TMP/a" --backend "$BACKEND" --precision high --no-cache "$INPUT" >"$TMP/a.log" 2>&1 \
  || { echo "check_determinism: FAIL — run 1 errored"; cat "$TMP/a.log"; exit 1; }
"$RUN" "$MODEL" "$TMP/b" --backend "$BACKEND" --precision high --no-cache "$INPUT" >"$TMP/b.log" 2>&1 \
  || { echo "check_determinism: FAIL — run 2 errored"; cat "$TMP/b.log"; exit 1; }

n=0; diff=0
for f in "$TMP"/a/*; do
  [ -e "$f" ] || continue
  bn="$(basename "$f")"
  n=$((n+1))
  if ! cmp -s "$f" "$TMP/b/$bn"; then
    echo "check_determinism: FAIL — output '$bn' differs between runs"; diff=1
  fi
done
if [ "$n" -eq 0 ]; then
  echo "check_determinism: FAIL — no outputs produced"; exit 1
fi
if [ "$diff" -ne 0 ]; then exit 1; fi
echo "check_determinism: PASS — $n output(s) byte-identical across two $BACKEND runs"
