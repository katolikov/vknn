#!/usr/bin/env bash
# gate_op.sh — reusable per-op DEVICE byte gate.
#
# Given a probe ONNX (one op or a small op chain) plus its input .bin files, this compiles the model
# two ways — fused and --no-fuse-pointwise — for a precision, pushes both to the device, runs each
# with a pinned deterministic config, and asserts:
#   1. byte-identity: every fused output file is `cmp`-identical to its unfused twin, and
#   2. zero CPU fallback: the fused run's log contains no "falling back" line
#      (the Session prints "... -> falling back to <backend>" for any op with no GPU kernel;
#       see src/core/session.cpp). A future `vknn_compile --support-report` will make this a
#       structured check — until it exists, the run-log grep is the fallback detector.
#
# The gate is a BRANCH-vs-REF check, never an absolute score: fused==unfused byte-identity means
# "fusion introduced no numeric change on this op", independent of any golden. To also confirm the
# branch did not regress vs main, pass --ref-binary <run_io-built-from-fresh-main> and the same
# probe is run through both binaries and their fused outputs `cmp`-compared (see gate_pw_probes.sh,
# which layers a whole probe suite on top of this).
#
# Usage:
#   benchmark/scripts/gate_op.sh --onnx PROBE.onnx --inputs "a.bin b.bin" [options]
#
# Options:
#   --onnx FILE          probe model (required unless --builder is given)
#   --builder "CMD"      shell command that WRITES the probe onnx to the path in $GATE_ONNX and its
#                        inputs next to it (an op_test.py-style generator); run before compiling
#   --inputs "F1 F2"     space-separated input .bin files (in model input order); default none
#   --name NAME          label for output/log dirs on device (default: onnx basename)
#   --prec low|high      run precision: low=fp16, high=fp32 (default low; pass twice via --prec-both)
#   --prec-both          run BOTH fp16 and fp32 (default: only --prec)
#   --winograd off|on|auto   forced conv kernel for the run (default off; deterministic)
#   --run-binary PATH    device path of vknn_run_io to test    (default /data/local/tmp/pw/vknn_run_io)
#   --ref-binary FILE    host path of a fresh-main vknn_run_io; pushed and cross-compared for
#                        no-regression (optional; without it the gate is fused-vs-unfused only)
#   --compile PATH       host vknn_compile                     (default ./build-host/vknn_compile)
#   --device SERIAL      adb serial to target        (default $DEVICE, else the connected device)
#   --dev-dir DIR        on-device working directory  (default /data/local/tmp/pw/gate_op)
#   --keep               keep the on-device work dir (default: leave it, re-used next run)
#   -h, --help           this text
#
# Environment: none. Every knob is a flag. adb must see exactly the requested device.
set -euo pipefail

# Default device: empty — adb targets the single connected device (override with --device).
DEFAULT_DEVICE=""

ONNX="" BUILDER="" INPUTS="" NAME="" PREC="low" PREC_BOTH=0 WINO="off"
RUN_BINARY="/data/local/tmp/pw/vknn_run_io" REF_BINARY=""
COMPILE="./build-host/vknn_compile" DEVICE="${DEVICE:-$DEFAULT_DEVICE}"
DEV_DIR="/data/local/tmp/pw/gate_op" KEEP=0

usage() { sed -n '2,40p' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --onnx)       ONNX="$2"; shift 2 ;;
    --builder)    BUILDER="$2"; shift 2 ;;
    --inputs)     INPUTS="$2"; shift 2 ;;
    --name)       NAME="$2"; shift 2 ;;
    --prec)       PREC="$2"; shift 2 ;;
    --prec-both)  PREC_BOTH=1; shift ;;
    --winograd)   WINO="$2"; shift 2 ;;
    --run-binary) RUN_BINARY="$2"; shift 2 ;;
    --ref-binary) REF_BINARY="$2"; shift 2 ;;
    --compile)    COMPILE="$2"; shift 2 ;;
    --device)     DEVICE="$2"; shift 2 ;;
    --dev-dir)    DEV_DIR="$2"; shift 2 ;;
    --keep)       KEEP=1; shift ;;
    -h|--help)    usage 0 ;;
    *) echo "gate_op.sh: unknown arg '$1'" >&2; usage 1 ;;
  esac
done

HERE="$(dirname "$0")"
cd "$HERE/../.." || { echo "gate_op.sh: cannot cd to repo root" >&2; exit 1; }
# shellcheck source=benchmark/scripts/gate_lib.sh
. "$HERE/gate_lib.sh"

ADB="adb${DEVICE:+ -s $DEVICE}"

fail() { echo "GATE FAIL: $*" >&2; exit 1; }

# --- resolve the probe ---
if [ -n "$BUILDER" ]; then
  : "${ONNX:=/tmp/gate_op_probe.onnx}"
  GATE_ONNX="$ONNX" sh -c "$BUILDER" || fail "builder command failed"
fi
[ -n "$ONNX" ] || fail "need --onnx FILE (or --builder that writes \$GATE_ONNX)"
[ -f "$ONNX" ] || fail "probe onnx not found: $ONNX"
[ -n "$NAME" ] || NAME="$(basename "$ONNX" .onnx)"
[ -x "$COMPILE" ] || fail "vknn_compile not found/executable: $COMPILE (run ./build.sh)"

# --- device reachability: fail gracefully, do not hang ---
gate_require_device "$ADB" "$DEVICE" || fail "no device"

precisions="$PREC"
[ "$PREC_BOTH" = 1 ] && precisions="low high"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "== gate_op: $NAME  (device $DEVICE, winograd $WINO) =="

# --- compile fused + unfused per precision ---
compile_one() { # <tag> <fp16-flag> <extra-flags>
  local tag="$1" fp16="$2" extra="$3" out
  out="$WORK/${NAME}_${tag}.vxm"
  # shellcheck disable=SC2086
  if ! "$COMPILE" "$ONNX" "$out" $fp16 $extra >"$WORK/c_${tag}.log" 2>&1; then
    cat "$WORK/c_${tag}.log" >&2; fail "compile failed ($tag)"
  fi
  grep -qi "error" "$WORK/c_${tag}.log" && { cat "$WORK/c_${tag}.log" >&2; fail "compile error ($tag)"; }
}

for prec in $precisions; do
  fp16=""; [ "$prec" = "low" ] && fp16="--fp16"
  compile_one "f_$prec" "$fp16" ""
  compile_one "n_$prec" "$fp16" "--no-fuse-pointwise"
done

# --- push probe artifacts + inputs ---
$ADB shell "mkdir -p $DEV_DIR" >/dev/null
# shellcheck disable=SC2086
$ADB push $WORK/*.vxm "$DEV_DIR/" >/dev/null
for f in $INPUTS; do
  [ -f "$f" ] || fail "input file not found: $f"
  $ADB push "$f" "$DEV_DIR/" >/dev/null
done
in_names=""
for f in $INPUTS; do in_names="$in_names $(basename "$f")"; done

if [ -n "$REF_BINARY" ]; then
  [ -f "$REF_BINARY" ] || fail "--ref-binary not found: $REF_BINARY"
  $ADB push "$REF_BINARY" "$DEV_DIR/ref_run_io" >/dev/null
  $ADB shell "chmod +x $DEV_DIR/ref_run_io" >/dev/null
fi

# --- run + compare, per precision (shared helpers in gate_lib.sh) ---
PASS=0; FAIL=0
for prec in $precisions; do
  pf="high"; [ "$prec" = "low" ] && pf="low"
  fd="of_${NAME}_$prec" nd="on_${NAME}_$prec"
  # shellcheck disable=SC2086
  gate_pinned_run "$ADB" "$DEV_DIR" "$RUN_BINARY" "${NAME}_f_$prec.vxm" "$fd" "$pf" "$WINO" "lf_$prec.log" $in_names
  # shellcheck disable=SC2086
  gate_pinned_run "$ADB" "$DEV_DIR" "$RUN_BINARY" "${NAME}_n_$prec.vxm" "$nd" "$pf" "$WINO" "ln_$prec.log" $in_names

  cok=$(gate_cmp_dirs "$ADB" "$DEV_DIR" "$fd" "$nd")
  fb=$(gate_fallback_count "$ADB" "$DEV_DIR" "lf_$prec.log")
  if [ "$cok" = "1" ] && [ "$fb" = "0" ]; then
    echo "PASS  $NAME fp16/32=$prec  fused==unfused, 0 fallback"; PASS=$((PASS+1))
  else
    echo "FAIL  $NAME fp16/32=$prec  (cmp=$cok fallback=$fb)"; FAIL=$((FAIL+1))
  fi

  # optional no-regression: this branch's fused output vs a fresh-main run of the same fused vxm
  if [ -n "$REF_BINARY" ]; then
    rd="or_${NAME}_$prec"
    # shellcheck disable=SC2086
    gate_pinned_run "$ADB" "$DEV_DIR" "ref_run_io" "${NAME}_f_$prec.vxm" "$rd" "$pf" "$WINO" "lr_$prec.log" $in_names
    rok=$(gate_cmp_dirs "$ADB" "$DEV_DIR" "$fd" "$rd")
    if [ "$rok" = "1" ]; then
      echo "PASS  $NAME fp16/32=$prec  branch==ref (no regression)"; PASS=$((PASS+1))
    else
      echo "FAIL  $NAME fp16/32=$prec  branch != ref binary (regression)"; FAIL=$((FAIL+1))
    fi
  fi
done

[ "$KEEP" = 1 ] || $ADB shell "rm -rf $DEV_DIR" >/dev/null 2>&1 || true

echo "== gate_op $NAME: $PASS passed, $FAIL failed =="
[ "$FAIL" = 0 ]
