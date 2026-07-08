#!/usr/bin/env bash
# gate_pw_probes.sh — device byte gate for producer-epilogue fusion over a fixed probe suite.
#
# Per probe: compile fused + --no-fuse-pointwise (fp32 and fp16), run on device with the pinned
# deterministic config (shared with gate_op.sh via gate_lib.sh), byte-compare each fused output
# against its unfused twin, and fail on any CPU fallback in the fused run. One extra Winograd-on
# case for p_conv3x3w fp16.
#
# WHAT PASS/FAIL MEANS — branch-vs-ref byte-identity, never an absolute score.
#   fused==unfused byte-identity is a per-branch invariant: it says fusion changed no output bytes
#   on this probe, independent of any golden. The count of probes that "pass" is NOT a meaningful
#   grade: the probe generator (make_pw_probes.py) drifts across revisions, so a freshly generated
#   suite can score e.g. ~22/45 at a healthy BASE with no real regression. The gate that means
#   something is THIS branch against a FRESH-BUILT main:
#     * generate probes ONCE (make_pw_probes.py), build main fresh into a ref vknn_run_io, then
#     * run this gate with --ref-binary <that ref>: each fused output is also `cmp`-compared to the
#       ref binary's fused output. branch==ref for every probe = no regression. Do NOT compare
#       against the a1/ref on-device install — it has drifted from main; build main fresh.
#   Without --ref-binary the gate still enforces the per-branch fused==unfused + zero-fallback
#   invariant, which is the correct absolute contract for a single build.
#
# Usage:
#   benchmark/scripts/make_pw_probes.py /tmp/pwprobe                 # generate probe onnx + inputs
#   benchmark/scripts/gate_pw_probes.sh [compile|push|run|all] [probes-dir] [device-dir] [options]
#
# Options (after the three positionals):
#   --device SERIAL      adb serial (default $DEVICE, else the connected device)
#   --ref-binary FILE    host vknn_run_io built from a FRESH main; pushed and cross-compared so the
#                        verdict is branch-vs-ref (no-regression), not just per-branch byte-identity
set -u
HERE="$(dirname "$0")"
cd "$HERE/../.." || { echo "gate_pw_probes.sh: cannot cd to repo root" >&2; exit 1; }
# shellcheck source=benchmark/scripts/gate_lib.sh
. "$HERE/gate_lib.sh"

# Default device: empty — adb targets the single connected device (override with --device).
DEFAULT_DEVICE=""

phase="${1:-all}"
PROBES_DIR=${2:-/tmp/pwprobe}
DEV=${3:-/data/local/tmp/pw/probes}
COMPILE=./build-host/vknn_compile
DEVICE="${DEVICE:-$DEFAULT_DEVICE}"
REF_BINARY=""

# strip the (up to three) positional args, then parse optional flags
shift $(( $# < 3 ? $# : 3 )) 2>/dev/null || true
while [ $# -gt 0 ]; do
  case "$1" in
    --device)     DEVICE="$2"; shift 2 ;;
    --ref-binary) REF_BINARY="$2"; shift 2 ;;
    *) echo "gate_pw_probes.sh: unknown arg '$1'" >&2; exit 2 ;;
  esac
done

ADB="adb${DEVICE:+ -s $DEVICE}"
RUN="../vknn_run_io"      # runner lives one dir above $DEV (pushed to /data/local/tmp/pw)

PROBES="p_conv p_conv1x1 p_dwconv p_convtr p_softmax p_layernorm p_reduce p_gridsample p_resize p_avgpool p_maxpool p_gap p_gemm p_matmul p_conv1x1s2 p_conv1x1deep p_conv3x3w p_softmax_nc4 p_gemm_nobias p_layernorm_nobeta p_reduce_attr p_maxpool_full"

if [ "$phase" = "compile" ] || [ "$phase" = "all" ]; then
  echo "== compile =="
  for p in $PROBES; do
    for cfg in "f32:" "n32:--no-fuse-pointwise" "f16:--fp16" "n16:--fp16 --no-fuse-pointwise"; do
      tag="${cfg%%:*}"; flags="${cfg#*:}"
      # shellcheck disable=SC2086
      out=$($COMPILE $PROBES_DIR/$p.onnx $PROBES_DIR/${p}_$tag.vxm $flags 2>&1)
      fuse=$(echo "$out" | grep -o "fused [0-9]* chain(s), [0-9]* into producer epilogues" | head -1)
      if [ "$tag" = "f32" ]; then echo "$p: ${fuse:-no chains}"; fi
      if echo "$out" | grep -qi "error"; then echo "COMPILE FAIL $p $tag"; echo "$out" | tail -3; fi
    done
  done
fi

if [ "$phase" = "push" ] || [ "$phase" = "all" ]; then
  echo "== push =="
  gate_require_device "$ADB" "$DEVICE" || exit 1
  $ADB shell "mkdir -p $DEV" > /dev/null
  # shellcheck disable=SC2086
  $ADB push $PROBES_DIR/*.vxm $PROBES_DIR/*_in.bin $PROBES_DIR/p_gridsample_grid.bin $DEV/ 2>&1 | tail -1
  $ADB push build-android/vknn_run_io /data/local/tmp/pw/vknn_run_io 2>&1 | tail -1
  if [ -n "$REF_BINARY" ]; then
    [ -f "$REF_BINARY" ] || { echo "--ref-binary not found: $REF_BINARY" >&2; exit 1; }
    $ADB push "$REF_BINARY" /data/local/tmp/pw/ref_run_io 2>&1 | tail -1
    $ADB shell "chmod +x /data/local/tmp/pw/ref_run_io" >/dev/null
  fi
fi

if [ "$phase" = "run" ] || [ "$phase" = "all" ]; then
  echo "== run + compare =="
  gate_require_device "$ADB" "$DEVICE" || exit 1
  [ -n "$REF_BINARY" ] && echo "(branch-vs-ref mode: fused output also compared to $REF_BINARY)"
  REF="../ref_run_io"
  PASS=0; FAIL=0
  for p in $PROBES; do
    ins="${p}_in.bin"
    if [ "$p" = "p_gridsample" ]; then ins="${p}_in.bin p_gridsample_grid.bin"; fi
    for prec in 32 16; do
      pflag="high"; [ "$prec" = "16" ] && pflag="low"
      wflag="off"
      # shellcheck disable=SC2086
      gate_pinned_run "$ADB" "$DEV" "$RUN" "${p}_f$prec.vxm" "o_${p}_f$prec" "$pflag" "$wflag" "l_${p}_f$prec.log" $ins
      # shellcheck disable=SC2086
      gate_pinned_run "$ADB" "$DEV" "$RUN" "${p}_n$prec.vxm" "o_${p}_n$prec" "$pflag" "$wflag" "l_${p}_n$prec.log" $ins
      cok=$(gate_cmp_dirs "$ADB" "$DEV" "o_${p}_f$prec" "o_${p}_n$prec")
      fb=$(gate_fallback_count "$ADB" "$DEV" "l_${p}_f$prec.log")
      refok=1
      if [ -n "$REF_BINARY" ]; then
        # shellcheck disable=SC2086
        gate_pinned_run "$ADB" "$DEV" "$REF" "${p}_f$prec.vxm" "o_${p}_r$prec" "$pflag" "$wflag" "l_${p}_r$prec.log" $ins
        refok=$(gate_cmp_dirs "$ADB" "$DEV" "o_${p}_f$prec" "o_${p}_r$prec")
      fi
      if [ "$cok" = "1" ] && [ "$fb" = "0" ] && [ "$refok" = "1" ]; then
        echo "PASS $p fp$prec"; PASS=$((PASS+1))
      else
        echo "FAIL $p fp$prec  (cmp=$cok fallback=$fb ref=$refok)"; FAIL=$((FAIL+1))
      fi
    done
  done
  # Winograd path: p_conv3x3w fp16 with --winograd on
  gate_pinned_run "$ADB" "$DEV" "$RUN" "p_conv3x3w_f16.vxm" "o_w_f" "low" "on" "l_w_f.log" "p_conv3x3w_in.bin"
  gate_pinned_run "$ADB" "$DEV" "$RUN" "p_conv3x3w_n16.vxm" "o_w_n" "low" "on" "l_w_n.log" "p_conv3x3w_in.bin"
  cok=$(gate_cmp_dirs "$ADB" "$DEV" "o_w_f" "o_w_n")
  fb=$(gate_fallback_count "$ADB" "$DEV" "l_w_f.log")
  refok=1
  if [ -n "$REF_BINARY" ]; then
    gate_pinned_run "$ADB" "$DEV" "$REF" "p_conv3x3w_f16.vxm" "o_w_r" "low" "on" "l_w_r.log" "p_conv3x3w_in.bin"
    refok=$(gate_cmp_dirs "$ADB" "$DEV" "o_w_f" "o_w_r")
  fi
  if [ "$cok" = "1" ] && [ "$fb" = "0" ] && [ "$refok" = "1" ]; then
    echo "PASS p_conv3x3w fp16 winograd-on"; PASS=$((PASS+1))
  else
    echo "FAIL p_conv3x3w fp16 winograd-on (cmp=$cok fallback=$fb ref=$refok)"; FAIL=$((FAIL+1))
  fi
  echo "== $PASS passed, $FAIL failed =="
  [ "$FAIL" = 0 ]
fi
