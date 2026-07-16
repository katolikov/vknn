#!/usr/bin/env bash
# dev_perfab.sh — cooled, interleaved, min-of-N on-device perf A/B for two vknn_run_io binaries.
#
# Scripts the thermal A/B protocol that docs/BENCHMARK.md describes as prose: the device throttles
# 3-5x under sustained load, so back-to-back sweeps and absolute numbers are not trustworthy. This
# runs A and B interleaved (A,B,A,B,...) with a fixed cooldown BEFORE each run, takes the min submit
# wall over N iterations per binary per model, and prints a per-model A/B delta table, flagging any
# model where B is slower than A by more than a threshold.
#
# A MUST be a FRESH-BUILT main (build main into build-android/vknn_run_io, copy it aside), never the
# a1/ref on-device install — that install has DRIFTED from main and would make the delta meaningless.
# B is this branch's vknn_run_io. Both are host files; this script pushes them.
#
# The per-run metric is the "submit+gpu" wall from --timing (src/backend/vulkan/vk_backend.cpp),
# i.e. the real submit+GPU time, not the barrier-inflated per-op profiler sum. Model timing is warm:
# the runner loads the model, runs --repeat to build the command buffer + steady state, and only the
# min over N cooled iterations is kept.
#
# Usage:
#   benchmark/scripts/dev_perfab.sh --a REF_run_io --b BRANCH_run_io --models MODELS.txt [options]
#
# --models FILE : one model per line, each line "vxm_or_onnx in0.bin [in1.bin ...]" (paths on host).
# Options:
#   --a FILE            host vknn_run_io built from FRESH main (the reference)      (required)
#   --b FILE            host vknn_run_io from this branch                           (required)
#   --models FILE       model list (required; see format above)
#   --device SERIAL     adb serial            (default $DEVICE, else the connected device)
#   --n N               iterations per binary per model, min kept   (default 5)
#   --cooldown SEC      idle seconds BEFORE each run                (default 12)
#   --repeat R          --repeat passed to the runner to warm steady state (default 3)
#   --precision P       low|normal|high                            (default low)
#   --winograd W        auto|on|off                                (default auto)
#   --tuning T          none|fast|heavy                            (default none; fast/heavy
#                       re-race per run — untimed at session build, the timed metric is steady state)
#   --threshold PCT     flag B slower than A by more than PCT      (default 3)
#   --dev-dir DIR       on-device working dir     (default /data/local/tmp/pw/perfab)
#   -h, --help          this text
#
# Environment: none. Every knob is a flag.
set -euo pipefail

# Default device: empty — adb targets the single connected device (override with --device).
DEFAULT_DEVICE=""

A="" B="" MODELS="" DEVICE="${DEVICE:-$DEFAULT_DEVICE}"
N=5 COOLDOWN=12 REPEAT=3 PRECISION="low" WINO="auto" TUNING="none" THRESHOLD=3
DEV_DIR="/data/local/tmp/pw/perfab"

usage() { sed -n '2,37p' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --a)         A="$2"; shift 2 ;;
    --b)         B="$2"; shift 2 ;;
    --models)    MODELS="$2"; shift 2 ;;
    --device)    DEVICE="$2"; shift 2 ;;
    --n)         N="$2"; shift 2 ;;
    --cooldown)  COOLDOWN="$2"; shift 2 ;;
    --repeat)    REPEAT="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --winograd)  WINO="$2"; shift 2 ;;
    --tuning)    TUNING="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --dev-dir)   DEV_DIR="$2"; shift 2 ;;
    -h|--help)   usage 0 ;;
    *) echo "dev_perfab.sh: unknown arg '$1'" >&2; usage 1 ;;
  esac
done

cd "$(dirname "$0")/../.." || { echo "dev_perfab.sh: cannot cd to repo root" >&2; exit 1; }
# shellcheck source=benchmark/scripts/gate_lib.sh
. "$(dirname "$0")/gate_lib.sh"

die() { echo "dev_perfab: $*" >&2; exit 1; }

[ -n "$A" ] && [ -f "$A" ] || die "need --a FILE (fresh-main vknn_run_io); not found: $A"
[ -n "$B" ] && [ -f "$B" ] || die "need --b FILE (branch vknn_run_io); not found: $B"
[ -n "$MODELS" ] && [ -f "$MODELS" ] || die "need --models FILE; not found: $MODELS"

ADB="adb${DEVICE:+ -s $DEVICE}"
gate_require_device "$ADB" "$DEVICE" || die "no device"

# --- stage binaries ---
$ADB shell "mkdir -p $DEV_DIR" >/dev/null
$ADB push "$A" "$DEV_DIR/a_run_io" >/dev/null
$ADB push "$B" "$DEV_DIR/b_run_io" >/dev/null
$ADB shell "chmod +x $DEV_DIR/a_run_io $DEV_DIR/b_run_io" >/dev/null

# Push every model + input referenced by the list.
while read -r model rest; do
  [ -z "$model" ] && continue
  case "$model" in \#*) continue ;; esac
  [ -f "$model" ] || die "model not found: $model"
  # </dev/null: adb inside a `while read` loop otherwise swallows the rest of the model list.
  $ADB push "$model" "$DEV_DIR/" >/dev/null </dev/null
  # an ONNX with external weights needs its .data sidecar next to it or every tensor reads as zeros
  [ -f "$model.data" ] && $ADB push "$model.data" "$DEV_DIR/" >/dev/null </dev/null
  for inp in $rest; do
    [ -f "$inp" ] || die "input not found: $inp"
    $ADB push "$inp" "$DEV_DIR/" >/dev/null </dev/null
  done
done < "$MODELS"

# Parse the "submit+gpu=NNNms" from a single run's log; echoes the number (ms) or empty.
parse_submit() { sed -n 's/.*submit+gpu=\([0-9.][0-9.]*\)ms.*/\1/p' | tail -1; }

# One cooled run of one binary on one model -> submit+gpu ms (or "NA").
run_one() { # <a_run_io|b_run_io> <model-basename> <in-basenames> <precflag>
  local bin="$1" mb="$2" ins="$3" pf="$4" log
  # cool BEFORE the run (thermal protocol); ADPF/DVFS settle during idle.
  sleep "$COOLDOWN"
  # </dev/null: adb inside the caller's `while read` loop otherwise swallows the model list.
  log=$($ADB shell "cd $DEV_DIR && ./$bin $mb out_$bin --backend vulkan --precision $pf \
    --winograd $WINO --tuning $TUNING --no-cache --timing --repeat $REPEAT --cache . $ins 2>&1" </dev/null)
  echo "$log" | parse_submit
}

# min of a whitespace-separated list of numbers (awk; empty -> NA).
min_of() { awk 'BEGIN{m=""} {for(i=1;i<=NF;i++){if(m==""||$i<m)m=$i}} END{print (m==""?"NA":m)}'; }

pf="$PRECISION"
printf "\n== dev_perfab: A=%s  B=%s  N=%d  cooldown=%ds  threshold=%s%% ==\n" \
  "$(basename "$A")" "$(basename "$B")" "$N" "$COOLDOWN" "$THRESHOLD"
printf "%-28s %10s %10s %10s  %s\n" "model" "A(min ms)" "B(min ms)" "delta%" "verdict"
printf -- "--------------------------------------------------------------------------\n"

REGRESSED=0
while read -r model rest; do
  [ -z "$model" ] && continue
  case "$model" in \#*) continue ;; esac
  mb=$(basename "$model")
  ins=""
  for inp in $rest; do ins="$ins $(basename "$inp")"; done

  a_times="" b_times=""
  i=0
  while [ "$i" -lt "$N" ]; do
    # interleave A then B each iteration so thermal drift hits both equally
    a=$(run_one a_run_io "$mb" "$ins" "$pf"); [ -n "$a" ] && a_times="$a_times $a"
    b=$(run_one b_run_io "$mb" "$ins" "$pf"); [ -n "$b" ] && b_times="$b_times $b"
    i=$((i+1))
  done
  amin=$(echo "$a_times" | min_of)
  bmin=$(echo "$b_times" | min_of)

  if [ "$amin" = "NA" ] || [ "$bmin" = "NA" ]; then
    printf "%-28s %10s %10s %10s  %s\n" "$mb" "$amin" "$bmin" "-" "NO TIMING"
    continue
  fi
  delta=$(awk -v a="$amin" -v b="$bmin" 'BEGIN{printf "%+.1f", (b-a)/a*100}')
  verdict=$(awk -v a="$amin" -v b="$bmin" -v t="$THRESHOLD" \
    'BEGIN{d=(b-a)/a*100; if(d>t) print "REGRESSION"; else if(d<-t) print "faster"; else print "ok"}')
  [ "$verdict" = "REGRESSION" ] && REGRESSED=$((REGRESSED+1))
  printf "%-28s %10s %10s %10s  %s\n" "$mb" "$amin" "$bmin" "$delta" "$verdict"
done < "$MODELS"

printf -- "--------------------------------------------------------------------------\n"
if [ "$REGRESSED" -gt 0 ]; then
  echo "== $REGRESSED model(s) regressed beyond ${THRESHOLD}% =="
  exit 1
fi
echo "== no regression beyond ${THRESHOLD}% =="
