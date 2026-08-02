#!/usr/bin/env bash
# bench_run.sh — what run() costs on the device, and which stage owns it.
#
# run() is bind + segments + collect, and a segment is pack + submit+gpu + unpack. Those five
# numbers decide what is worth optimizing: host-side work cannot be moved by a kernel change, and
# GPU work cannot be moved by a boundary change. The engine already prints them per run; this turns
# a log into a verdict.
#
# Three disciplines it enforces, because each has produced a wrong answer before:
#   - the FIRST run is discarded. It pays one-time costs (staging allocation, first-touch faults,
#     pipeline warm-up) and reads 10-1000x the steady-state pack cost, which drags any mean.
#   - min AND median are both reported. A min alone hides a distribution with a long tail; a median
#     alone hides a floor the device only reaches when cool. When they disagree, the device is busy
#     or throttling and the number is not a measurement.
#   - the profiled pass is separate and labelled. --profile serializes the pipeline, so a profiled
#     total is NOT latency; it is attribution only, and per-node GPU times OVERLAP, so their sum
#     over-reports. Never quote a profiled number as a runtime.
#
# Usage:
#   benchmark/scripts/bench_run.sh --onnx model.onnx --inputs "a.bin b.bin" [options]
#
#   --onnx PATH        model to compile and run (or --vxm for a prebuilt one)
#   --vxm PATH         skip compilation and run this .vxm
#   --inputs "..."     input .bin/.npy files in the model's declared order (host paths)
#   --serial SER       adb device serial (default: the only attached device)
#   --repeat N         timed runs after the discarded warm-up (default 30)
#   --precision P      low | normal | high (default low — the shipping default)
#   --backend B        vulkan | cpu (default vulkan)
#   --no-ops           skip the per-op-type attribution pass (it runs by default)
#   --flags "..."      extra runner flags for arm A
#   --compare PATH     a second vknn_run_io (e.g. built from main) to run as arm B
#   --compare-flags "..."  run arm B as the SAME binary with these flags instead — the cheapest way
#                      to test an engine decision on a real model, e.g. --compare-flags "--no-flat"
#                      to ask whether the packed path beats the flat one on this graph.
#                      Either form runs the two arms in INTERLEAVED rounds so a drifting device
#                      cannot favour one, and a paired sign test says whether the difference is real
#   --binary PATH      vknn_run_io to use (default build-android/vknn_run_io)
#   --compiler PATH    vknn_compile (default build-host/vknn_compile)
#   --keep             leave the device scratch directory in place
set -uo pipefail

ONNX="" VXM="" INPUTS="" SER="" PRECISION="low" BACKEND="vulkan"
REPEAT=30
PROFILE=1
COMPARE="" FLAGS_A="" FLAGS_B=""
BIN="build-android/vknn_run_io"
COMPILER="build-host/vknn_compile"
WORKDIR="/data/local/tmp/benchrun"
KEEP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx)      ONNX="$2"; shift 2 ;;
    --vxm)       VXM="$2"; shift 2 ;;
    --inputs)    INPUTS="$2"; shift 2 ;;
    --serial)    SER="$2"; shift 2 ;;
    --repeat)    REPEAT="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --backend)   BACKEND="$2"; shift 2 ;;
    --no-ops)    PROFILE=0; shift ;;
    --flags)         FLAGS_A=" $2"; shift 2 ;;
    --compare)       COMPARE="$2"; shift 2 ;;
    --compare-flags) FLAGS_B=" $2"; shift 2 ;;
    --binary)    BIN="$2"; shift 2 ;;
    --compiler)  COMPILER="$2"; shift 2 ;;
    --keep)      KEEP=1; shift ;;
    -h|--help)   sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$ONNX" || -n "$VXM" ]] || { echo "bench_run: --onnx or --vxm is required" >&2; exit 2; }
[[ -x "$BIN" ]] || { echo "bench_run: no runner at $BIN (build it with ./build.sh --android)" >&2; exit 2; }
[[ -z "$COMPARE" || -x "$COMPARE" ]] || { echo "bench_run: no runner at $COMPARE" >&2; exit 2; }

ADB="adb"
[[ -n "$SER" ]] && ADB="adb -s $SER"
$ADB get-state >/dev/null 2>&1 || { echo "bench_run: no device (pass --serial)" >&2; exit 2; }

# A second VKNN process contends for the GPU and, on the CPU arm, for the cores. Timing taken
# alongside one looks tight and is wrong, so refuse rather than measure.
busy=$($ADB shell "ps -A | grep -E 'vknn|run_io|run_bn' | grep -vc grep" </dev/null | tr -d '\r')
[[ "$busy" == "0" ]] || { echo "bench_run: $busy VKNN process(es) already running on the device" >&2; exit 2; }

WORK=$(mktemp -d)
cleanup() {
  rm -rf "$WORK"
  [[ "$KEEP" == "1" ]] || $ADB shell "rm -rf $WORKDIR" >/dev/null 2>&1
}
trap cleanup EXIT

if [[ -z "$VXM" ]]; then
  [[ -x "$COMPILER" ]] || { echo "bench_run: no compiler at $COMPILER" >&2; exit 2; }
  VXM="$WORK/$(basename "${ONNX%.onnx}").vxm"
  echo "compiling $(basename "$ONNX") ..."
  "$COMPILER" "$ONNX" "$VXM" >"$WORK/compile.log" 2>&1 || {
    echo "bench_run: compilation failed:" >&2; tail -20 "$WORK/compile.log" >&2; exit 1; }
fi

$ADB shell "rm -rf $WORKDIR && mkdir -p $WORKDIR" >/dev/null
$ADB push "$BIN" "$WORKDIR/run_bn_a" >/dev/null
if [[ -n "$COMPARE" ]]; then
  $ADB push "$COMPARE" "$WORKDIR/run_bn_b" >/dev/null
elif [[ -n "$FLAGS_B" ]]; then
  # Same build, different engine decision: arm B is a copy of arm A so the two differ only in flags.
  $ADB shell "cp $WORKDIR/run_bn_a $WORKDIR/run_bn_b" >/dev/null
fi
$ADB shell "chmod +x $WORKDIR/run_bn_*" >/dev/null
$ADB push "$VXM" "$WORKDIR/model.vxm" >/dev/null
inNames=""
for f in $INPUTS; do
  [[ -f "$f" ]] || { echo "bench_run: input not found: $f" >&2; exit 2; }
  $ADB push "$f" "$WORKDIR/$(basename "$f")" >/dev/null
  inNames="$inNames $(basename "$f")"
done

# The device's own thermal reading, when it exposes one. Reported rather than waited on: a threshold
# picked from a cooler session is unreachable and turns a benchmark into an idle loop.
deviceTemp() {
  $ADB shell "cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1" </dev/null | tr -d '\r'
}
tempC() { # raw -> Celsius (kernels report milli-C or C)
  local t="${1:-}"
  [[ "$t" =~ ^[0-9]+$ ]] || { echo ""; return; }
  if [[ "$t" -gt 1000 ]]; then echo "$((t / 1000))"; else echo "$t"; fi
}

runArm() { # binaryName logfile extraFlags  -- one process, REPEAT+1 runs, the first discarded
  $ADB shell "cd $WORKDIR && ./$1 model.vxm out --backend $BACKEND --precision $PRECISION --no-cache --timing --repeat $((REPEAT + 1))${3:-}$inNames" \
    </dev/null >"$2" 2>&1
}

# --- collect ------------------------------------------------------------------------------------
tBefore=$(tempC "$(deviceTemp)")
echo "running $((REPEAT + 1)) iteration(s) on $BACKEND, precision $PRECISION ..."
if [[ -n "$COMPARE" || -n "$FLAGS_B" ]]; then
  # Interleaved rounds: a device that drifts mid-benchmark drifts across BOTH arms equally, which a
  # back-to-back A-then-B layout cannot claim.
  : >"$WORK/a.log"; : >"$WORK/b.log"
  for ((r = 0; r < 5; ++r)); do
    runArm run_bn_a "$WORK/a_$r.log" "$FLAGS_A"; cat "$WORK/a_$r.log" >>"$WORK/a.log"
    runArm run_bn_b "$WORK/b_$r.log" "$FLAGS_B"; cat "$WORK/b_$r.log" >>"$WORK/b.log"
  done
else
  runArm run_bn_a "$WORK/a.log" "$FLAGS_A"
fi
tAfter=$(tempC "$(deviceTemp)")

armB=""
[[ -n "$COMPARE" ]] && armB="$(basename "$COMPARE")"
[[ -z "$armB" && -n "$FLAGS_B" ]] && armB="same build,$FLAGS_B"
python3 - "$WORK" "$REPEAT" "$armB" "${tBefore:-}" "${tAfter:-}" "${FLAGS_A:-}" <<'PY'
import sys, os, re
work, repeat, compare, tB, tA = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
flagsA = sys.argv[6] if len(sys.argv) > 6 else ""

STAGE = re.compile(r"timing: pack=([0-9.eE+-]+)ms submit\+gpu=([0-9.eE+-]+)ms unpack=([0-9.eE+-]+)ms")
TOTAL = re.compile(r"sess::run bind=([0-9.eE+-]+)ms segments=([0-9.eE+-]+)ms collect=([0-9.eE+-]+)ms")

def parse(path):
    """Per-run rows, minus the first run of every process: it pays one-time costs (staging
    allocation, first-touch faults, pipeline warm-up) and is not steady state."""
    runs, stages, seenInProcess = [], [], 0
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = STAGE.search(line)
        if m:
            stages.append(tuple(float(x) for x in m.groups()))
            continue
        m = TOTAL.search(line)
        if m:
            bind, seg, coll = (float(x) for x in m.groups())
            runs.append((bind, seg, coll))
    n = min(len(runs), len(stages))
    rows = []
    for i in range(n):
        # A fresh process restarts the warm-up, so drop the first row of each; the logs are
        # concatenated per process and every process opens with its own cold run.
        rows.append(stages[i] + runs[i])
    return rows

def dropColdRuns(rows, perProcess):
    """The concatenated log holds `perProcess` runs per invocation; drop index 0 of each block."""
    out = []
    for i, r in enumerate(rows):
        if perProcess and i % perProcess == 0:
            continue
        out.append(r)
    return out

def stat(v):
    s = sorted(v)
    if not s:
        return (0.0, 0.0, 0.0)
    def q(p):
        return s[min(len(s) - 1, int(p * len(s)))]
    return (s[0], q(0.5), q(0.9))

def report(name, rows):
    pack = [r[0] for r in rows]; gpu = [r[1] for r in rows]; unpack = [r[2] for r in rows]
    bind = [r[3] for r in rows]; seg = [r[4] for r in rows]; coll = [r[5] for r in rows]
    total = [r[3] + r[4] + r[5] for r in rows]
    tMin, tMed, tP90 = stat(total)
    print()
    print("%s — %d timed run(s)" % (name, len(rows)))
    print("  %-14s %9s %9s %9s   %s" % ("stage", "min ms", "median", "p90", "share of median run()"))
    for label, v in (("pack", pack), ("submit+gpu", gpu), ("unpack", unpack),
                     ("bind", bind), ("collect", coll)):
        a, m, p = stat(v)
        share = (m / tMed * 100.0) if tMed > 0 else 0.0
        print("  %-14s %9.3f %9.3f %9.3f   %5.1f%%" % (label, a, m, p, share))
    print("  %-14s %9.3f %9.3f %9.3f   %5.1f%%" % ("run() total", tMin, tMed, tP90, 100.0))
    if tMed > 0 and tMin > 0 and tMed / tMin > 1.5:
        print("  WARNING: median is %.1fx the min — the device is busy or throttling, and this"
              % (tMed / tMin))
        print("           distribution is not a measurement. Let it settle and re-run.")
    gMin, gMed, _ = stat(gpu)
    hostMed = tMed - gMed
    if tMed > 0:
        if gMed / tMed > 0.9:
            print("  => submit+gpu is %.0f%% of run(): the work is on the GPU, and host-side changes"
                  % (gMed / tMed * 100))
            print("     (packing, boundary conversion, allocation) cannot move this number.")
        elif hostMed / tMed > 0.4:
            print("  => %.0f%% of run() is host-side (%.3f ms outside submit+gpu): the boundary, not"
                  % (hostMed / tMed * 100, hostMed))
            print("     the kernels, is what to attack.")
    return (tMin, tMed, total)

perProcess = repeat + 1
def loadArm(tag, label):
    """An arm with no timed runs did not run. Say so and show why, rather than reporting a table of
    zeros -- a benchmark that scores a failed arm as 0.000 ms reads as a result."""
    path = os.path.join(work, tag + ".log")
    rows = dropColdRuns(parse(path), perProcess)
    if rows:
        return rows
    print()
    text = [re.sub(r"\x1b\[[0-9;]*m", "", l.rstrip())
            for l in open(path, encoding="utf-8", errors="ignore")]
    ranAtAll = any("sess::run" in l for l in text)
    if ranAtAll:
        # The model executed but no segment reported pack/submit/unpack, which is what a GPU segment
        # prints. Everything ran somewhere else -- almost always a CPU fallback.
        print("bench_run: arm %s ran, but NO segment reported GPU timing: nothing executed on the" % label)
        print("  GPU, so there is no runtime to compare. Why the planner said so:")
        why = [l for l in text if "fall back to CPU" in l or "falling back" in l]
        for l in (why[:6] or text[-10:]):
            print("    " + l.strip())
    else:
        print("bench_run: arm %s produced NO runs -- it did not execute. Its output:" % label)
        bad = [l for l in text if re.search(r"error|failed|refus|unsupported|abort", l, re.I)]
        for l in (bad[:8] or text[-15:]):
            print("    " + l)
    sys.exit(1)

rowsA = loadArm("a", "A")
if tB or tA:
    print("device thermal reading: %s C before, %s C after" % (tB or "?", tA or "?"))
aStat = report("A%s" % (" (%s)" % flagsA.strip() if flagsA.strip() else ""), rowsA)

if compare:
    rowsB = loadArm("b", "B (%s)" % compare)
    bStat = report("B (%s)" % compare, rowsB)
    print()
    # Paired sign test over the interleaved rounds: the question is not "is A's mean lower" but
    # "does A win more rounds than chance", which a drifting device cannot fake.
    pairs = list(zip(aStat[2], bStat[2]))
    wins = sum(1 for a, b in pairs if a < b)
    n = len(pairs)
    delta = (bStat[1] - aStat[1]) / bStat[1] * 100.0 if bStat[1] > 0 else 0.0
    print("A vs B: median %.3f vs %.3f ms (%+.1f%% for A), A faster in %d of %d paired run(s)"
          % (aStat[1], bStat[1], delta, wins, n))
    if n and (wins / n > 0.9 or wins / n < 0.1):
        print("  the sign test is decisive.")
    else:
        print("  the sign test is NOT decisive — treat the two as equal.")
PY

# --- attribution (separate, and explicitly not latency) ------------------------------------------
if [[ "$PROFILE" == "1" ]]; then
  echo
  echo "per-op attribution (a SEPARATE pass: --profile serializes the pipeline, so these are shares,"
  echo "not latency) ..."
  $ADB shell "cd $WORKDIR && ./run_bn_a model.vxm outp --backend $BACKEND --precision $PRECISION --no-cache --profile --repeat 5$inNames" \
    </dev/null >"$WORK/prof.log" 2>&1
  sed 's/\x1b\[[0-9;]*m//g' "$WORK/prof.log" >"$WORK/prof.txt"
  python3 - "$WORK/prof.txt" <<'PYP'
import sys, re
lines = open(sys.argv[1], encoding="utf-8", errors="ignore").read().splitlines()

# "  <OpType>   <cpu ms> /   <gpu ms> /   <dispatches>" under the "Per op-type" heading.
ROW = re.compile(r"^\s+(\S+)\s+([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*(\d+)\s*$")
SPAN = re.compile(r"elapsed GPU span ([0-9.]+) ms")
rows, span, inSection = [], None, False
for line in lines:
    m = SPAN.search(line)
    if m:
        span = float(m.group(1))
    if line.startswith("Per op-type"):
        inSection = True
        continue
    if inSection:
        m = ROW.match(line)
        if m:
            rows.append((m.group(1), float(m.group(2)), float(m.group(3)), int(m.group(4))))
        elif line.strip() == "":
            inSection = False
if not rows:
    print("  (the profiler printed no per-op-type table)")
    sys.exit(0)

# The op-type gpu column is a SUM of per-node intervals, and those intervals overlap on a GPU that
# runs work concurrently. Shares are taken against that sum -- they are proportions of attributed
# work, which is the question "which op should I improve" actually asks. The elapsed span is the
# only absolute number and is printed beside it so the two are never confused.
attributed = sum(r[2] for r in rows)
rows.sort(key=lambda r: -r[2])
print()
print("  %-24s %10s %8s %12s" % ("op type", "gpu ms", "share", "dispatches"))
for name, cpu, gpu, disp in rows:
    note = "   <- 0 dispatches: elided, this time is attribution only" if disp == 0 else ""
    print("  %-24.24s %10.3f %7.1f%% %12d%s" % (name, gpu, gpu / attributed * 100 if attributed else 0, disp, note))
print("  %-24s %10.3f" % ("attributed sum", attributed))
if span:
    print("  %-24s %10.3f   <- the only absolute figure; the column above overlaps and over-reports"
          % ("elapsed GPU span", span))

real = [r for r in rows if r[3] > 0]
if real:
    print()
    cum, top = 0.0, []
    realTotal = sum(r[2] for r in real)
    for name, _, gpu, _ in real:
        top.append(name)
        cum += gpu
        if realTotal and cum / realTotal >= 0.8:
            break
    print("  => %s %s for %.0f%% of the dispatched work; nothing else moves the total until it does."
          % (", ".join(top), "accounts" if len(top) == 1 else "account",
             cum / realTotal * 100 if realTotal else 0))
phantom = [r[0] for r in rows if r[3] == 0 and r[2] > 0.05 * attributed]
if phantom:
    print("  => %s %s time with ZERO dispatches: the op was elided and the interval is charged"
          % (", ".join(phantom), "carries" if len(phantom) == 1 else "carry"))
    print("     to a neighbour. There is nothing there to optimize.")
PYP
fi
