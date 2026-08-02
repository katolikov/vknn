#!/usr/bin/env bash
# gate_cpu_gpu.sh — run one model on the device's GPU and on its CPU, and compare the outputs.
#
# The CPU backend is the engine's numeric oracle: it is a complete, fp32 implementation of every op,
# so a GPU result that disagrees with it points at a GPU kernel, a layout, or a precision decision --
# not at the model. This gate makes that comparison a single command, on the device, for any ONNX.
#
# It answers the question a golden cannot: a golden tells you the answer moved, this tells you WHICH
# SIDE moved. Run it on two builds and the four numbers separate the cases -- if CPU-vs-GPU is clean
# on both, the difference lives in something both backends share (an import pass, the model itself);
# if it degrades on one build, that build's GPU path is where to look.
#
# Usage:
#   benchmark/scripts/gate_cpu_gpu.sh --onnx model.onnx --inputs "a.bin b.bin" [options]
#
#   --onnx PATH         model to compile and run (required)
#   --inputs "..."      input .bin files in the model's declared order, space-separated. Paths are
#                       host paths; they are pushed next to the model. Omit only for a model that
#                       declares no inputs.
#   --precision P       low | normal | high   (default low — the shipping default)
#   --serial SER        adb device serial (default: the only attached device)
#   --binary PATH       vknn_run_io to use (default build-android/vknn_run_io)
#   --compiler PATH     vknn_compile to use (default build-host/vknn_compile)
#   --vxm PATH          skip compilation and use this prebuilt .vxm instead of --onnx
#   --workdir DIR       device scratch directory (default /data/local/tmp/cpugpu)
#   --keep              leave the device scratch directory in place when finished
#   --localize          on a disagreement, dump every activation on both backends and name the
#                       FIRST tensor that diverges in node order -- everything downstream of it
#                       differs for free, so that one name is the whole answer
#   --rel PCT           what counts as diverged, as a percentage of the tensor's own range
#                       (default 1.0). fp16 carries ~0.05% per rounding and a few ops accumulate to
#                       a few tenths, so a tighter bar names rounding rather than a defect.
#   --no-fold-islands   keep tiny GPU op-islands on the GPU. A few-node probe is folded to the CPU
#                       by default because the round trip costs more than the work, and then the
#                       "GPU" arm IS the CPU arm and agreement means nothing. Pass this for probes.
#
# The GPU arm's node coverage is reported either way: a run whose nodes all fell back to the CPU is
# called out rather than scored, because a gate that grades the oracle against itself always passes.
#
# Metrics per output: max absolute difference, PSNR, SNR, cosine similarity, and the share of
# elements that differ at all. Element width is derived from the printed shape and the file size, so
# uint8, fp16 and fp32 outputs are all read correctly without being told which they are.
set -uo pipefail

ONNX="" INPUTS="" PRECISION="low" SER="" VXM=""
BIN="build-android/vknn_run_io"
COMPILER="build-host/vknn_compile"
WORKDIR="/data/local/tmp/cpugpu"
KEEP=0
LOCALIZE=0
REL_PCT="1.0"
FOLD_FLAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx)      ONNX="$2"; shift 2 ;;
    --inputs)    INPUTS="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --serial)    SER="$2"; shift 2 ;;
    --binary)    BIN="$2"; shift 2 ;;
    --compiler)  COMPILER="$2"; shift 2 ;;
    --vxm)       VXM="$2"; shift 2 ;;
    --workdir)   WORKDIR="$2"; shift 2 ;;
    --keep)      KEEP=1; shift ;;
    --localize)  LOCALIZE=1; shift ;;
    --rel)       REL_PCT="$2"; shift 2 ;;
    --no-fold-islands) FOLD_FLAG=" --no-fold-islands"; shift ;;
    -h|--help)   sed -n '2,42p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$ONNX" || -n "$VXM" ]] || { echo "gate_cpu_gpu: --onnx or --vxm is required" >&2; exit 2; }
[[ -x "$BIN" ]] || { echo "gate_cpu_gpu: no runner at $BIN (build it with ./build.sh --android)" >&2; exit 2; }

ADB="adb"
[[ -n "$SER" ]] && ADB="adb -s $SER"
$ADB get-state >/dev/null 2>&1 || { echo "gate_cpu_gpu: no device (pass --serial)" >&2; exit 2; }

# A second VKNN process on the device would contend for the GPU and, on the CPU arm, for the cores.
# The comparison is numeric rather than timed so contention cannot change the verdict, but a
# half-finished run from an earlier gate can, so refuse to start on top of one.
busy=$($ADB shell "ps -A | grep -E 'vknn|run_io|run_cg' | grep -vc grep" </dev/null | tr -d '\r')
[[ "$busy" == "0" ]] || { echo "gate_cpu_gpu: $busy VKNN process(es) already running on the device" >&2; exit 2; }

WORK=$(mktemp -d)
cleanup() {
  rm -rf "$WORK"
  [[ "$KEEP" == "1" ]] || $ADB shell "rm -rf $WORKDIR" >/dev/null 2>&1
}
trap cleanup EXIT

# --- compile ------------------------------------------------------------------------------------
if [[ -z "$VXM" ]]; then
  [[ -x "$COMPILER" ]] || { echo "gate_cpu_gpu: no compiler at $COMPILER (build it with ./build.sh)" >&2; exit 2; }
  VXM="$WORK/$(basename "${ONNX%.onnx}").vxm"
  echo "compiling $(basename "$ONNX") ..."
  "$COMPILER" "$ONNX" "$VXM" >"$WORK/compile.log" 2>&1 || {
    echo "gate_cpu_gpu: compilation failed:" >&2; tail -20 "$WORK/compile.log" >&2; exit 1; }
fi

# --- push ---------------------------------------------------------------------------------------
$ADB shell "rm -rf $WORKDIR && mkdir -p $WORKDIR" >/dev/null
$ADB push "$BIN" "$WORKDIR/run_cg" >/dev/null
$ADB shell "chmod +x $WORKDIR/run_cg" >/dev/null
$ADB push "$VXM" "$WORKDIR/model.vxm" >/dev/null
inNames=""
for f in $INPUTS; do
  [[ -f "$f" ]] || { echo "gate_cpu_gpu: input not found: $f" >&2; exit 2; }
  $ADB push "$f" "$WORKDIR/$(basename "$f")" >/dev/null
  inNames="$inNames $(basename "$f")"
done

# --- run both backends --------------------------------------------------------------------------
# --no-cache on both: a cached autotune pick is per-device state, and the point here is what THIS
# build computes, not what an earlier one left behind.
run_arm() { # backend outdir logfile
  $ADB shell "cd $WORKDIR && rm -rf $2 && ./run_cg model.vxm $2 --backend $1 --precision $PRECISION --no-cache$FOLD_FLAG$inNames" \
    </dev/null >"$3" 2>&1
}
echo "running GPU (vulkan, precision $PRECISION) ..."
run_arm vulkan out_gpu "$WORK/gpu.log"
echo "running CPU (fp32 oracle) ..."
run_arm cpu out_cpu "$WORK/cpu.log"

for arm in gpu cpu; do
  n=$($ADB shell "ls $WORKDIR/out_$arm 2>/dev/null | wc -l" </dev/null | tr -d '\r')
  if [[ "$n" == "0" ]]; then
    echo "gate_cpu_gpu: the $arm arm produced no outputs; its log:" >&2
    sed 's/\x1b\[[0-9;]*m//g' "$WORK/$arm.log" | tail -25 >&2
    exit 1
  fi
done
# How much of the graph the GPU arm actually ran on the GPU. A node the planner sent to the CPU is
# compared against itself, so a run that placed EVERY node on the CPU agrees perfectly while proving
# nothing -- the failure mode a gate must never report as a pass. The Session announces both numbers
# ("Planned N segment(s) over M nodes", "K node(s) fall back to CPU"), summed here over buckets.
plain=$(sed 's/\x1b\[[0-9;]*m//g' "$WORK/gpu.log")
nodes=$(echo "$plain" | grep -oE "Planned [0-9]+ segment\(s\) over [0-9]+ nodes" | grep -oE "over [0-9]+" | awk '{s+=$2} END{print s+0}')
onCpu=$(echo "$plain" | grep -oE "[0-9]+ node\(s\) fall back to CPU" | awk '{s+=$1} END{print s+0}')
if [[ "$nodes" -gt 0 && "$onCpu" -ge "$nodes" ]]; then
  echo
  echo "gate_cpu_gpu: the GPU arm ran ALL $nodes node(s) on the CPU -- this would compare the oracle"
  echo "  against itself and agree no matter what the GPU does. Reason the planner gave:"
  echo "$plain" | grep -oE "fall back to CPU.*" | head -1 | sed 's/^/    /'
  echo "  A few-node graph is folded to the CPU on purpose (the round trip costs more than the work);"
  echo "  pass --no-fold-islands to keep it on the GPU."
  exit 3
fi
[[ "$onCpu" == "0" ]] || echo "note: $onCpu of $nodes node(s) ran on the CPU in the GPU arm; those are compared against themselves"

$ADB pull "$WORKDIR/out_gpu" "$WORK/gpu" >/dev/null 2>&1
$ADB pull "$WORKDIR/out_cpu" "$WORK/cpu" >/dev/null 2>&1

# --- compare ------------------------------------------------------------------------------------
# The element width comes from the run log's declared shape and the file size, so a uint8 image, an
# fp16 tensor and an fp32 tensor are each read as themselves without a dtype flag.
sed 's/\x1b\[[0-9;]*m//g' "$WORK/gpu.log" | grep -oE "output +'[^']+' +\[[0-9,]+\]" > "$WORK/shapes.txt" || true
cat >"$WORK/compare.py" <<'PY'
import sys, os, re, glob
import numpy as np

work = sys.argv[1]
shapes = {}
for line in open(os.path.join(work, "shapes.txt"), encoding="utf-8", errors="ignore"):
    m = re.search(r"output +'([^']+)' +\[([0-9,]+)\]", line)
    if m:
        shapes[m.group(1)] = [int(x) for x in m.group(2).split(",") if x != ""]

def load(path, name):
    """Read an output file as float64, deriving the element width from shape and file size."""
    size = os.path.getsize(path)
    dims = shapes.get(name)
    elems = int(np.prod(dims)) if dims else 0
    width = size // elems if elems and size % elems == 0 else 0
    dtype = {1: np.uint8, 2: np.float16, 4: np.float32}.get(width)
    if dtype is None:                      # unknown width: fall back to fp32 if it divides evenly
        dtype, width = (np.float32, 4) if size % 4 == 0 else (np.uint8, 1)
    return np.fromfile(path, dtype=dtype).astype(np.float64), width

rows, worst = [], None
for p in sorted(glob.glob(os.path.join(work, "gpu", "**", "*.bin"), recursive=True)):
    name = os.path.splitext(os.path.basename(p))[0]
    q = None
    for cand in glob.glob(os.path.join(work, "cpu", "**", os.path.basename(p)), recursive=True):
        q = cand
        break
    if q is None:
        rows.append((name, "MISSING on the CPU arm", None)); continue
    a, width = load(p, name)
    b, _     = load(q, name)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    d = np.abs(a - b)
    peak = 255.0 if width == 1 else max(float(np.abs(b).max()), 1e-12)
    mse = float((d ** 2).mean())
    psnr = float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)
    sig = float((b ** 2).mean())
    snr = float("inf") if mse == 0 else 10 * np.log10(sig / mse) if sig > 0 else float("nan")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a.dot(b) / (na * nb)) if na > 0 and nb > 0 else float("nan")
    rows.append((name, None, (n, width, float(d.max()), psnr, snr, cos, float((d > 0).mean() * 100))))
    worst = psnr if worst is None else min(worst, psnr)

print()
print("%-24s %9s %5s %10s %9s %9s %10s %8s" % ("output", "elements", "bytes", "max|diff|", "PSNR dB", "SNR dB", "cosine", "differ%"))
print("-" * 92)
for name, err, r in rows:
    if err:
        print("%-24s %s" % (name, err)); continue
    n, w, mx, psnr, snr, cos, pct = r
    print("%-24s %9d %5d %10.5g %9.2f %9.2f %10.7f %7.1f%%" % (name, n, w, mx, psnr, snr, cos, pct))
print()
if worst is None:
    print("VERDICT: nothing to compare"); sys.exit(1)
# An fp16 GPU path against an fp32 CPU oracle does not reproduce it bit for bit; what matters is
# whether the gap is rounding or a different computation. These thresholds separate those two, they
# are not a quality bar: a kernel bug, a wrong layout or a dropped attribute lands far below them.
if worst == float("inf"):
    print("VERDICT: identical — the GPU reproduced the CPU oracle exactly")
elif worst >= 40:
    print("VERDICT: agree (worst PSNR %.2f dB) — consistent with fp16 rounding" % worst)
elif worst >= 25:
    print("VERDICT: marginal (worst PSNR %.2f dB) — larger than rounding; inspect the ops involved" % worst)
else:
    print("VERDICT: DISAGREE (worst PSNR %.2f dB) — the GPU is computing something different" % worst)
sys.exit(0 if worst >= 40 else 5)
PY
python3 "$WORK/compare.py" "$WORK"
gateVerdict=$?

# --- is it precision, or is it the kernel? ---------------------------------------------------------
# A gap this size has two very different causes and one cheap way to tell them apart: run the same
# GPU path with fp32 activations. If the gap closes, nothing computes the wrong thing -- some value
# simply does not survive fp16 storage (a sampling coordinate at a two-thousand-pixel width rounds by
# up to half a pixel, which moves an image without ever looking like an arithmetic error). If the gap
# stays, precision is not the story and a kernel is.
if [[ "$gateVerdict" != "0" && "$PRECISION" != "high" ]]; then
  echo
  echo "re-running the GPU arm with fp32 activations (--precision high) against the same oracle ..."
  $ADB shell "cd $WORKDIR && rm -rf out_hi && ./run_cg model.vxm out_hi --backend vulkan --precision high --no-cache$FOLD_FLAG$inNames" \
    </dev/null >"$WORK/hi.log" 2>&1
  rm -rf "$WORK/gpu"
  $ADB pull "$WORKDIR/out_hi" "$WORK/gpu" >/dev/null 2>&1
  sed 's/\x1b\[[0-9;]*m//g' "$WORK/hi.log" | grep -oE "output +'[^']+' +\[[0-9,]+\]" > "$WORK/shapes.txt" || true
  python3 "$WORK/compare.py" "$WORK"
  hiVerdict=$?
  echo
  if [[ "$hiVerdict" == "0" ]]; then
    echo "  => at fp32 the GPU agrees with the oracle. The difference is PRECISION, not a kernel:"
    echo "     some value in this graph does not survive fp16 storage."
  else
    echo "  => the difference is there at fp32 too, so it is not fp16 storage. A kernel computes"
    echo "     something different; the localize pass below names which."
  fi
  rm -rf "$WORK/gpu"
  $ADB pull "$WORKDIR/out_gpu" "$WORK/gpu" >/dev/null 2>&1
fi

# --- localize -------------------------------------------------------------------------------------
# Which tensor diverged FIRST. A disagreement at the output says only that something upstream is
# wrong; every tensor after the first bad one differs because its input did. Dumping both backends
# and walking the segment's own node order turns "the GPU is different" into one op's name.
walk_dumps() { # vxmOnDevice tag -- dump both arms of one .vxm and name where they part
  for arm in gpu cpu; do
    be=$([[ "$arm" == "gpu" ]] && echo vulkan || echo cpu)
    $ADB shell "cd $WORKDIR && rm -rf ld_$2_$arm && mkdir -p ld_$2_$arm && ./run_cg $1 ld_out_$2_$arm --backend $be --precision $PRECISION --no-cache$FOLD_FLAG --layer-dump --layer-dump-dir ld_$2_$arm$inNames" \
      </dev/null >"$WORK/ld_$2_$arm.log" 2>&1
  done
  rm -rf "$WORK/ldg" "$WORK/ldc"
  $ADB pull "$WORKDIR/ld_$2_gpu" "$WORK/ldg" >/dev/null 2>&1
  $ADB pull "$WORKDIR/ld_$2_cpu" "$WORK/ldc" >/dev/null 2>&1
  python3 "$WORK/walk.py" "$WORK" "$REL_PCT"
}

if [[ "$LOCALIZE" == "1" ]]; then
  cat >"$WORK/walk.py" <<'PYL'
import sys, os, re, glob
import numpy as np
work = sys.argv[1]
relBar = float(sys.argv[2]) / 100.0 if len(sys.argv) > 2 else 0.01
def side(tag):
    for d in (os.path.join(work, tag), ):
        for root, _, files in os.walk(d):
            if any(f.endswith(".bin") for f in files):
                return root
    return None
g, c = side("ldg"), side("ldc")
if not g or not c:
    print("  no dumps came back"); sys.exit(0)
# The index rows are "tensor <TAB> op <TAB> operand,operand,...", written in the order the segment
# recorded. Older builds wrote the bare tensor name, so a missing column just means no op attribution.
order, producer, operands = [], {}, {}
for cand in (os.path.join(g, "_order.txt"), os.path.join(c, "_order.txt")):
    if os.path.exists(cand):
        for line in open(cand):
            f = line.rstrip("\n").split("\t")
            if not f or not f[0].strip():
                continue
            # An index written by an older runner spells raw tensor names; the dumps flatten path
            # characters, so flatten here too and the two agree either way.
            def dumpName(x):
                return x.strip().replace("/", "_").replace(":", "_")
            nm = dumpName(f[0])
            order.append(nm)
            producer[nm] = f[1].strip() if len(f) > 1 else ""
            operands[nm] = [dumpName(o) for o in (f[2].split(",") if len(f) > 2 else []) if o.strip()]
        break
gpuHave = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(g, "*.bin"))}
cpuHave = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(c, "*.bin"))}
# The GPU keeps a tensor in whichever layout and precision its kernel wants, and names each variant
# by suffixing the original: a layout bridge writes "X#nc4<n>" / "X#flat<n>", a precision bridge
# writes "X#f32<n>" / "X#f16<n>". The CPU has neither kind of bridge and knows only "X". Matching
# names literally therefore drops every GPU-internal tensor -- including the twin that feeds the op
# a divergence is blamed on, which is the one that says whether its input was already wrong. Both
# dumps are canonical NCHW fp32, so a variant is comparable against the name it derives from.
INTERNAL_VARIANT = re.compile(r"#(?:nc4|flat|f32|f16)\d*$")
def counterpart(n):
    if n in cpuHave:
        return n
    base = INTERNAL_VARIANT.sub("", n)
    return base if base in cpuHave else None
inOrder = [n for n in order if n in gpuHave and counterpart(n)]
names = inOrder + sorted(x for x in gpuHave if x not in set(order) and counterpart(x))
aliased = sum(1 for n in names if counterpart(n) != n)
print("  %d tensor(s) comparable, %d of them against their pre-convert twin, bar %.2f%% of range%s"
      % (len(names), aliased, relBar * 100, "" if order else "  (no node-order index; name order)"))
first, rows, verdict, skipped, empty = None, [], {}, [], []
for n in names:
    pg, pc = os.path.join(g, n + ".bin"), os.path.join(c, counterpart(n) + ".bin")
    # Equal byte counts or the two sides hold different element types, and reading both as fp32
    # would compare one side's values against the other's bit patterns -- a fabricated divergence
    # that looks exactly like a real one. Report the mismatch instead of ranking noise.
    if os.path.getsize(pg) != os.path.getsize(pc):
        skipped.append((n, os.path.getsize(pg), os.path.getsize(pc)))
        continue
    a = np.fromfile(pg, dtype=np.float32).astype(np.float64)
    b = np.fromfile(pc, dtype=np.float32).astype(np.float64)
    k = a.size
    if k == 0:
        continue
    d = np.abs(a - b)
    scale = max(float(np.abs(b).max()), 1e-12)
    rel = float(d.max()) / scale
    rows.append((n, k, float(d.max()), rel))
    # Two thresholds, because either alone lies. Relative alone calls a 1e-9 difference on a tensor
    # whose values are 1e-9 a "100% divergence" -- true and useless, since both sides are zero to
    # every consumer. Absolute alone would miss a real disagreement in a small-valued tensor. A
    # divergence has to be big against the tensor's own range AND big enough to survive fp16.
    # A tensor whose largest difference equals its own largest value is not "drifted", it is
    # missing: one side computed essentially nothing. Worth calling out, because the causes are
    # disjoint -- an unwritten buffer or a skipped dispatch, never accumulated rounding.
    gmax, cmax = float(np.abs(a).max()), float(np.abs(b).max())
    if rel > 0.99 and min(gmax, cmax) <= 0.01 * max(gmax, cmax):
        empty.append((n, "gpu" if gmax < cmax else "cpu"))
    verdict[n] = rel > relBar and float(d.max()) > 1e-5
    if first is None and verdict[n]:
        first = (n, k, float(d.max()), rel)
if empty:
    print("  %d tensor(s) where ONE SIDE IS ESSENTIALLY ZERO and the other is not:" % len(empty))
    for nm, side in empty[:8]:
        print("    %-40.40s %s side is ~zero" % (nm, side))
if skipped:
    print("  %d tensor(s) not comparable (the two backends stored different element widths):" % len(skipped))
    for nm, bg, bc in skipped[:8]:
        print("    %-34.34s gpu %d B, cpu %d B" % (nm, bg, bc))
def operandLine(nm):
    ins = operands.get(nm, [])
    if not ins:
        return "    operands: none recorded"
    parts = []
    for o in ins:
        parts.append("%s [%s]" % (o, "DIVERGES" if verdict.get(o) else ("agrees" if o in verdict else "not dumped")))
    return "    operands: " + ", ".join(parts)
if first is None:
    print("  every dumped tensor agrees within the bar")
    sys.exit(0)
else:
    n, k, mx, rel = first
    print()
    print("  FIRST DIVERGENCE: '%s'  (%d elements, max|diff| %.6g, %.2f%% of the tensor's range)" % (n, k, mx, rel * 100))
    if producer.get(n):
        print("  produced by %s" % producer[n])
    print(operandLine(n))
    print("  everything after this differs because its input did; this is the op to look at.")
    print()
    # A window ending at the divergence, not the head of the list: the rows that matter are the last
    # few that still agreed, since they are what the failing op read.
    kAgreeingRowsShown = 12
    cut = next(i for i, r in enumerate(rows) if r[0] == n)
    lo = max(0, cut - kAgreeingRowsShown)
    print("  %-34s %10s %12s %9s" % ("tensor (node order)", "elements", "max|diff|", "rel%"))
    if lo > 0:
        print("  ... %d earlier tensor(s) agree" % lo)
    for nm, kk, mm, rr in rows[lo: cut + 1]:
        mark = "  <== first" if nm == n else ""
        print("  %-34.34s %10d %12.6g %8.3f%%%s" % (nm, kk, mm, rr * 100, mark))
    # Node order is one valid topological order among many, so "first" is a good lead but not a
    # proof. An op whose output diverges while every operand it read AGREES is the proof: nothing
    # upstream handed it a bad value, so the difference was made there. List them all.
    culprits = [nm for nm, _, _, _ in rows
                if verdict.get(nm)
                and any(o in verdict for o in operands.get(nm, []))
                and all(not verdict.get(o, False) for o in operands[nm])]
    if culprits:
        print()
        print("  ops that diverge on operands that AGREE (the difference is made here):")
        for nm in culprits[:12]:
            print("    %-34.34s %s" % (nm, producer.get(nm, "?")))
            print("  " + operandLine(nm))
    sys.exit(4)
PYL
  echo
  echo "localizing: dumping every activation on both backends ..."
  walk_dumps model.vxm fused
  fusedWalk=$?

  # Pointwise fusion is the one transform that changes WHICH kernel computes a tensor without
  # changing the graph's meaning, and it also swallows the intermediates a walk needs to see -- a
  # fused unit exports one tensor where the source had six. Recompiling without it answers both at
  # once: if the disagreement vanishes, fusion introduced it; if it survives, the finer dump names a
  # smaller op. Only possible when the model was given as ONNX, since a prebuilt .vxm is already fused.
  if [[ -n "$ONNX" && "$fusedWalk" != "0" ]]; then
    echo
    echo "re-checking without pointwise fusion (the same graph, one kernel per op) ..."
    NOFUSE="$WORK/nofuse.vxm"
    if "$COMPILER" "$ONNX" "$NOFUSE" --no-fuse-pointwise >"$WORK/nofuse_compile.log" 2>&1; then
      $ADB push "$NOFUSE" "$WORKDIR/nofuse.vxm" >/dev/null
      walk_dumps nofuse.vxm nofuse
      if [[ $? == 0 ]]; then
        echo
        echo "  => unfused, the GPU matches the oracle everywhere. POINTWISE FUSION introduces the"
        echo "     difference: the fused unit computes something its unfused equivalent does not."
      else
        echo
        echo "  => the difference survives without fusion, and the unfused dump above names a"
        echo "     smaller op than the fused one could."
      fi
    else
      echo "  unfused compilation failed:"; tail -5 "$WORK/nofuse_compile.log" | sed 's/^/    /'
    fi
  fi
fi
