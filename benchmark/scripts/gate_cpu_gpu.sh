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
    -h|--help)   sed -n '2,32p' "$0"; exit 0 ;;
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
  $ADB shell "cd $WORKDIR && rm -rf $2 && ./run_cg model.vxm $2 --backend $1 --precision $PRECISION --no-cache$inNames" \
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
# A CPU fallback on the "GPU" arm means the comparison is GPU-vs-CPU for only part of the graph.
falls=$(sed 's/\x1b\[[0-9;]*m//g' "$WORK/gpu.log" | grep -c "falling back")
[[ "$falls" == "0" ]] || echo "note: the GPU arm fell back to another backend for $falls op(s); those parts are compared against themselves"

$ADB pull "$WORKDIR/out_gpu" "$WORK/gpu" >/dev/null 2>&1
$ADB pull "$WORKDIR/out_cpu" "$WORK/cpu" >/dev/null 2>&1

# --- compare ------------------------------------------------------------------------------------
# The element width comes from the run log's declared shape and the file size, so a uint8 image, an
# fp16 tensor and an fp32 tensor are each read as themselves without a dtype flag.
sed 's/\x1b\[[0-9;]*m//g' "$WORK/gpu.log" | grep -oE "output +'[^']+' +\[[0-9,]+\]" > "$WORK/shapes.txt" || true
python3 - "$WORK" <<'PY'
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
PY
