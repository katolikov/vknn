#!/usr/bin/env bash
# Full on-device VKNN profile for one model on one device.
#
# Captures everything needed to explain a device's runtime behaviour in a single pass:
#   - GPU identity + capabilities (deviceName, driver, Vulkan version, fp16/subgroup flags)
#   - the per-op GPU profile table and GPU total, for every op in the graph
#   - cold (no cache) vs warm (cache hit) session and steady-state timing
#   - fp16 (precision low) vs fp32 (precision high) side by side
#   - queue priority high, to separate scheduling from clocks
#   - per-op CPU-fallback isolation (which op, if forced to CPU, dominates the runtime)
#   - a scan for any op that fell back to the CPU on this driver
#   - memory in use: peak host (CPU) RSS across a full load+run, and the engine's own device (GPU)
#     live/peak buffer accounting, so a change that trades memory for speed is visible here
#
# The device serial and the model are arguments, so nothing device- or model-specific is baked in.
# A model with no supplied inputs runs zero-filled (op timing is shape-driven, not value-driven,
# so the profile is still valid); pass real input .bin files for an exact run.
#
#   scripts/profile_on_device.sh <adb-serial> <model.vxm|model.onnx> [in0.bin in1.bin ...] [-- extra run_io flags]
#
# Examples:
#   scripts/profile_on_device.sh ABC123XYZ model.vxm
#   scripts/profile_on_device.sh ABC123XYZ model.vxm a.bin b.bin -- --fp32-tensors 'grid_,stats_'
#
# Env:
#   OUTDIR=<dir>   host directory for the captured logs (default: vknn_profile_out)
#   REPEAT=<n>     timed iterations per run (default: 25)
#   BUILD_DIR=<d>  android build dir holding vknn_run_io (default: build-android)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log()  { printf '\033[36m>> %s\033[0m\n' "$*"; }
die()  { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }
strip() { sed 's/\x1b\[[0-9;]*m//g'; }   # drop ANSI colour so captured logs stay greppable

[[ $# -ge 2 ]] || die "usage: scripts/profile_on_device.sh <adb-serial> <model.vxm|onnx> [inputs...] [-- extra run_io flags]"
SERIAL="$1"; MODEL="$2"; shift 2

# Split the rest into positional input files (before '--') and pass-through run_io flags (after '--').
INPUTS=(); EXTRA=()
seen_dashdash=0
for a in "$@"; do
    if [[ $seen_dashdash -eq 1 ]]; then EXTRA+=("$a")
    elif [[ "$a" == "--" ]]; then seen_dashdash=1
    else INPUTS+=("$a"); fi
done

BUILD_DIR="${BUILD_DIR:-build-android}"
RUNNER="$BUILD_DIR/vknn_run_io"
COMPILER="build-host/vknn_compile"
REPEAT="${REPEAT:-25}"
OUTDIR="${OUTDIR:-vknn_profile_out}"
DEV_DIR="/data/local/tmp/vknn/profile"

command -v adb >/dev/null || die "adb not found on PATH"
adb -s "$SERIAL" get-state >/dev/null 2>&1 || die "device '$SERIAL' not connected (check: adb devices)"
[[ -f "$MODEL" ]]  || die "model '$MODEL' not found"
[[ -x "$RUNNER" ]] || die "$RUNNER missing -- build it first: ./scripts/build_android.sh"

# A .onnx is compiled to a temporary .vxm on the host; a .vxm is pushed as-is.
mkdir -p "$OUTDIR"
if [[ "$MODEL" == *.onnx ]]; then
    [[ -x "$COMPILER" ]] || die "$COMPILER missing (needed to compile the .onnx) -- build the host tools first"
    VXM="$OUTDIR/$(basename "${MODEL%.onnx}").vxm"
    log "compiling $MODEL -> $VXM (host)"
    "$COMPILER" "$MODEL" "$VXM" >/dev/null
else
    VXM="$MODEL"
fi
MODEL_ON_DEV="$(basename "$VXM")"

log "push -> $SERIAL:$DEV_DIR"
adb -s "$SERIAL" shell "mkdir -p $DEV_DIR"
adb -s "$SERIAL" push "$RUNNER" "$DEV_DIR/" >/dev/null
adb -s "$SERIAL" push "$VXM"    "$DEV_DIR/" >/dev/null
adb -s "$SERIAL" shell "chmod +x $DEV_DIR/vknn_run_io"
INPUT_ARGS=()
for f in "${INPUTS[@]:-}"; do
    [[ -n "$f" ]] || continue
    [[ -f "$f" ]] || die "input '$f' not found"
    adb -s "$SERIAL" push "$f" "$DEV_DIR/" >/dev/null
    INPUT_ARGS+=("$(basename "$f")")
done
[[ ${#INPUT_ARGS[@]} -gt 0 ]] || log "no inputs supplied -> running zero-filled (timing is shape-driven, still valid)"

# run <label> <flags...> : execute run_io on the device, tee the stripped output to a per-run log.
run() {
    local label="$1"; shift
    local logf="$OUTDIR/${label}.log"
    adb -s "$SERIAL" shell "cd $DEV_DIR && LD_LIBRARY_PATH=. ./vknn_run_io $MODEL_ON_DEV out \
        --backend vulkan $* ${EXTRA[*]:-} ${INPUT_ARGS[*]:-}" 2>&1 | strip > "$logf"
    echo "$logf"
}

# Cold = no cache on disk: the run compiles the pipelines fresh AND writes the cache, so the next
# run is a genuine warm (cache-hit) comparison. (--no-cache is avoided here precisely because it
# would suppress that write and leave RUN 2 cold as well.)
log "RUN 1/7  cold (fresh compile, writes cache), fp16, full profile"
adb -s "$SERIAL" shell "cd $DEV_DIR && rm -f ${MODEL_ON_DEV%.vxm}.cache" >/dev/null 2>&1 || true
L_COLD=$(run 01_cold_fp16 --precision low  --profile --timing --repeat "$REPEAT")

log "RUN 2/7  warm, fp16, full profile"
L_WARM=$(run 02_warm_fp16 --precision low  --profile --timing --repeat "$REPEAT")

log "RUN 3/7  warm, fp32 (precision high)"
L_FP32=$(run 03_warm_fp32 --precision high --profile --timing --repeat "$REPEAT")

log "RUN 4/7  fp16, priority high"
L_PRIO=$(run 04_priority_high --precision low --profile --timing --repeat "$REPEAT" --priority high)

log "RUN 5/7  cpu backend (oracle per-op timing)"
adb -s "$SERIAL" shell "cd $DEV_DIR && LD_LIBRARY_PATH=. ./vknn_run_io $MODEL_ON_DEV out \
    --backend cpu --profile --repeat 1 ${EXTRA[*]:-} ${INPUT_ARGS[*]:-}" 2>&1 | strip > "$OUTDIR/05_cpu.log"

log "RUN 6/7  per-op CPU-fallback isolation (top GPU op-types)"
# Isolate each of the graph's op-types onto the CPU in turn; whichever removal collapses the GPU
# time names the op that dominates, and reveals what a driver-side fallback of it would cost.
OPTYPES=()   # portable read (avoids mapfile, absent from stock macOS bash 3.2)
while IFS= read -r op; do OPTYPES+=("$op"); done \
    < <(grep -A40 'Per op-type' "$L_WARM" | grep -E '^[[:space:]]+[A-Za-z]' | awk '{print $1}' | head -6)
for op in "${OPTYPES[@]:-}"; do
    [[ -n "$op" ]] || continue
    adb -s "$SERIAL" shell "cd $DEV_DIR && LD_LIBRARY_PATH=. ./vknn_run_io $MODEL_ON_DEV out \
        --backend vulkan --precision low --profile --repeat 5 --disable-vk-ops $op ${EXTRA[*]:-} ${INPUT_ARGS[*]:-}" \
        2>&1 | strip > "$OUTDIR/06_cpu_${op}.log"
done

log "RUN 7/7  memory: peak host RSS sampled across a full load+run"
# Host RSS: VmHWM is the kernel's own high-water mark for the process, so it survives the peak even
# though this samples periodically -- the sampling loop only has to observe the counter, not catch
# the instant. Device memory comes from the engine's own accounting line (live / peak bytes over
# the buffers it allocated), which is the number a driver-side query cannot attribute to us alone
# on a shared GPU.
# `cd DIR && cmd &` backgrounds the WHOLE list, so $! would be the subshell and the sampler would
# read the shell's own RSS. Separating with `;` backgrounds only run_io, and the cd still applies to
# the log path.
adb -s "$SERIAL" shell "cd $DEV_DIR; LD_LIBRARY_PATH=. ./vknn_run_io $MODEL_ON_DEV out \
    --backend vulkan --precision low --repeat 3 ${EXTRA[*]:-} ${INPUT_ARGS[*]:-} >mem_run.log 2>&1 & \
    pid=\$!; hwm=0; \
    while kill -0 \$pid 2>/dev/null; do \
        v=\$(cat /proc/\$pid/status 2>/dev/null | grep VmHWM | tr -dc '0-9'); \
        if [ -n \"\$v\" ] && [ \"\$v\" -gt \"\$hwm\" ]; then hwm=\$v; fi; \
    done; \
    wait \$pid; echo \"peak_host_rss_kb=\$hwm\"; cat mem_run.log" 2>&1 | strip > "$OUTDIR/07_memory.log"

# ---- summary --------------------------------------------------------------------------------
sep() { printf '%s\n' "--------------------------------------------------------------------------------"; }
{
    sep; echo "VKNN device profile summary"; sep
    echo "device serial : $SERIAL"
    echo "model         : $MODEL  (on device: $MODEL_ON_DEV)"
    echo "inputs        : ${INPUT_ARGS[*]:-<zero-filled>}"
    echo "extra flags   : ${EXTRA[*]:-<none>}"
    echo "logs          : $OUTDIR/"
    sep; echo "GPU identity + capabilities"; sep
    grep -E 'Vulkan ready|fp16=|Compute queue family|Active backends' "$L_WARM" || true
    sep; echo "backend assignment / CPU fallback"; sep
    if grep -qE 'fall back to CPU|\[FALLBACK\]' "$L_WARM"; then
        grep -E 'fall back to CPU|\[FALLBACK\]' "$L_WARM"
    else
        echo "no CPU fallback -- every op ran on Vulkan"
    fi
    sep; echo "per-op GPU profile (warm, fp16)   [op-type    cpu ms / gpu ms]"; sep
    grep -E '^[[:space:]]+[A-Za-z].*/[[:space:]]+[0-9]' "$L_WARM" || true
    grep -E 'GPU total' "$L_WARM" | tail -1 || true
    sep; echo "memory (CPU host + GPU device)"; sep
    # Every lookup below tolerates a miss: under `set -e` an empty grep would otherwise abort the
    # whole summary and discard the runs that did succeed.
    rss_kb=$(grep -oE 'peak_host_rss_kb=[0-9]+' "$OUTDIR/07_memory.log" 2>/dev/null | cut -d= -f2 | tail -1 || true)
    if [[ -n "${rss_kb:-}" && "$rss_kb" != 0 ]]; then
        awk -v kb="$rss_kb" 'BEGIN { printf "peak host RSS      : %.1f MB (VmHWM, whole process: weights + arena staging + runtime)\n", kb / 1024 }'
    else
        echo "peak host RSS      : <not sampled>"
    fi
    # The engine's own device accounting: live and peak bytes over the buffers IT allocated, plus
    # the host RSS it observed at the same moment. Reported per segment build.
    if grep -q 'vk memory' "$OUTDIR/07_memory.log" 2>/dev/null; then
        grep -E 'vk memory' "$OUTDIR/07_memory.log" | tail -3 | sed 's/^.*vk memory/GPU device memory  : /'
    else
        { grep -E 'vk memory' "$L_WARM" | tail -3 | sed 's/^.*vk memory/GPU device memory  : /'; } || echo "GPU device memory  : <not reported>"
    fi
    { grep -E 'freed .* of host weights' "$OUTDIR/07_memory.log" | tail -1 | sed 's/^.*freed/host weight reclaim: freed/' | grep . ; } \
        || echo "host weight reclaim: <none reported>"
    { grep -E 'mapped .* read-only' "$OUTDIR/07_memory.log" | tail -1 | sed 's/^.*\] /model bytes        : /'; } || true
    sep; echo "timing (GPU total / session)"; sep
    printf 'cold  : '; grep -E 'GPU total' "$L_COLD" | tail -1
    printf 'warm  : '; grep -E 'GPU total' "$L_WARM" | tail -1
    printf 'fp32  : '; grep -E 'GPU total' "$L_FP32" | tail -1
    printf 'prio+ : '; grep -E 'GPU total' "$L_PRIO" | tail -1
    grep -E 'Session created' "$L_COLD" | tail -1 | sed 's/^/cold session : /'
    grep -E 'Session created' "$L_WARM" | tail -1 | sed 's/^/warm session : /'
    grep -E 'gpu span|submit\+gpu' "$L_WARM" | tail -1 | sed 's/^/warm wall    : /'
    sep
} | tee "$OUTDIR/summary.txt"

log "done -- full logs in $OUTDIR/ (per-run *.log + summary.txt)"
