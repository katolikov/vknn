#!/usr/bin/env bash
# gate_lib.sh — shared helpers for the VKNN device byte gates (gate_op.sh, gate_pw_probes.sh).
#
# Sourced, never executed. Provides the pinned deterministic run config and the fused-vs-unfused
# byte-identity + zero-fallback comparison so both gates agree on what "byte gate" means:
#
#   * pinned run config: --backend vulkan --no-cache --no-fold-islands --tuning none (deterministic;
#     the caller supplies --precision and --winograd). "none" tuning removes per-shape autotune noise.
#   * fused == unfused byte-identity: every fused output file `cmp`-identical to its --no-fuse-pointwise
#     twin. This is a BRANCH-INTERNAL invariant (fusion changed no bytes), NOT an absolute pass count.
#   * zero CPU fallback: the fused run log has no "falling back" line (Session prints one per op with
#     no GPU kernel — see src/core/session.cpp). No device-side --support-report flag exists yet;
#     the run-log grep is the fallback detector until it lands.
#
# The absolute number of probes that "pass" is meaningless across probe-generator revisions (the
# generator drifts from a gate's expectations); a meaningful gate is THIS branch vs a fresh-built
# main. gate_gpu_byte_ref() runs a second, ref binary and cross-compares so the verdict is
# branch-vs-ref, not a score.

# gate_pinned_run <adb> <workdir> <binary> <vxm> <outdir> <precflag> <winograd> <logfile> <in-names...>
#   cd to <workdir> on device, then run one model with the pinned deterministic config, capturing
#   stdout+stderr to <logfile>. <binary>, <vxm>, <outdir>, <logfile> and the input names are all
#   interpreted relative to <workdir> (or absolute). --cache . keeps the cache in <workdir>.
gate_pinned_run() {
  local adb="$1" wd="$2" bin="$3" vxm="$4" od="$5" pf="$6" wino="$7" log="$8"; shift 8
  local ins="$*"
  # shellcheck disable=SC2086
  $adb shell "cd $wd && rm -rf $od && $bin $vxm $od --backend vulkan --precision $pf \
    --no-cache --no-fold-islands --winograd $wino --tuning none --cache . $ins > $log 2>&1; echo done" >/dev/null
}

# gate_cmp_dirs <adb> <workdir> <dir-a> <dir-b>  -> echoes 1 if every file in dir-a is byte-identical
#   in dir-b (and dir-a is non-empty), else 0. dir-a/dir-b are relative to <workdir> (or absolute).
gate_cmp_dirs() {
  local adb="$1" wd="$2" a="$3" b="$4"
  $adb shell "cd $wd && n=\$(ls $a 2>/dev/null | wc -l); ok=1; [ \$n -lt 1 ] && ok=0; \
    for f in \$(ls $a 2>/dev/null); do cmp $a/\$f $b/\$f >/dev/null 2>&1 || ok=0; done; echo \$ok" \
    | tr -d '[:space:]'
}

# gate_fallback_count <adb> <workdir> <logfile>  -> count of "falling back" lines in the run log
#   (relative to <workdir> or absolute).
gate_fallback_count() {
  local adb="$1" wd="$2" log="$3"
  $adb shell "cd $wd && grep -ci 'falling back' $log 2>/dev/null" | tr -d '[:space:]'
}

# gate_require_device <adb-cmd> <serial>  -> exits non-zero (message on stderr) unless adb sees it.
gate_require_device() {
  local adb="$1" serial="$2"
  if ! command -v adb >/dev/null 2>&1; then
    echo "gate: adb not found on PATH — a connected device is required" >&2; return 1
  fi
  if ! $adb get-state >/dev/null 2>&1; then
    echo "gate: device '$serial' not reachable (adb -s $serial get-state failed)." >&2
    echo "      Pass a serial as the DEVICE arg, or attach the device." >&2
    return 1
  fi
  return 0
}
