#!/usr/bin/env python3
"""Unified VKNN benchmark driver.

One JSON config describes one or more *stages*; each stage runs independently on the device:
provide either an ONNX model (converted to .vxm with the given optimization options) or a ready
.vxm (runs as-is), feed .npy or raw .bin inputs (or none, for a runtime-only measurement), optionally
save outputs as .npy / .png, compare each output against a golden (cosine / PSNR / SNR / relL2 / max),
and collect a per-stage result.json with timing and (optional) per-operator profiling.

  run.py run    CONFIG.json [-v] [--no-build]  # build (./build.sh --android) + run every stage on device
  run.py convert ONNX OUT.vxm [opts]           # standalone convert (host or device)

`run` first runs ./build.sh --android (incremental; --no-build to skip), then for each stage logs the
files it pushes, the inputs it uses, the run timing, the pulled result.json, and saves the device logcat
to results/<stage>.logcat.txt.

Config (sectioned; see benchmark/configs/example.json and USAGE.md):
  { "defaults": { ...shared sections merged into every stage... },
    "stages": [
      { "name": "encoder8",
        "model":   { "onnx": "encoder.onnx" },          # or { "vxm": "encoder.vxm" }
        "convert": { "fp16": true, "fuse_se": false, "fuse_dwpw": false, "no_fuse_swish": false },
        "device":  { "backend": "vulkan", "serial": "", "precision": "fp16", "dir": "/data/local/tmp/vxrt/bench",
                     "no_weight_cache": true, "max_submit_nodes": 500, "cooldown": 22 },  # serial: adb id (multi-device)
        "inputs":  { "image": "image8.npy", "intrinsics": "intr8.bin" },   # or [...]; omit -> runtime only
        "outputs": { "save": ["npy","png"], "golden": { "means": "means_gold.npy" },
                     "metrics": ["cosine","psnr","snr"] },
        "profile": true, "bench": 3, "tolerance": 0.999, "result": "encoder8.result.json" } ] }
A single-stage config may omit "stages" and put the stage fields at the top level.
"""
import argparse, json, os, re, statistics, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd):
    # errors="replace": adb logcat (and other device output) can contain non-UTF-8 bytes that would
    # otherwise crash the text decoder.
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace")


# Logging. log() is always printed (flushed live, so multi-minute pushes/runs show progress);
# vlog() only with --verbose (full device stdout/stderr + the generated config).
VERBOSE = False


def log(msg=""):
    print(msg, flush=True)


def vlog(msg):
    if VERBOSE and msg:
        print(msg, flush=True)


def _indent(text, prefix):
    text = (text or "").rstrip()
    return prefix + text.replace("\n", "\n" + prefix) if text else ""


def human(n):
    s = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if s < 1024 or u == "TB":
            return f"{s:.0f} {u}" if u == "B" else f"{s:.1f} {u}"
        s /= 1024


# Target device (adb serial / id). Set per-stage from device.serial so a host with several phones
# attached is unambiguous; None = whatever single device adb finds.
_SERIAL = None


def set_serial(s):
    global _SERIAL
    _SERIAL = s or None


def adb(args):
    return sh(["adb"] + (["-s", _SERIAL] if _SERIAL else []) + args)


def push(src, dst):
    """adb-push one host file to the device, logging the transfer (name, size, destination).
    A missing local file is logged and skipped — the device may already hold a copy from an
    earlier run. Returns True only on a successful push."""
    if not os.path.exists(src):
        log(f"  [push] MISSING {src}  (skipped; device may already have it)")
        return False
    log(f"  [push] {os.path.basename(src)}  {human(os.path.getsize(src))}  -> {dst}")
    r = adb(["push", src, dst])
    if r.returncode != 0:
        log(f"  [push] FAILED {os.path.basename(src)}: {(r.stderr or r.stdout).strip()}")
        return False
    return True


def dev_exists(path):
    return adb(["shell", f"[ -f {path} ] && echo Y || echo N"]).stdout.strip().endswith("Y")


def need_device():
    devs = [l.split()[0] for l in sh(["adb", "devices"]).stdout.splitlines()[1:] if "\tdevice" in l]
    if not devs:
        sys.exit("no adb device (check `adb devices`; the phone may be asleep)")
    if _SERIAL and _SERIAL not in devs:
        sys.exit(f"device serial '{_SERIAL}' not attached. connected: {', '.join(devs) or '(none)'}")
    if not _SERIAL and len(devs) > 1:
        sys.exit(f"multiple devices attached ({', '.join(devs)}); set device.serial in the config")


def host_bin(name):
    p = os.path.join(ROOT, "build-host", name)
    return p if os.path.exists(p) else None


# Auto-build the Android binaries so `run` never needs a manual `./build.sh --android` first. The
# build is incremental (Ninja) — a near-no-op when nothing changed, a rebuild when sources changed.
# --no-build skips it (e.g. offline from the NDK, reusing existing binaries).
BUILD = True


def build_android():
    log("  [build] ./build.sh --android")
    r = subprocess.run(["bash", os.path.join(ROOT, "build.sh"), "--android"], cwd=ROOT)
    if r.returncode != 0:
        sys.exit("android build failed (see output above)")


def android_bin(name):
    p = os.path.join(ROOT, "build-android", name)
    if not os.path.exists(p):
        if not BUILD:
            sys.exit(f"missing {p} — run ./build.sh --android (or drop --no-build)")
        build_android()
        if not os.path.exists(p):
            sys.exit(f"{name} not produced by ./build.sh --android")
    return p


def merge(defaults, stage):
    out = json.loads(json.dumps(defaults)) if defaults else {}
    for k, v in stage.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------- convert
def convert_flags(conv):
    f = ["--fp16"] if conv.get("fp16", True) else []
    if conv.get("no_fuse_swish"):
        f.append("--no-fuse-swish")
    if conv.get("fuse_se"):
        f.append("--fuse-se")
    if conv.get("fuse_dwpw"):
        f.append("--fuse-dwpw")
    return f


def convert(onnx, out_vxm, conv, where="host"):
    flags = convert_flags(conv)
    if where == "host" and host_bin("vknn_compile"):
        log(f"[convert] host: {os.path.basename(onnx)} -> {os.path.basename(out_vxm)}  {' '.join(flags)}")
        r = sh([host_bin("vknn_compile"), onnx, out_vxm] + flags)
        if r.returncode != 0:
            sys.exit("convert failed:\n" + r.stdout + r.stderr)
        log(f"[convert] wrote {out_vxm}  ({human(os.path.getsize(out_vxm))})")
        return out_vxm
    need_device()
    ddir = "/data/local/tmp/vxrt/bench"
    adb(["shell", "mkdir", "-p", ddir])
    log(f"[convert] device: {os.path.basename(onnx)} -> {os.path.basename(out_vxm)}  {' '.join(flags)}")
    push(onnx, f"{ddir}/_src.onnx")
    wb = os.path.join(os.path.dirname(onnx), "weights.bin")
    if os.path.exists(wb):
        push(wb, f"{ddir}/weights.bin")
    push(android_bin("vknn_compile"), f"{ddir}/vknn_compile")
    adb(["shell", "chmod", "+x", f"{ddir}/vknn_compile"])
    r = adb(["shell", f"cd {ddir} && ./vknn_compile _src.onnx {os.path.basename(out_vxm)} " + " ".join(flags)])
    if r.stdout.strip():
        vlog(r.stdout.strip())
    if r.stderr.strip():  # the engine logs (incl. errors) all go to stderr
        log("  [compile] " + r.stderr.strip().replace("\n", "\n  [compile] "))
    if r.returncode != 0:
        sys.exit("device convert failed")
    log(f"[convert] device wrote {os.path.basename(out_vxm)}")
    return None  # already on device under its basename


# ----------------------------------------------------------------- per stage
def parse(out):
    # `out` is the device run's stdout+stderr. Prefer benchmark.cpp's end-to-end "run X ms"
    # (whole-inference wall, what the headline numbers are measured against); fall back to the
    # engine's per-submit "submit+gpu=" only when no run line is present (it under-counts a model
    # that is chunked across several submits for the GPU watchdog).
    run = re.search(r"run ([0-9.]+) ms", out)
    sg = re.search(r"submit\+gpu=([0-9.]+)ms", out)
    rows = re.findall(r"^\s+(\S+)\s+(cos=[0-9.\-]+.*?)(PASS|FAIL|SIZE-MISMATCH|NOT-AN-OUTPUT)\s*$", out, re.M)
    ok = ("ALL OUTPUTS PASS" in out) or ("ALL OUTPUTS PASS" not in out and "FAIL" not in out and "golden" not in out)
    t = float(run.group(1)) if run else (float(sg.group(1)) if sg else None)
    return t, rows, ok


def run_stage(stage, base, idx, where_convert="host"):
    name = stage.get("name", f"stage{idx}")
    model = stage.get("model", {})
    dev = stage.get("device", {})
    ddir = dev.get("dir", "/data/local/tmp/vxrt/bench")
    set_serial(dev.get("serial") or dev.get("id") or dev.get("hash"))  # pick this stage's device
    need_device()
    adb(["shell", "mkdir", "-p", ddir])

    def host(p):
        return p if os.path.isabs(p) else os.path.join(base, p)

    log(f"\n==== stage: {name} ====")
    log(f"  [device] serial={_SERIAL or '(single attached)'}  dir={ddir}")
    # ---- model: convert onnx, or push a ready vxm ----
    if model.get("vxm"):
        vxm = host(model["vxm"]); model_name = os.path.basename(vxm)
        push(vxm, f"{ddir}/{model_name}")
        log(f"  [model] {model_name}  (ready vxm, no convert)")
    elif model.get("onnx"):
        onnx = host(model["onnx"]); conv = stage.get("convert", {})
        model_name = conv.get("out") or (os.path.splitext(os.path.basename(onnx))[0] + ".vxm")
        local = os.path.join(tempfile.gettempdir(), model_name)
        res = convert(onnx, local, conv, where_convert)
        if res:  # converted on host -> push
            push(local, f"{ddir}/{model_name}")
        log(f"  [model] {model_name}")
    else:
        sys.exit(f"stage {name}: model needs \"onnx\" or \"vxm\"")

    # ---- inputs (npy/raw) + goldens ----
    def push_io(m):
        if isinstance(m, list):
            for p in m:
                push(host(p), f"{ddir}/{os.path.basename(p)}")
            return [os.path.basename(p) for p in m]
        dev_m = {}
        for k, p in (m or {}).items():
            push(host(p), f"{ddir}/{os.path.basename(p)}")
            dev_m[k] = os.path.basename(p)
        return dev_m

    out_cfg = stage.get("outputs", {})
    dcfg = {"model": model_name, "backend": dev.get("backend", "vulkan"),
            "precision": dev.get("precision", "fp16"), "no_weight_cache": dev.get("no_weight_cache", True),
            "timing": True, "profile": stage.get("profile", False),
            "tolerance": stage.get("tolerance", 0.999), "result": "result.json", "save_dir": "."}
    if "max_submit_nodes" in dev:
        dcfg["max_submit_nodes"] = dev["max_submit_nodes"]
    if dev.get("cache"):  # unified per-model cache file (default on device: <model>.cache)
        dcfg["cache"] = os.path.basename(dev["cache"])
    if stage.get("generate_cache") or dev.get("generate_cache"):  # untimed warm-up load to populate it
        dcfg["generate_cache"] = True
    for k in ("winograd", "tuning", "winogradVariant", "winogradUnit", "directConv3x3"):  # conv kernel hints
        v = stage.get(k, dev.get(k))
        if v is not None:
            dcfg[k] = v
    if stage.get("inputs"):
        dcfg["inputs"] = push_io(stage["inputs"])
    if out_cfg.get("save"):
        dcfg["save"] = out_cfg["save"]
    if out_cfg.get("golden"):
        dcfg["golden"] = push_io(out_cfg["golden"])
    if out_cfg.get("metrics"):
        dcfg["metrics"] = out_cfg["metrics"]

    # ---- what the device run will actually use ----
    if dcfg.get("inputs"):
        items = dcfg["inputs"].items() if isinstance(dcfg["inputs"], dict) else enumerate(dcfg["inputs"])
        for k, v in items:
            log(f"  [input] {k} = {v}")
    else:
        log("  [input] (none -> zero-filled, runtime-only)")
    if dcfg.get("golden"):
        log(f"  [golden] {', '.join(f'{k}={v}' for k, v in dcfg['golden'].items())}")
    log(f"  [opts] precision={dcfg['precision']} no_weight_cache={dcfg['no_weight_cache']} "
        f"profile={dcfg['profile']} max_submit_nodes={dcfg.get('max_submit_nodes', '(default)')} "
        f"tolerance={dcfg['tolerance']}")

    local_cfg = os.path.join(tempfile.gettempdir(), f".cfg_{name}.json")
    json.dump(dcfg, open(local_cfg, "w"), indent=2)
    vlog("  [config.json] " + json.dumps(dcfg))
    push(local_cfg, f"{ddir}/config.json")
    push(android_bin("vknn_benchmark"), f"{ddir}/vknn_benchmark")
    adb(["shell", "chmod", "+x", f"{ddir}/vknn_benchmark"])
    # drop any stale result so its presence afterwards is a real success signal
    adb(["shell", f"rm -f {ddir}/result.json"])

    # ---- run (bench times) ----
    n = stage.get("bench", 1)
    cd = dev.get("cooldown", 22 if n > 1 else 0)
    cmd = f"cd {ddir} && ./vknn_benchmark config.json"
    log(f"  [run] {n}x  cooldown={cd}s  $ {cmd}")
    adb(["logcat", "-c"])  # start this run's logcat window clean
    times, ok_all, last = [], True, ""
    for i in range(n):
        if cd:
            adb(["shell", "sleep", str(cd)])
        r = adb(["shell", cmd])
        last = r.stdout + r.stderr  # the engine (timing + errors) writes to stderr
        vlog(_indent(r.stdout, "  [device] "))
        vlog(_indent(r.stderr, "  [device:err] "))
        t, _, ok = parse(last)
        ok_all = ok_all and ok
        if t:
            times.append(t)
        tag = "" if n == 1 else f"run {i+1}/{n}: "
        if t:
            log(f"  {tag}run={t:.1f} ms")
        else:
            log(f"  {tag}(no timing) -- vknn_benchmark printed no run/submit line; exit={r.returncode}")
            err = (r.stderr or r.stdout).strip()
            if err:  # surface the device failure (missing input, load failure, crash, ...)
                log(_indent(err, "    "))
    for ln in last.splitlines():
        if "cos=" in ln or "PASS" in ln or "FAIL" in ln or "no inputs" in ln:
            log("   " + ln.strip())
    if len(times) > 1:
        log(f"  -> min={min(times):.1f} median={statistics.median(times):.1f} max={max(times):.1f} ms")

    # ---- pull result.json + report saved outputs ----
    results_dir = os.path.join(base, "results")
    os.makedirs(results_dir, exist_ok=True)
    rj = stage.get("result", f"{name}.result.json")
    rj_local = os.path.join(results_dir, os.path.basename(rj))
    if dev_exists(f"{ddir}/result.json"):
        pr = adb(["pull", f"{ddir}/result.json", rj_local])
        if pr.returncode == 0 and os.path.exists(rj_local):
            log(f"  [result] results/{os.path.basename(rj)}  ({human(os.path.getsize(rj_local))})")
        else:
            log(f"  [result] pull FAILED: {(pr.stderr or pr.stdout).strip()}")
            ok_all = False
    else:
        log("  [result] device wrote NO result.json -- the run failed before writing it "
            "(see the device error above); nothing pulled")
        ok_all = False
    if out_cfg.get("save"):
        for fmt in out_cfg["save"]:
            ls = adb(["shell", f"ls {ddir}/*.{fmt} 2>/dev/null"]).stdout.split()
            if ls:  # left on device — pull them yourself
                log(f"  [saved:{fmt}] " + ", ".join(os.path.basename(x) for x in ls) + f"  (on device in {ddir})")

    # ---- save this run's logcat (the GPU driver / OOM-killer / watchdog reset logs land here, not in
    #      the executor's stderr) ----
    lc_path = os.path.join(results_dir, f"{name}.logcat.txt")
    lc = adb(["logcat", "-d"])
    if lc.returncode == 0:
        with open(lc_path, "w") as f:
            f.write(lc.stdout)
        log(f"  [logcat] results/{name}.logcat.txt  ({human(os.path.getsize(lc_path))})")
    else:
        log(f"  [logcat] capture failed: {(lc.stderr or lc.stdout).strip()}")
    return ok_all


def main():
    global VERBOSE, BUILD
    parser = argparse.ArgumentParser(description="VKNN benchmark: convert + run + validate + profile on device")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("config")
    run_parser.add_argument("--convert-on", choices=["host", "device"], default="host")
    run_parser.add_argument("-v", "--verbose", action="store_true", help="print device stdout/stderr and the generated config")
    run_parser.add_argument("--no-build", action="store_true", help="skip the automatic ./build.sh --android")

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("onnx")
    convert_parser.add_argument("out")
    convert_parser.add_argument("--fp16", action="store_true", default=True)
    convert_parser.add_argument("--fp32", dest="fp16", action="store_false")
    convert_parser.add_argument("--fuse-se", action="store_true")
    convert_parser.add_argument("--fuse-dwpw", action="store_true")
    convert_parser.add_argument("--no-fuse-swish", action="store_true")
    convert_parser.add_argument("--on", choices=["host", "device"], default="host")
    convert_parser.add_argument("--serial", default=None, help="adb device serial (for --on device with multiple devices)")
    convert_parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    VERBOSE = getattr(args, "verbose", False)
    BUILD = not getattr(args, "no_build", False)

    if args.cmd == "convert":
        set_serial(args.serial)
        convert(args.onnx, args.out, {"fp16": args.fp16, "fuse_se": args.fuse_se, "fuse_dwpw": args.fuse_dwpw,
                                      "no_fuse_swish": args.no_fuse_swish}, args.on)
        log("done.")
        return

    cfg = json.load(open(args.config))
    # Model/input/golden paths in a config (and the results/ dir) resolve against the benchmark/ root —
    # where run.py, models/, and results/ live — so a config under benchmark/configs/ still finds
    # "models/...". Absolute paths in a config are used as-is.
    base = os.path.dirname(os.path.abspath(__file__))
    defaults = cfg.get("defaults", {})
    stages = cfg.get("stages") or [cfg]
    log(f"config: {args.config}  ({len(stages)} stage{'s' if len(stages) != 1 else ''})  base={base}")
    if BUILD:
        build_android()
    ok = True
    for i, st in enumerate(stages):
        ok = run_stage(merge(defaults, st), base, i, args.convert_on) and ok
    log("\n=== ALL STAGES PASS ===" if ok else "\n=== SOME STAGES FAILED ===")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
