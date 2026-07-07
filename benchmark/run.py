#!/usr/bin/env python3
"""Unified VKNN benchmark driver.

One JSON config describes one or more *stages*; each stage runs independently on the device:
provide an ONNX model (converted to .vxm with the given optimization level) or a ready .vxm, feed
.npy / raw .bin inputs (a file map, a file list, or one-or-more input DIRECTORIES), optionally
compare each output against a golden (cosine / PSNR / SNR / relL2 / max), and collect everything
under result/<RUN>/<stage>/ — result.json, saved outputs, the device logcat, and (on request) the
compiled .vxm + its .cache.

  run.py run     CONFIG.json [--run NAME] [--clean] [-v] [--no-build] [--convert-on host|device]
  run.py convert ONNX OUT.vxm [-O N] [opts]      # standalone convert (host or device)

Every stage prints its timing on the host (load + run min/median/avg/max ms) and the ops that fell
back to the CPU (a release run shows 0). A missing model/input/golden file STOPS the stage loudly.

Config (see benchmark/configs/example.json and USAGE.md):
  { "defaults": { ...shared sections merged into every stage... },
    "stages": [
      { "name": "resnet50",
        "model": "models/resnet50.onnx",            # .onnx (converted) or .vxm (as-is)
        "convert": { "fp16": true, "opt": 1 },      # opt level -O0..-O3 + per-fusion overrides
        "device":  { "serial": "", "dir": "/data/local/tmp/vknn/bench", "clean": false,
                     "cooldown": 0 },               # cooldown: seconds to idle before the run
        "run":     { "backend": "vulkan", "precision": "low",
                     "iters": 10, "warmup": 2, "profile": false, "fold_islands": true,
                     "max_submit_nodes": 500, "winograd": "auto", "tuning": "fast",
                     "tolerance": 0.999 },
        "inputs":  { "image": "image.npy" },        # or [files...] (input order)
                                                    # or "dir" / ["dir1","dir2"]: each dir is one
                                                    #   input SET; files map to inputs by stem name,
                                                    #   <name>_gold.npy in the dir is that set's golden
        "golden":  { "means": "means_gold.npy" },   # for file inputs; dir inputs carry their own
        "metrics": ["cosine","psnr","snr"],
        "save":    ["npy"],                         # output formats, pulled to result/<RUN>/<stage>/outputs/
        "pull":    ["vxm","cache"] } ] }            # also pull the compiled model + cache
A single-stage config may omit "stages" and put the stage fields at the top level.
"""
import argparse, datetime, json, os, re, statistics, subprocess, sys, tempfile

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


def push(src, dst, what="file"):
    """adb-push one host file, or STOP the run: a missing/failed transfer would otherwise surface as
    a confusing device-side error (or, worse, a stale copy from an earlier run would be used)."""
    if not os.path.exists(src):
        sys.exit(f"MISSING {what}: {src}")
    log(f"  [push] {os.path.basename(src)}  {human(os.path.getsize(src))}  -> {dst}")
    r = adb(["push", src, dst])
    if r.returncode != 0:
        sys.exit(f"push FAILED {os.path.basename(src)}: {(r.stderr or r.stdout).strip()}")


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
    f.append(f"-O{int(conv.get('opt', 1))}")  # optimization level; per-fusion keys override below
    for key, flag_ in (("no_fuse_swish", "--no-fuse-swish"), ("fuse_se", "--fuse-se"),
                       ("fuse_dwpw", "--fuse-dwpw"), ("no_fuse_pointwise", "--no-fuse-pointwise")):
        if conv.get(key):
            f.append(flag_)
    return f


def onnx_external_data(onnx):
    """External-data file names an ONNX references (initializer external_data 'location' entries),
    relative to the onnx's directory. Empty list when the model is self-contained or the `onnx`
    package is unavailable — run.py otherwise depends only on the stdlib, so the import is lazy and
    optional and a missing package degrades to the legacy weights.bin behavior with a warning."""
    try:
        import onnx as onnx_mod
    except Exception as e:
        log(f"  [convert] WARNING: onnx package unavailable ({e}); external-data weights (if any) not pushed")
        return []
    try:
        model = onnx_mod.load(onnx, load_external_data=False)
    except Exception as e:
        log(f"  [convert] WARNING: could not parse {os.path.basename(onnx)} for external data ({e})")
        return []
    locs = []
    for init in model.graph.initializer:
        for kv in init.external_data:
            if kv.key == "location" and kv.value and kv.value not in locs:
                locs.append(kv.value)
    return locs


def convert(onnx, out_vxm, conv, where="host"):
    if not os.path.exists(onnx):
        sys.exit(f"MISSING model: {onnx}")
    flags = convert_flags(conv)
    if where == "host" and host_bin("vknn_compile"):
        log(f"  [convert] host: {os.path.basename(onnx)} -> {os.path.basename(out_vxm)}  {' '.join(flags)}")
        r = sh([host_bin("vknn_compile"), onnx, out_vxm] + flags)
        if r.returncode != 0:
            sys.exit("convert failed:\n" + r.stdout + r.stderr)
        log(f"  [convert] wrote {out_vxm}  ({human(os.path.getsize(out_vxm))})")
        return out_vxm
    need_device()
    ddir = "/data/local/tmp/vknn/bench"
    adb(["shell", "mkdir", "-p", ddir])
    log(f"  [convert] device: {os.path.basename(onnx)} -> {os.path.basename(out_vxm)}  {' '.join(flags)}")
    push(onnx, f"{ddir}/_src.onnx", "model")
    wb = os.path.join(os.path.dirname(onnx), "weights.bin")
    if os.path.exists(wb):
        push(wb, f"{ddir}/weights.bin")
    # External-data weights: an ONNX names its weight file(s) internally (e.g. "mnasnet1_0.onnx.data").
    # The importer resolves each location relative to _src.onnx's directory, so push it under the same
    # relative name into ddir; without this those weights load as 0 bytes and the .vxm is all-zero.
    onnx_dir = os.path.dirname(onnx)
    for loc in onnx_external_data(onnx):
        src = os.path.join(onnx_dir, loc)
        if os.path.exists(src):
            push(src, f"{ddir}/{loc}", "external-data")
        else:
            log(f"  [convert] WARNING: external-data file referenced by onnx not found: {src}")
    push(android_bin("vknn_compile"), f"{ddir}/vknn_compile")
    adb(["shell", "chmod", "+x", f"{ddir}/vknn_compile"])
    r = adb(["shell", f"cd {ddir} && ./vknn_compile _src.onnx {os.path.basename(out_vxm)} " + " ".join(flags)])
    if r.stdout.strip():
        vlog(r.stdout.strip())
    if r.stderr.strip():  # the engine logs (incl. errors) all go to stderr
        log("  [compile] " + r.stderr.strip().replace("\n", "\n  [compile] "))
    if r.returncode != 0:
        sys.exit(f"device convert FAILED (exit={r.returncode}): the on-device vknn_compile did not finish "
                 f"— on a large model this can be an LMK (low-memory) kill. See the [compile] output above.")
    log(f"  [convert] device wrote {os.path.basename(out_vxm)}")
    return None  # already on device under its basename


# ----------------------------------------------------------------- inputs
def expand_input_dirs(dirs, host):
    """Each directory is one input SET: <stem>.npy/.bin/.raw maps to the model input named <stem>;
    <name>_gold.npy is that set's golden for the output named <name>. Returns
    (sets, goldens, files) where sets/goldens are name->hostpath dicts per dir."""
    sets, goldens, files = [], [], []
    for d in dirs:
        hd = host(d)
        if not os.path.isdir(hd):
            sys.exit(f"MISSING input dir: {hd}")
        ins, gold = {}, {}
        for fn in sorted(os.listdir(hd)):
            stem, ext = os.path.splitext(fn)
            if ext not in (".npy", ".bin", ".raw"):
                continue
            p = os.path.join(hd, fn)
            if stem.endswith("_gold"):
                gold[stem[:-5]] = p
            else:
                ins[stem] = p
        if not ins:
            sys.exit(f"input dir has no .npy/.bin/.raw inputs: {hd}")
        sets.append(ins)
        goldens.append(gold)
        files += list(ins.values()) + list(gold.values())
    return sets, goldens, files


# ----------------------------------------------------------------- device output parsing
def parse(out):
    """Extract (times, fallback_count, ok) from the device run's stdout+stderr."""
    m = re.search(r"run median ([0-9.]+) ms\s+\(min ([0-9.]+), avg ([0-9.]+), max ([0-9.]+)", out)
    single = re.search(r"run ([0-9.]+) ms", out)
    times = None
    if m:
        times = {"median": float(m.group(1)), "min": float(m.group(2)),
                 "avg": float(m.group(3)), "max": float(m.group(4))}
    elif single:
        times = {"median": float(single.group(1))}
    load = re.search(r"load ([0-9.]+) ms", out)
    fb = re.search(r"fallbacks: (\d+)", out)
    ok = ("SOME OUTPUTS FAIL" not in out) and ("FAIL" not in out or "ALL OUTPUTS PASS" in out)
    return times, (float(load.group(1)) if load else None), (int(fb.group(1)) if fb else None), ok


# ----------------------------------------------------------------- per stage
def run_stage(stage, base, idx, run_dir, clean_cli, where_convert="host"):
    name = stage.get("name", f"stage{idx}")
    dev = stage.get("device", {})
    rc = stage.get("run", {})
    ddir = dev.get("dir", "/data/local/tmp/vknn/bench")
    set_serial(dev.get("serial"))
    need_device()

    stage_dir = os.path.join(run_dir, name)
    os.makedirs(stage_dir, exist_ok=True)

    def host(p):
        return p if os.path.isabs(p) else os.path.join(base, p)

    log(f"\n==== stage: {name} ====")
    log(f"  [device] serial={_SERIAL or '(single attached)'}  dir={ddir}")
    if clean_cli or dev.get("clean"):
        log(f"  [clean] rm -rf {ddir}")
        adb(["shell", f"rm -rf {ddir}"])
    adb(["shell", "mkdir", "-p", ddir])

    # ---- model: a .vxm runs as-is, an .onnx is converted first ----
    model = stage.get("model", "")
    if isinstance(model, dict):  # legacy {"onnx":..} / {"vxm":..}
        model = model.get("onnx") or model.get("vxm") or ""
    if not model:
        sys.exit(f"stage {name}: missing \"model\"")
    model_host = host(model)
    conv = stage.get("convert", {})
    if model.endswith(".vxm"):
        if not os.path.exists(model_host):
            sys.exit(f"MISSING model: {model_host}")
        model_name = os.path.basename(model_host)
        push(model_host, f"{ddir}/{model_name}", "model")
        local_vxm = model_host
        log(f"  [model] {model_name}  (ready vxm, no convert)")
    else:
        model_name = conv.get("out") or (os.path.splitext(os.path.basename(model_host))[0] + ".vxm")
        local_vxm = os.path.join(tempfile.gettempdir(), model_name)
        if convert(model_host, local_vxm, conv, where_convert):
            push(local_vxm, f"{ddir}/{model_name}", "model")
        log(f"  [model] {model_name}")

    # ---- inputs: file map / file list / directory set(s) ----
    inputs = stage.get("inputs")
    input_sets, golden_sets = None, None
    dcfg_inputs, dcfg_golden = None, None
    if isinstance(inputs, str):
        inputs = [inputs]
    if isinstance(inputs, list) and inputs and isinstance(inputs[0], str) and os.path.isdir(host(inputs[0])):
        sets, golds, _ = expand_input_dirs(inputs, host)
        # device names are set-prefixed: two dirs both holding "input.npy" must not collide
        input_sets, golden_sets = [], []
        for i, (s, g) in enumerate(zip(sets, golds)):
            for m, out in ((s, input_sets), (g, golden_sets)):
                dm = {}
                for k, v in m.items():
                    dn = f"set{i}_{os.path.basename(v)}"
                    push(v, f"{ddir}/{dn}", "input/golden")
                    dm[k] = dn
                out.append(dm)
            log(f"  [inputs:set{i}] " + ", ".join(f"{k}={os.path.basename(v)}" for k, v in s.items()))
        if not any(golds):
            golden_sets = None
    elif isinstance(inputs, list):
        for p in inputs:
            push(host(p), f"{ddir}/{os.path.basename(p)}", "input")
        dcfg_inputs = [os.path.basename(p) for p in inputs]
        log("  [inputs] " + ", ".join(dcfg_inputs))
    elif isinstance(inputs, dict):
        dcfg_inputs = {}
        for k, p in inputs.items():
            push(host(p), f"{ddir}/{os.path.basename(p)}", "input")
            dcfg_inputs[k] = os.path.basename(p)
        log("  [inputs] " + ", ".join(f"{k}={v}" for k, v in dcfg_inputs.items()))
    else:
        log("  [inputs] (none -> zero-filled, runtime-only)")

    golden = stage.get("golden") or stage.get("outputs", {}).get("golden")
    if golden and input_sets is None:
        dcfg_golden = {}
        for k, p in golden.items():
            push(host(p), f"{ddir}/{os.path.basename(p)}", "golden")
            dcfg_golden[k] = os.path.basename(p)
        log("  [golden] " + ", ".join(f"{k}={v}" for k, v in dcfg_golden.items()))

    # ---- device config ----
    save = stage.get("save") or stage.get("outputs", {}).get("save") or []
    dcfg = {"model": model_name,
            "backend": rc.get("backend", "vulkan"),
            "precision": rc.get("precision", "low"),
            "timing": True,
            "profile": rc.get("profile", stage.get("profile", False)),
            "tolerance": rc.get("tolerance", stage.get("tolerance", 0.999)),
            "result": "result.json", "save_dir": "."}
    for k_json, k_cfg in (("iters", "iters"), ("warmup", "warmup"), ("max_submit_nodes", "max_submit_nodes"),
                          ("winograd", "winograd"), ("tuning", "tuning"), ("fp32_tensors", "fp32_tensors"),
                          ("winogradVariant", "winogradVariant"), ("winogradUnit", "winogradUnit"),
                          ("directConv3x3", "directConv3x3"), ("generate_cache", "generate_cache")):
        v = rc.get(k_json, stage.get(k_json))
        if v is not None:
            dcfg[k_cfg] = v
    if "iters" not in dcfg and stage.get("bench"):  # legacy alias
        dcfg["iters"] = int(stage["bench"])
    if rc.get("fold_islands") is not None:
        dcfg["fold_islands"] = rc["fold_islands"]
    if rc.get("cache") or dev.get("cache"):  # unified per-model cache file (default: <model>.cache)
        dcfg["cache"] = os.path.basename(rc.get("cache") or dev.get("cache"))
    if input_sets is not None:
        dcfg["input_sets"] = input_sets
        if golden_sets:
            dcfg["golden_sets"] = golden_sets
    elif dcfg_inputs is not None:
        dcfg["inputs"] = dcfg_inputs
    if dcfg_golden:
        dcfg["golden"] = dcfg_golden
    if save:
        dcfg["save"] = save
    metrics = stage.get("metrics") or stage.get("outputs", {}).get("metrics")
    if metrics:
        dcfg["metrics"] = metrics

    log(f"  [opts] precision={dcfg['precision']} "
        f"iters={dcfg.get('iters', 1)} warmup={dcfg.get('warmup', '(auto)')} "
        f"profile={dcfg['profile']} tolerance={dcfg['tolerance']}")

    local_cfg = os.path.join(tempfile.gettempdir(), f".cfg_{name}.json")
    json.dump(dcfg, open(local_cfg, "w"), indent=2)
    vlog("  [config.json] " + json.dumps(dcfg))
    push(local_cfg, f"{ddir}/config.json")
    push(android_bin("vknn_benchmark"), f"{ddir}/vknn_benchmark")
    adb(["shell", "chmod", "+x", f"{ddir}/vknn_benchmark"])
    adb(["shell", f"rm -f {ddir}/result.json"])  # stale-result guard

    # ---- run ----
    cd = dev.get("cooldown", 0)
    if cd:
        log(f"  [cooldown] {cd}s")
        adb(["shell", "sleep", str(cd)])
    cmd = f"cd {ddir} && ./vknn_benchmark config.json"
    log(f"  [run] $ {cmd}")
    adb(["logcat", "-c"])  # start this run's logcat window clean
    r = adb(["shell", cmd])
    out = r.stdout + r.stderr
    vlog(_indent(r.stdout, "  [device] "))
    vlog(_indent(r.stderr, "  [device:err] "))
    times, load_ms, fb, ok = parse(out)

    # ---- host-side report: timing + fallbacks + validation lines ----
    if load_ms is not None:
        log(f"  [load] {load_ms:.1f} ms")
    if times:
        if "min" in times:
            log(f"  [run]  median={times['median']:.1f}  min={times['min']:.1f}  "
                f"avg={times['avg']:.1f}  max={times['max']:.1f} ms")
        else:
            log(f"  [run]  {times['median']:.1f} ms")
    else:
        log(f"  [run]  NO TIMING — the device run failed (exit={r.returncode}):")
        log(_indent((r.stderr or r.stdout).strip(), "    "))
        ok = False
    if fb is not None:
        log(f"  [fallbacks] {fb} op(s) off the requested backend" + ("" if fb == 0 else "  <-- NOT a clean release run"))
        if fb:  # the executor lists each one right after its "fallbacks:" line
            listing = out[out.find("fallbacks:"):].splitlines()[1:fb + 1]
            for ln in listing:
                log("    " + ln.strip())
    for ln in out.splitlines():
        if "cos=" in ln or "PASS" in ln or "FAIL" in ln:
            log("   " + ln.strip())

    # ---- collect: result.json + saved outputs + logcat + (optional) vxm/cache ----
    if dev_exists(f"{ddir}/result.json"):
        pr = adb(["pull", f"{ddir}/result.json", os.path.join(stage_dir, "result.json")])
        if pr.returncode == 0:
            log(f"  [result] {os.path.relpath(os.path.join(stage_dir, 'result.json'), base)}")
        else:
            log(f"  [result] pull FAILED: {(pr.stderr or pr.stdout).strip()}")
            ok = False
    else:
        log("  [result] device wrote NO result.json — the run failed before writing it")
        ok = False
    if save:
        out_dir = os.path.join(stage_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        pulled = 0
        for fmt in save:
            for f in adb(["shell", f"ls {ddir}/*.{fmt} 2>/dev/null"]).stdout.split():
                if adb(["pull", f, out_dir]).returncode == 0:
                    pulled += 1
        log(f"  [outputs] pulled {pulled} file(s) -> {os.path.relpath(out_dir, base)}/")
    for what in stage.get("pull", []):
        src = f"{ddir}/{model_name}" if what == "vxm" else f"{ddir}/{os.path.splitext(model_name)[0]}.cache"
        if what == "cache" and not dev_exists(src):
            src = f"{ddir}/{model_name}.cache"
        if dev_exists(src):
            adb(["pull", src, stage_dir])
            log(f"  [pull:{what}] {os.path.basename(src)} -> {os.path.relpath(stage_dir, base)}/")
        else:
            log(f"  [pull:{what}] not found on device: {src}")
    lc = adb(["logcat", "-d"])
    if lc.returncode == 0:
        lc_path = os.path.join(stage_dir, "logcat.txt")
        with open(lc_path, "w") as f:
            f.write(lc.stdout)
        log(f"  [logcat] {os.path.relpath(lc_path, base)}  ({human(os.path.getsize(lc_path))})")
    return ok


def main():
    global VERBOSE, BUILD
    parser = argparse.ArgumentParser(description="VKNN benchmark: convert + run + validate + profile on device")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("config")
    run_parser.add_argument("--run", default=None, help="run name (default: UTC timestamp); results land in result/<RUN>/<stage>/")
    run_parser.add_argument("--clean", action="store_true", help="delete the device run directory before each stage")
    run_parser.add_argument("--convert-on", choices=["host", "device"], default="host")
    run_parser.add_argument("-v", "--verbose", action="store_true", help="print device stdout/stderr and the generated config")
    run_parser.add_argument("--no-build", action="store_true", help="skip the automatic ./build.sh --android")

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("onnx")
    convert_parser.add_argument("out")
    convert_parser.add_argument("--fp16", action="store_true", default=True)
    convert_parser.add_argument("--fp32", dest="fp16", action="store_false")
    convert_parser.add_argument("-O", "--opt", type=int, default=1, choices=[0, 1, 2, 3], help="optimization level (default 1)")
    convert_parser.add_argument("--fuse-se", action="store_true")
    convert_parser.add_argument("--fuse-dwpw", action="store_true")
    convert_parser.add_argument("--no-fuse-swish", action="store_true")
    convert_parser.add_argument("--on", choices=["host", "device"], default="host")
    convert_parser.add_argument("--serial", default=None, help="adb device serial (for --on device with multiple devices)")
    convert_parser.add_argument("--no-build", action="store_true", help="skip the automatic ./build.sh --android before a device convert")
    convert_parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    VERBOSE = getattr(args, "verbose", False)
    BUILD = not getattr(args, "no_build", False)

    if args.cmd == "convert":
        set_serial(args.serial)
        # A device convert must push the CURRENT Android binary; android_bin() only rebuilds a MISSING
        # one, so build here first (mirrors the `run` path) unless --no-build. A stale/foreign
        # vknn_compile otherwise writes a .vxm the runner rejects as an incompatible version.
        if args.on == "device" and BUILD:
            build_android()
        convert(args.onnx, args.out, {"fp16": args.fp16, "opt": args.opt, "fuse_se": args.fuse_se,
                                      "fuse_dwpw": args.fuse_dwpw, "no_fuse_swish": args.no_fuse_swish}, args.on)
        log("done.")
        return

    cfg = json.load(open(args.config))
    # Model/input/golden paths in a config resolve against the benchmark/ root — where run.py,
    # models/, and result/ live — so a config under benchmark/configs/ still finds "models/...".
    # Absolute paths in a config are used as-is.
    base = os.path.dirname(os.path.abspath(__file__))
    run_name = args.run or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base, "result", run_name)
    os.makedirs(run_dir, exist_ok=True)
    defaults = cfg.get("defaults", {})
    stages = cfg.get("stages") or [cfg]
    log(f"config: {args.config}  ({len(stages)} stage{'s' if len(stages) != 1 else ''})  results -> result/{run_name}/")
    if BUILD:
        build_android()
    ok = True
    for i, st in enumerate(stages):
        ok = run_stage(merge(defaults, st), base, i, run_dir, args.clean, args.convert_on) and ok
    log("\n=== ALL STAGES PASS ===" if ok else "\n=== SOME STAGES FAILED ===")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
