#!/usr/bin/env python3
"""Unified VKNN benchmark driver.

One JSON config describes one or more *stages*; each stage runs independently on the device:
provide either an ONNX model (converted to .vxm with the given optimization options) or a ready
.vxm (runs as-is), feed .npy or raw .bin inputs (or none, for a runtime-only measurement), optionally
save outputs as .npy / .png, compare each output against a golden (cosine / PSNR / SNR / relL2 / max),
and collect a per-stage result.json with timing and (optional) per-operator profiling.

  run.py run    CONFIG.json          # run every stage (convert if needed, on device)
  run.py convert ONNX OUT.vxm [opts] # standalone convert (host or device)

Config (sectioned; see benchmark/example.json and USAGE.md):
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
    return subprocess.run(cmd, capture_output=True, text=True)


# Target device (adb serial / id). Set per-stage from device.serial so a host with several phones
# attached is unambiguous; None = whatever single device adb finds.
_SERIAL = None


def set_serial(s):
    global _SERIAL
    _SERIAL = s or None


def adb(args):
    return sh(["adb"] + (["-s", _SERIAL] if _SERIAL else []) + args)


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


def android_bin(name):
    p = os.path.join(ROOT, "build-android", name)
    if not os.path.exists(p):
        sys.exit(f"missing {p} — build with ./build.sh --android")
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
        print(f"[convert] host: {os.path.basename(onnx)} -> {os.path.basename(out_vxm)} {' '.join(flags)}")
        r = sh([host_bin("vknn_compile"), onnx, out_vxm] + flags)
        if r.returncode != 0:
            sys.exit("convert failed:\n" + r.stdout + r.stderr)
        return out_vxm
    need_device()
    ddir = "/data/local/tmp/vxrt/bench"
    adb(["shell", "mkdir", "-p", ddir])
    print(f"[convert] device: pushing {os.path.basename(onnx)} + compiler ...")
    adb(["push", onnx, f"{ddir}/_src.onnx"])
    wb = os.path.join(os.path.dirname(onnx), "weights.bin")
    if os.path.exists(wb):
        adb(["push", wb, f"{ddir}/weights.bin"])
    adb(["push", android_bin("vknn_compile"), f"{ddir}/vknn_compile"])
    adb(["shell", "chmod", "+x", f"{ddir}/vknn_compile"])
    r = adb(["shell", f"cd {ddir} && ./vknn_compile _src.onnx {os.path.basename(out_vxm)} " + " ".join(flags)])
    print(r.stdout[-300:])
    if r.returncode != 0:
        sys.exit("device convert failed")
    return None  # already on device under its basename


# ----------------------------------------------------------------- per stage
def parse(out):
    sg = re.search(r"submit\+gpu=([0-9.]+)ms", out)
    run = re.search(r"run ([0-9.]+) ms", out)
    rows = re.findall(r"^\s+(\S+)\s+(cos=[0-9.\-]+.*?)(PASS|FAIL|SIZE-MISMATCH|NOT-AN-OUTPUT)\s*$", out, re.M)
    ok = ("ALL OUTPUTS PASS" in out) or ("ALL OUTPUTS PASS" not in out and "FAIL" not in out and "golden" not in out)
    t = float(sg.group(1)) if sg else (float(run.group(1)) if run else None)
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

    print(f"\n==== stage: {name} ====")
    # ---- model: convert onnx, or push a ready vxm ----
    if model.get("vxm"):
        vxm = host(model["vxm"]); model_name = os.path.basename(vxm)
        adb(["push", vxm, f"{ddir}/{model_name}"])
        print(f"[model] vxm (no convert): {model_name}")
    elif model.get("onnx"):
        onnx = host(model["onnx"]); conv = stage.get("convert", {})
        model_name = conv.get("out") or (os.path.splitext(os.path.basename(onnx))[0] + ".vxm")
        local = os.path.join(tempfile.gettempdir(), model_name)
        res = convert(onnx, local, conv, where_convert)
        if res:  # converted on host -> push
            adb(["push", local, f"{ddir}/{model_name}"])
    else:
        sys.exit(f"stage {name}: model needs \"onnx\" or \"vxm\"")

    # ---- inputs (npy/raw) + goldens ----
    def push_io(m):
        if isinstance(m, list):
            for p in m:
                adb(["push", host(p), f"{ddir}/{os.path.basename(p)}"])
            return [os.path.basename(p) for p in m]
        dev_m = {}
        for k, p in (m or {}).items():
            adb(["push", host(p), f"{ddir}/{os.path.basename(p)}"])
            dev_m[k] = os.path.basename(p)
        return dev_m

    out_cfg = stage.get("outputs", {})
    dcfg = {"model": model_name, "backend": dev.get("backend", "vulkan"),
            "precision": dev.get("precision", "fp16"), "no_weight_cache": dev.get("no_weight_cache", True),
            "timing": True, "profile": stage.get("profile", False),
            "tolerance": stage.get("tolerance", 0.999), "result": "result.json", "save_dir": "."}
    if "max_submit_nodes" in dev:
        dcfg["max_submit_nodes"] = dev["max_submit_nodes"]
    if stage.get("inputs"):
        dcfg["inputs"] = push_io(stage["inputs"])
    if out_cfg.get("save"):
        dcfg["save"] = out_cfg["save"]
    if out_cfg.get("golden"):
        dcfg["golden"] = push_io(out_cfg["golden"])
    if out_cfg.get("metrics"):
        dcfg["metrics"] = out_cfg["metrics"]

    local_cfg = os.path.join(tempfile.gettempdir(), f".cfg_{name}.json")
    json.dump(dcfg, open(local_cfg, "w"), indent=2)
    adb(["push", local_cfg, f"{ddir}/config.json"])
    adb(["push", android_bin("vknn_benchmark"), f"{ddir}/vknn_benchmark"])
    adb(["shell", "chmod", "+x", f"{ddir}/vknn_benchmark"])

    # ---- run (bench times) ----
    n = stage.get("bench", 1)
    cd = dev.get("cooldown", 22 if n > 1 else 0)
    times, ok_all, last = [], True, ""
    for i in range(n):
        if cd:
            adb(["shell", "sleep", str(cd)])
        last = adb(["shell", f"cd {ddir} && ./vknn_benchmark config.json"]).stdout
        t, rows, ok = parse(last)
        ok_all = ok_all and ok
        if t:
            times.append(t)
        tag = "" if n == 1 else f"run {i+1}/{n}: "
        print(f"  {tag}submit+gpu={t:.1f} ms" if t else f"  {tag}(no timing)")
    for ln in last.splitlines():
        if "cos=" in ln or "PASS" in ln or "FAIL" in ln or "no inputs" in ln:
            print("   " + ln.strip())
    if len(times) > 1:
        print(f"  -> min={min(times):.1f} median={statistics.median(times):.1f} max={max(times):.1f} ms")

    # ---- pull result.json + saved outputs ----
    results_dir = os.path.join(base, "results")
    os.makedirs(results_dir, exist_ok=True)
    rj = stage.get("result", f"{name}.result.json")
    adb(["pull", f"{ddir}/result.json", os.path.join(results_dir, os.path.basename(rj))])
    if out_cfg.get("save"):
        for fmt in out_cfg["save"]:
            adb(["shell", f"ls {ddir}/*.{fmt} 2>/dev/null"])  # leave artifacts on device; pull on request
    print(f"  result -> results/{os.path.basename(rj)}")
    return ok_all


def main():
    ap = argparse.ArgumentParser(description="VKNN benchmark: convert + run + validate + profile on device")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("config"); r.add_argument("--convert-on", choices=["host", "device"], default="host")
    c = sub.add_parser("convert"); c.add_argument("onnx"); c.add_argument("out")
    c.add_argument("--fp16", action="store_true", default=True); c.add_argument("--fp32", dest="fp16", action="store_false")
    c.add_argument("--fuse-se", action="store_true"); c.add_argument("--fuse-dwpw", action="store_true")
    c.add_argument("--no-fuse-swish", action="store_true"); c.add_argument("--on", choices=["host", "device"], default="host")
    c.add_argument("--serial", default=None, help="adb device serial (for --on device with multiple devices)")
    args = ap.parse_args()

    if args.cmd == "convert":
        set_serial(args.serial)
        convert(args.onnx, args.out, {"fp16": args.fp16, "fuse_se": args.fuse_se, "fuse_dwpw": args.fuse_dwpw,
                                      "no_fuse_swish": args.no_fuse_swish}, args.on)
        print("done.")
        return

    cfg = json.load(open(args.config))
    base = os.path.dirname(os.path.abspath(args.config))
    defaults = cfg.get("defaults", {})
    stages = cfg.get("stages") or [cfg]
    ok = True
    for i, st in enumerate(stages):
        ok = run_stage(merge(defaults, st), base, i, args.convert_on) and ok
    print("\n=== ALL STAGES PASS ===" if ok else "\n=== SOME STAGES FAILED ===")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
