#!/usr/bin/env python3
"""Unified VKNN benchmark driver: convert a model, push it to the device, run inference on the
device GPU (Vulkan), validate every output against a golden .npy, and report runtime.

Subcommands
-----------
  convert ONNX OUT.vxm [--fp16] [--on host|device]   compile onnx -> vxm (host by default)
  run     CONFIG.json                                 push + run once + validate on device
  bench   CONFIG.json [-n N] [--cooldown S]           N timed runs, report median submit+gpu
  all     CONFIG.json [-n N]                           convert (cfg.onnx) + run + bench

Config (benchmark.json) — extends the vknn_validate config; host paths are pushed to the device:
  {
    "onnx": "/abs/encoder.onnx",          # source for `convert`/`all`        (optional)
    "model": "encoder8_fp16.vxm",         # device model filename (convert output / what runs)
    "fp16": true,                         # convert in fp16                    (default true)
    "backend": "vulkan", "precision": "fp16",
    "device_dir": "/data/local/tmp/vxrt/bench",
    "inputs":  {"image": "image8.npy", "intrinsics": "intr8.npy"},   # or ["image8.npy", ...]
    "outputs": {"means": "means_gold.npy", "scales": "scales_gold.npy"},
    "tolerance": 0.999, "cooldown": 22, "bench": 5
  }

Run all-in-one:  benchmark.py all benchmark.json
"""
import argparse, json, os, re, statistics, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, **kw)


def adb(args, **kw):
    return sh(["adb"] + args, **kw)


def need_device():
    out = adb(["devices"]).stdout
    devs = [l.split()[0] for l in out.splitlines()[1:] if "\tdevice" in l]
    if not devs:
        sys.exit("no adb device (check `adb devices`; the phone may be asleep)")
    return devs[0]


def host_bin(name):
    p = os.path.join(ROOT, "build-host", name)
    return p if os.path.exists(p) else None


def android_bin(name):
    p = os.path.join(ROOT, "build-android", name)
    if not os.path.exists(p):
        sys.exit(f"missing {p} — build with ./build.sh --android")
    return p


# ---------------------------------------------------------------- convert
def convert(onnx, out_vxm, fp16=True, where="host"):
    flags = (["--fp16"] if fp16 else [])
    if where == "host" and host_bin("vknn_compile"):
        print(f"[convert] host: {os.path.basename(onnx)} -> {out_vxm} {'fp16' if fp16 else 'fp32'}")
        r = sh([host_bin("vknn_compile"), onnx, out_vxm] + flags)
        if r.returncode != 0:
            sys.exit("convert failed:\n" + r.stdout + r.stderr)
        return out_vxm  # local path
    # on device: push onnx + compiler, compile there
    need_device(); ddir = "/data/local/tmp/vxrt/bench"
    adb(["shell", "mkdir", "-p", ddir])
    print(f"[convert] device: pushing {os.path.basename(onnx)} + vknn_compile ...")
    adb(["push", onnx, f"{ddir}/_src.onnx"])
    # external weights file (onnxruntime/vknn external data) sits next to the onnx as weights.bin
    wb = os.path.join(os.path.dirname(onnx), "weights.bin")
    if os.path.exists(wb):
        adb(["push", wb, f"{ddir}/weights.bin"])
    adb(["push", android_bin("vknn_compile"), f"{ddir}/vknn_compile"])
    adb(["shell", "chmod", "+x", f"{ddir}/vknn_compile"])
    r = adb(["shell", f"cd {ddir} && ./vknn_compile _src.onnx {os.path.basename(out_vxm)} " + " ".join(flags)])
    print(r.stdout[-400:])
    if r.returncode != 0:
        sys.exit("device convert failed")
    return None  # already on device


# ---------------------------------------------------------------- stage + run on device
def stage(cfg, cfg_path, local_vxm=None):
    """Push model + input/golden npys + a device-side config to device_dir; return device_dir."""
    need_device()
    base = os.path.dirname(os.path.abspath(cfg_path))
    ddir = cfg.get("device_dir", "/data/local/tmp/vxrt/bench")
    adb(["shell", "mkdir", "-p", ddir])

    def host(p):
        return p if os.path.isabs(p) else os.path.join(base, p)

    # model: from a freshly-converted local vxm, else assume it is already on device (cfg.model name)
    if local_vxm:
        adb(["push", local_vxm, f"{ddir}/{cfg['model']}"])

    # collect inputs/outputs into device basenames
    def push_map(m):
        dev = {}
        items = m.items() if isinstance(m, dict) else [(None, p) for p in m]
        order = []
        for k, p in items:
            bn = os.path.basename(p)
            adb(["push", host(p), f"{ddir}/{bn}"])
            if k is None:
                order.append(bn)
            else:
                dev[k] = bn
        return order if order else dev

    dev_inputs = push_map(cfg["inputs"])
    dev_outputs = {k: os.path.basename(v) for k, v in cfg.get("outputs", {}).items()}
    for k, v in cfg.get("outputs", {}).items():
        adb(["push", host(v), f"{ddir}/{os.path.basename(v)}"])

    dcfg = {
        "model": cfg["model"], "backend": cfg.get("backend", "vulkan"),
        "precision": cfg.get("precision", "fp16"), "no_weight_cache": True,
        "timing": True, "tolerance": cfg.get("tolerance", 0.999),
        "inputs": dev_inputs, "outputs": dev_outputs,
    }
    local_dcfg = os.path.join(tempfile.gettempdir(), ".bench_device_config.json")
    json.dump(dcfg, open(local_dcfg, "w"), indent=2)
    adb(["push", local_dcfg, f"{ddir}/config.json"])
    adb(["push", android_bin("vknn_validate"), f"{ddir}/vknn_validate"])
    adb(["shell", "chmod", "+x", f"{ddir}/vknn_validate"])
    return ddir


def run_once(ddir, cooldown=0):
    if cooldown:
        adb(["shell", "sleep", str(cooldown)])
    r = adb(["shell", f"cd {ddir} && ./vknn_validate config.json"])
    return r.stdout + r.stderr


def parse(out):
    sg = re.search(r"submit\+gpu=([0-9.]+)ms", out)
    rows = re.findall(r"^\s+(\S+)\s+cos=([0-9.\-]+).*?(PASS|FAIL|SIZE-MISMATCH)", out, re.M)
    allpass = "ALL OUTPUTS PASS" in out
    return (float(sg.group(1)) if sg else None), rows, allpass


# ---------------------------------------------------------------- commands
def cmd_run(cfg, cfg_path, local_vxm=None):
    ddir = stage(cfg, cfg_path, local_vxm)
    out = run_once(ddir, cfg.get("cooldown", 0))
    sg, rows, ok = parse(out)
    for name, cos, verdict in rows:
        print(f"  {name:16} cos={cos}  {verdict}")
    print(f"submit+gpu = {sg:.1f} ms" if sg else "(no timing)")
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def cmd_bench(cfg, cfg_path, n, local_vxm=None):
    ddir = stage(cfg, cfg_path, local_vxm)
    cd = cfg.get("cooldown", 22)
    times, ok_all = [], True
    for i in range(n):
        sg, rows, ok = parse(run_once(ddir, cd))
        ok_all = ok_all and ok
        if sg:
            times.append(sg)
        print(f"  run {i+1}/{n}: submit+gpu = {sg:.1f} ms  {'PASS' if ok else 'FAIL'}")
    if times:
        print(f"\nsubmit+gpu: min={min(times):.1f}  median={statistics.median(times):.1f}  max={max(times):.1f} ms  (n={len(times)})")
    print("ACCURACY:", "PASS" if ok_all else "FAIL")
    return ok_all


def main():
    ap = argparse.ArgumentParser(description="VKNN benchmark driver (convert + run + validate on device)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert"); c.add_argument("onnx"); c.add_argument("out"); c.add_argument("--fp16", action="store_true", default=True); c.add_argument("--fp32", dest="fp16", action="store_false"); c.add_argument("--on", choices=["host", "device"], default="host")
    r = sub.add_parser("run"); r.add_argument("config")
    b = sub.add_parser("bench"); b.add_argument("config"); b.add_argument("-n", type=int, default=5); b.add_argument("--cooldown", type=int, default=None)
    a = sub.add_parser("all"); a.add_argument("config"); a.add_argument("-n", type=int, default=5)
    args = ap.parse_args()

    if args.cmd == "convert":
        convert(args.onnx, args.out, args.fp16, args.on)
        print("done.")
        return

    cfg = json.load(open(args.config))
    if args.cmd in ("bench", "all") and getattr(args, "cooldown", None) is not None:
        cfg["cooldown"] = args.cooldown

    local_vxm = None
    if args.cmd == "all" and cfg.get("onnx"):
        local_vxm = os.path.join(tempfile.gettempdir(), cfg["model"])
        convert(cfg["onnx"], local_vxm, cfg.get("fp16", True), "host")

    if args.cmd == "run":
        sys.exit(0 if cmd_run(cfg, args.config) else 3)
    else:  # bench, all
        sys.exit(0 if cmd_bench(cfg, args.config, args.n, local_vxm) else 3)


if __name__ == "__main__":
    main()
