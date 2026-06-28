#!/usr/bin/env python3
"""Build a vknn_benchmark config (+ ORT golden .npy files) from an ONNX model and input .npy files.

The companion of the vknn_benchmark tool: it runs the model once in onnxruntime to produce a golden
.npy per output, then writes a config.json that vknn_benchmark consumes to run the same inputs on the
device (Vulkan) and check every output against its golden.

Usage:
  make_validate.py MODEL.onnx OUT_DIR  name0=in0.npy [name1=in1.npy ...]  [--model-on-device NAME]
                   [--backend vulkan|cpu] [--precision fp16|fp32] [--tol 0.999]

Then:  adb push OUT_DIR/. /data/local/tmp/vxrt/yono/   &&   vknn_benchmark config.json
"""
import sys, os, json, argparse
import numpy as np
import onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out_dir")
ap.add_argument("inputs", nargs="+", help="NAME=path.npy per model input")
ap.add_argument("--model-on-device", default=None, help="model filename as referenced in config (default: basename of model)")
ap.add_argument("--backend", default="vulkan")
ap.add_argument("--precision", default="fp16")
ap.add_argument("--tol", type=float, default=0.999)
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
feeds, in_files = {}, {}
for spec in args.inputs:
    name, path = spec.split("=", 1)
    arr = np.load(path).astype(np.float32)
    feeds[name] = arr
    dst = os.path.join(args.out_dir, f"{name}.npy")
    np.save(dst, arr)
    in_files[name] = f"{name}.npy"
    print(f"input  {name:14} {arr.shape}  -> {dst}")

so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
out_names = [o.name for o in sess.get_outputs()]
res = sess.run(out_names, feeds)
golden = {}
for name, arr in zip(out_names, res):
    arr = np.ascontiguousarray(arr, np.float32)
    np.save(os.path.join(args.out_dir, f"{name}_gold.npy"), arr)
    golden[name] = f"{name}_gold.npy"
    print(f"golden {name:14} {arr.shape}  mean={arr.mean():+.4f} nan={int(np.isnan(arr).sum())}")

cfg = {
    "model": args.model_on_device or os.path.basename(args.model),
    "backend": args.backend,
    "precision": args.precision,
    "no_weight_cache": True,
    "tolerance": args.tol,
    "inputs": in_files,   # keyed by input name
    "outputs": golden,
}
json.dump(cfg, open(os.path.join(args.out_dir, "config.json"), "w"), indent=2)
print(f"\nwrote {args.out_dir}/config.json  ({len(in_files)} inputs, {len(golden)} goldens)")
print(f"next: adb push {args.out_dir}/. /data/local/tmp/vxrt/yono/ && (cd .../yono && ../vknn_benchmark config.json)")
