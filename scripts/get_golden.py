#!/usr/bin/env python3
"""Produce onnxruntime golden outputs for vxrt validation.

Outputs (under assets/):
  input.bin        fixed preprocessed input, raw float32 NCHW [1,3,224,224]
  input.npy        same, numpy
  golden.npy       final output [1,1000]
  golden_top5.txt  top-5 (idx, score)
  layers/<name>.npy per-layer outputs (every node output) for layer-dump diffing

Usage: get_golden.py [--image PATH] [--model assets/mobilenetv2.onnx]
"""
import argparse, os, sys
import numpy as np

def preprocess_image(path):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))[None]  # NCHW
    return np.ascontiguousarray(x, dtype=np.float32)

def synthetic_input(seed=1234):
    # Deterministic, smooth, normalized-looking input so argmax is stable & meaningful.
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
    # low-pass it a little so it's not pure noise
    k = np.ones((1, 1, 3, 3), np.float32) / 9.0
    import scipy.signal as ss  # optional; fall back if missing
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="assets/mobilenetv2.onnx")
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", default="assets")
    ap.add_argument("--layers", action="store_true", help="also dump per-layer goldens")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.image and os.path.exists(args.image):
        x = preprocess_image(args.image)
        print(f"input: image {args.image}")
    else:
        rng = np.random.default_rng(1234)
        x = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
        print("input: deterministic synthetic (seed 1234)")

    x.tofile(os.path.join(args.out, "input.bin"))
    np.save(os.path.join(args.out, "input.npy"), x)

    import onnxruntime as ort
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    y = sess.run(None, {iname: x})[0]
    np.save(os.path.join(args.out, "golden.npy"), y)
    np.ascontiguousarray(y, dtype=np.float32).tofile(os.path.join(args.out, "golden.bin"))
    top5 = np.argsort(-y[0])[:5]
    with open(os.path.join(args.out, "golden_top5.txt"), "w") as f:
        for i in top5:
            f.write(f"{int(i)}\t{float(y[0][i]):.6f}\n")
    print("final output shape", y.shape, "top-1:", int(top5[0]), "score", float(y[0][top5[0]]))
    print("top5:", [int(i) for i in top5])

    if args.layers:
        import onnx
        m = onnx.load(args.model)
        # expose every node output as a graph output
        names = []
        for n in m.graph.node:
            for o in n.output:
                if o:
                    names.append(o)
        from onnx import helper
        vis = [helper.make_empty_tensor_value_info(nm) for nm in names]
        del m.graph.output[:]
        m.graph.output.extend(vis)
        tmp = os.path.join(args.out, "_allout.onnx")
        onnx.save(m, tmp)
        s2 = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
        outs = s2.run(names, {iname: x})
        ldir = os.path.join(args.out, "layers")
        os.makedirs(ldir, exist_ok=True)
        for nm, arr in zip(names, outs):
            safe = nm.replace("/", "_").replace(":", "_")
            np.save(os.path.join(ldir, safe + ".npy"), np.asarray(arr))
        os.remove(tmp)
        print(f"dumped {len(names)} per-layer goldens -> {ldir}")

if __name__ == "__main__":
    main()
