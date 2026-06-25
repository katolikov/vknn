#!/usr/bin/env python3
"""Export a torchvision classification model to ONNX + a deterministic input + onnxruntime golden.

Outputs under assets/:  <name>.onnx, <name>_in.bin, <name>_gold.bin (raw float32), <name>_gold.npy

Usage: export_tv.py --model efficientnet_b0 [--size 224] [--no-pretrained]
"""
import argparse, os
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="torchvision model name, e.g. efficientnet_b0")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--out", default="assets")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch, torchvision
    ctor = getattr(torchvision.models, args.model)
    weights = None
    if not args.no_pretrained:
        try:
            weights = "IMAGENET1K_V1"
            net = ctor(weights=weights)
            print(f"loaded pretrained {weights}")
        except Exception as e:
            print(f"pretrained unavailable ({e}); using random init")
            weights = None
            net = ctor(weights=None)
    else:
        net = ctor(weights=None)
    net.eval()

    rng = np.random.default_rng(1234)
    x = rng.standard_normal((1, 3, args.size, args.size)).astype(np.float32)
    onnx_path = os.path.join(args.out, f"{args.model}.onnx")
    xt = torch.from_numpy(x)
    torch.onnx.export(net, xt, onnx_path, input_names=["input"], output_names=["output"],
                      opset_version=args.opset, do_constant_folding=True)
    print(f"exported {onnx_path}")

    x.tofile(os.path.join(args.out, f"{args.model}_in.bin"))

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    y = sess.run(None, {iname: x})[0]
    np.save(os.path.join(args.out, f"{args.model}_gold.npy"), y)
    np.ascontiguousarray(y, dtype=np.float32).tofile(os.path.join(args.out, f"{args.model}_gold.bin"))
    top5 = np.argsort(-y[0])[:5]
    print("output shape", y.shape, "top-1:", int(top5[0]), "score", float(y[0][top5[0]]))

    # op-type histogram (so we can see what the importer/fuser will face)
    import onnx
    from collections import Counter
    m = onnx.load(onnx_path)
    c = Counter(n.op_type for n in m.graph.node)
    print("nodes:", len(m.graph.node), "ops:", dict(sorted(c.items(), key=lambda kv: -kv[1])))

if __name__ == "__main__":
    main()
