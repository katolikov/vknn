#!/usr/bin/env python3
"""Compare VKNN per-layer dumps (.bin, float32, NCHW) against onnxruntime goldens (.npy).

Usage: compare_layers.py <dump_dir> <golden_layers_dir>
Prints per-layer cosine + max abs err, sorted worst-first, to locate first divergence.
"""
import os, sys, glob
import numpy as np

def cosine(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 1.0 if na == nb else 0.0
    return float(a.dot(b) / (na * nb))

def main():
    dump, gold = sys.argv[1], sys.argv[2]
    rows = []
    for gp in glob.glob(os.path.join(gold, "*.npy")):
        name = os.path.splitext(os.path.basename(gp))[0]
        bp = os.path.join(dump, name + ".bin")
        if not os.path.exists(bp): continue
        g = np.load(gp).astype(np.float32)
        v = np.fromfile(bp, dtype=np.float32)
        c = cosine(v, g)
        maxerr = float(np.max(np.abs(v[:min(len(v),g.size)] - g.ravel()[:min(len(v),g.size)]))) if g.size else 0
        rows.append((c, maxerr, name, g.shape, v.size))
    rows.sort(key=lambda r: r[0])
    print(f"{'cosine':>9} {'maxErr':>11}  {'layer':<14} golden_shape / vknn_elems")
    for c, e, name, gs, ve in rows[:25]:
        print(f"{c:9.5f} {e:11.4e}  {name:<14} {gs} / {ve}")
    print(f"... {len(rows)} layers compared; worst {min(r[0] for r in rows):.5f}" if rows else "no matches")

if __name__ == "__main__":
    main()
