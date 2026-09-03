#!/usr/bin/env python3
"""Per-op DRAM-traffic audit: join a model's ONNX graph with a `--profile` log.

For every op the profile lists, the minimal bytes it must move (fp16 NC4HW4 activations with the
channel padding the engine stores, fp16 weights) are set against its measured GPU time, giving the
bandwidth it achieved and, for the MAC-bearing ops, its arithmetic rate. Ops far below both the
device's stream rate and its arithmetic ceiling are latency- or launch-bound and the ones to
restructure or fuse; ops at the stream rate are as fast as their traffic allows and only a traffic
reduction (layout, fusion, precision) moves them. Engine-inserted ops (layout converts, fused
pointwise units) carry no ONNX node; they are listed with the graph-output traffic where it is
known and marked otherwise.

  traffic_audit.py model.onnx profile.log [--top N] [--csv out.csv]
"""
import argparse
import collections
import re
import sys

import onnx
from onnx import shape_inference

FP16_BYTES = 2
NC4_BLOCK = 4
PROFILE_NAME_WIDTH = 28  # the profile table truncates op names to this many characters
MAC_OPS = ("Conv", "ConvTranspose", "Gemm", "MatMul")


def nc4_bytes(shape):
    """Bytes of an fp16 tensor in the engine's storage: NC4HW4 (channels padded to 4) for rank >= 3."""
    dims = [d if d > 0 else 1 for d in shape]
    if len(dims) >= 3:
        n, c = dims[0], dims[1]
        rest = 1
        for d in dims[2:]:
            rest *= d
        blocks = (c + NC4_BLOCK - 1) // NC4_BLOCK
        return n * blocks * NC4_BLOCK * rest * FP16_BYTES
    elems = 1
    for d in dims:
        elems *= d
    return elems * FP16_BYTES


def load_graph(path):
    model = shape_inference.infer_shapes(onnx.load(path))
    graph = model.graph
    shapes = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    inits = {i.name: list(i.dims) for i in graph.initializer}
    return graph, shapes, inits


def node_traffic(node, shapes, inits):
    """(activation bytes in, weight bytes, activation bytes out, MACs) for one ONNX node."""
    act_in = weights = act_out = 0
    for name in node.input:
        if name in inits:
            elems = 1
            for d in inits[name]:
                elems *= d
            weights += elems * FP16_BYTES
        elif name in shapes:
            act_in += nc4_bytes(shapes[name])
    for name in node.output:
        if name in shapes:
            act_out += nc4_bytes(shapes[name])
    macs = 0
    if node.op_type in ("Conv", "ConvTranspose") and node.input[1] in inits and node.output[0] in shapes:
        w = inits[node.input[1]]
        y = shapes[node.output[0]]
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        group = attrs.get("group", 1)
        taps = 1
        for d in w[2:]:
            taps *= d
        spatial = 1
        for d in y[2:]:
            spatial *= d
        macs = (w[1] if node.op_type == "Conv" else w[0]) * w[0] // (1 if node.op_type == "Conv" else group) * taps * spatial
        if node.op_type == "Conv":
            macs = w[0] * w[1] * taps * spatial  # w[1] is already Cin/group
    elif node.op_type in ("Gemm", "MatMul") and node.output[0] in shapes:
        a = shapes.get(node.input[0], [])
        y = shapes[node.output[0]]
        k = a[-1] if a else 0
        out = 1
        for d in y:
            out *= max(d, 1)
        macs = out * k
    return act_in, weights, act_out, macs


def parse_profile(path):
    ops = []
    for line in open(path, errors="replace"):
        m = re.match(r"^(\S+)\s+(Vulkan|CPU)\s+(\S+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$", line.rstrip("\n"))
        if m:
            ops.append(dict(name=m.group(1), backend=m.group(2), type=m.group(3), cpu_ms=float(m.group(4)), gpu_ms=float(m.group(5)), dispatches=int(m.group(6))))
    return ops


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("profile")
    ap.add_argument("--top", type=int, default=40, help="ops to print, by GPU time (default 40)")
    ap.add_argument("--csv", default="", help="write every op's row to this CSV")
    args = ap.parse_args()
    graph, shapes, inits = load_graph(args.model)
    by_prefix = {}
    for node in graph.node:
        by_prefix[node.name[:PROFILE_NAME_WIDTH]] = node
    graph_out_bytes = sum(nc4_bytes(shapes[o.name]) for o in graph.output if o.name in shapes)
    rows = []
    for op in parse_profile(args.profile):
        node = by_prefix.get(op["name"])
        if node is not None:
            act_in, weights, act_out, macs = node_traffic(node, shapes, inits)
            origin = node.op_type
        elif op["name"].startswith("convertout"):
            act_in, weights, act_out, macs, origin = graph_out_bytes, 0, graph_out_bytes * 2, 0, "engine"  # fp16 NC4 in, fp32 flat out
        else:
            act_in = weights = act_out = macs = 0
            origin = "engine"
        floor = act_in + weights + act_out
        ms = op["gpu_ms"]
        if op["dispatches"] == 0:
            # A zero-copy view the planner elided: no dispatch, no traffic. The profiler still
            # brackets it and reports whatever the GPU was draining between its neighbours, so its
            # time is dropped from every sum here rather than counted as work.
            floor, ms, macs, origin = 0, 0.0, 0, "elided view"
        rows.append(dict(op, gpu_ms=ms, origin=origin, floor_mb=floor / 1e6, weights_mb=weights / 1e6, gbps=(floor / 1e6) / ms if ms > 0 else 0.0, gmac=macs / 1e9, tflops=(2 * macs / 1e12) / (ms / 1e3) if ms > 0 and macs else 0.0))
    rows.sort(key=lambda r: -r["gpu_ms"])
    elided = sum(1 for r in rows if r["origin"] == "elided view")
    total_ms = sum(r["gpu_ms"] for r in rows)
    total_mb = sum(r["floor_mb"] for r in rows)
    print(f"{'op':30s} {'type':14s} {'gpu ms':>7s} {'floor MB':>9s} {'GB/s':>6s} {'GMAC':>7s} {'TFLOPS':>7s}  note")
    for r in rows[: args.top]:
        note = r["origin"] if r["origin"] in ("elided view",) else "" if r["origin"] != "engine" else "engine-inserted" if r["floor_mb"] == 0 else "engine convert"
        print(f"{r['name'][:30]:30s} {r['type'][:14]:14s} {r['gpu_ms']:7.3f} {r['floor_mb']:9.2f} {r['gbps']:6.1f} {r['gmac']:7.3f} {r['tflops']:7.2f}  {note}")
    print(f"TOTAL {len(rows)} ops ({elided} elided views, not counted): {total_ms:.3f} ms of summed op time, {total_mb:.1f} MB of floor traffic, {total_mb / total_ms if total_ms else 0:.1f} GB/s aggregate")
    by_type = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for r in rows:
        t = by_type[r["type"]]
        t[0] += 1
        t[1] += r["gpu_ms"]
        t[2] += r["floor_mb"]
    print("per type:")
    for typ, (n, ms, mb) in sorted(by_type.items(), key=lambda kv: -kv[1][1]):
        print(f"  {typ:16s} n={n:3d} {ms:7.3f} ms {mb:8.1f} MB {mb / ms if ms else 0:6.1f} GB/s")
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
