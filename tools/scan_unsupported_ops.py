#!/usr/bin/env python3
"""Scan an ONNX model for operators VKNN does not yet support.

VKNN's single source of truth for "which ONNX op is recognized" is
`opTypeFromOnnx()` in src/core/op.cpp: any op name it does not map falls through
to OpType::kUnknown and cannot run. This script reads that file directly (so it
stays in sync with the engine), loads the model, runs shape inference, and emits
a report in two buckets, each with the shapes/types/attributes you'd need:

  1. UNSUPPORTED  -- op type unknown to VKNN (or custom domain): the model cannot
                     run until the op is implemented.
  2. GPU FALLBACK -- op type is recognized but its attribute variant has no GPU
                     kernel, so it runs on the (correct) CPU op. Prime targets
                     for a GPU kernel. Only attribute-checkable gates from
                     vk_backend.cpp::supportsNode are detected (see note below).

The intent: hand the resulting report (text or --json) to Claude Code as a
self-contained brief for implementing the missing operators.

Usage:
  tools/scan_unsupported_ops.py model.onnx
  tools/scan_unsupported_ops.py model.onnx --json report.json
  tools/scan_unsupported_ops.py model.onnx --src src/core/op.cpp

Requires: onnx (pip install onnx). Pure analysis; it never modifies the model.
"""
import argparse
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = os.path.join(REPO_ROOT, "src", "core", "op.cpp")

# Op names that op.cpp accepts but that never appear in an ONNX export (they are
# VKNN-internal fusion/layout pseudo-ops or the placeholder). Excluding them from
# the "supported" set is harmless for scanning and keeps the report honest.
_INTERNAL_NAMES = {"FusedSE", "FusedDwPw", "ConvertLayout", "Unknown", "Unary", "Binary"}


def supported_op_names(src_path):
    """Derive the set of supported ONNX op names from op.cpp.

    Every ONNX name the engine recognizes appears as a double-quoted string in
    op.cpp -- as a key in the opTypeFromOnnx map, in the ReduceSum/Max/... branch,
    or as a key of unaryFromOnnx / binaryFromOnnx. Collecting all quoted
    identifiers yields exactly that union; we drop the handful of internal names.
    """
    try:
        with open(src_path, "r") as f:
            text = f.read()
    except OSError as e:
        sys.exit("error: cannot read VKNN op source %s: %s\n"
                 "       pass the correct path with --src" % (src_path, e))
    names = set(re.findall(r'"([A-Za-z][A-Za-z0-9]*)"', text))
    names -= _INTERNAL_NAMES
    if not names:
        sys.exit("error: found no op names in %s -- is this the right file?" % src_path)
    return names


def dtype_name(elem_type):
    try:
        from onnx import TensorProto
        return TensorProto.DataType.Name(elem_type)
    except Exception:
        return str(elem_type)


def shape_of_type(type_proto):
    """Return (shape_list, dtype_str) for a ValueInfo/Tensor type, or (None, None)."""
    if type_proto is None or not type_proto.HasField("tensor_type"):
        return None, None
    tt = type_proto.tensor_type
    dt = dtype_name(tt.elem_type)
    if not tt.HasField("shape"):
        return None, dt
    dims = []
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        elif d.HasField("dim_param") and d.dim_param:
            dims.append(d.dim_param)  # symbolic, e.g. "batch"
        else:
            dims.append("?")
    return dims, dt


def build_value_index(graph):
    """Map every tensor name in a graph to {shape, dtype, source}."""
    from onnx import numpy_helper  # noqa: F401  (import validates onnx presence)
    index = {}
    for collection, source in ((graph.input, "input"),
                               (graph.value_info, "value_info"),
                               (graph.output, "output")):
        for vi in collection:
            shape, dt = shape_of_type(vi.type)
            index[vi.name] = {"shape": shape, "dtype": dt, "source": source}
    for init in graph.initializer:
        index[init.name] = {
            "shape": [int(d) for d in init.dims],
            "dtype": dtype_name(init.data_type),
            "source": "initializer",
        }
    return index


def attr_to_jsonable(attr):
    """Compact, lossless-enough rendering of an AttributeProto for the brief."""
    from onnx import AttributeProto, helper
    t = attr.type
    A = AttributeProto
    if t == A.INT:
        return attr.i
    if t == A.FLOAT:
        return attr.f
    if t == A.STRING:
        return attr.s.decode("utf-8", "replace")
    if t == A.INTS:
        return list(attr.ints)
    if t == A.FLOATS:
        return list(attr.floats)
    if t == A.STRINGS:
        return [s.decode("utf-8", "replace") for s in attr.strings]
    if t == A.TENSOR:
        return {"_tensor": {"dtype": dtype_name(attr.t.data_type),
                            "shape": [int(d) for d in attr.t.dims]}}
    if t in (A.GRAPH, A.GRAPHS):
        return "<subgraph>"
    try:
        return helper.get_attribute_value(attr)
    except Exception:
        return "<%s>" % A.AttributeType.Name(t)


def subgraphs_of(node):
    from onnx import AttributeProto
    for attr in node.attribute:
        if attr.type == AttributeProto.GRAPH:
            yield attr.g
        elif attr.type == AttributeProto.GRAPHS:
            for g in attr.graphs:
                yield g


def io_details(names, index):
    out = []
    for n in names:
        if n == "":  # optional, omitted input
            out.append({"name": "", "shape": None, "dtype": None, "source": "optional"})
            continue
        info = index.get(n, {"shape": None, "dtype": None, "source": "unknown"})
        out.append({"name": n, "shape": info["shape"], "dtype": info["dtype"],
                    "source": info["source"]})
    return out


# ---------------------------------------------------------------------------
# Attribute-level gates: op types VKNN *recognizes* but whose specific attribute
# variant has no GPU kernel, so the node falls back to the (correct) CPU op. The
# model still runs, but these are the prime targets for "implement a GPU path".
#
# Only attribute-checkable gates from vk_backend.cpp::supportsNode are encoded
# here -- they need nothing but the node's own attributes, so they're reliable
# from a raw ONNX file. The many shape/initializer/layout-dependent gates in
# supportsNode (MatMul rank, DepthToSpace divisibility, Resize channel/batch,
# const-initializer requirements, ...) are intentionally NOT replicated: they
# depend on post-pass internal state and would produce false positives.
_FLOAT_DTYPES = {"FLOAT", "FLOAT16", "DOUBLE", "BFLOAT16"}


def _pad_fallback(attrs):
    mode = attrs.get("mode", "constant")
    if mode not in ("constant", "edge", "reflect"):
        return "Pad mode=%r has no GPU kernel (GPU does only constant/edge/reflect)" % mode
    return None


def _gridsample_fallback(attrs):
    mode = attrs.get("mode", "bilinear")
    if mode not in ("bilinear", "linear", "nearest"):
        return ("GridSample mode=%r has no GPU kernel (GPU does only "
                "bilinear/linear/nearest; e.g. cubic falls back)" % mode)
    return None


def _cast_fallback(attrs):
    to = attrs.get("to")
    to_name = dtype_name(to) if isinstance(to, int) else str(to)
    if to_name not in _FLOAT_DTYPES:
        return ("Cast to=%s is a non-float target (GPU path is float->float only)" % to_name)
    return None


def _einsum_fallback(attrs):
    eq = "".join(c for c in attrs.get("equation", "") if c not in " \t")
    if eq != "i,j->ij":
        return ("Einsum equation=%r has no GPU kernel (only 'i,j->ij' outer product)" % eq)
    return None


GPU_FALLBACK_RULES = {
    "Pad": _pad_fallback,
    "GridSample": _gridsample_fallback,
    "Cast": _cast_fallback,
    "Einsum": _einsum_fallback,
}


def scan_graph(graph, supported, path, unsupported, fallbacks):
    """Recurse a graph (and subgraphs), bucketing nodes into unsupported / GPU-fallback."""
    index = build_value_index(graph)
    for i, node in enumerate(graph.node):
        # Only the default ONNX domain is implemented; custom domains are unsupported.
        in_default_domain = node.domain in ("", "ai.onnx")
        attrs = {a.name: attr_to_jsonable(a) for a in node.attribute}
        record = {
            "op_type": node.op_type,
            "domain": node.domain or "ai.onnx",
            "node_name": node.name or "(unnamed)",
            "graph": path,
            "index": i,
            "inputs": io_details(node.input, index),
            "outputs": io_details(node.output, index),
            "attributes": attrs,
        }
        if node.op_type not in supported or not in_default_domain:
            unsupported.append(record)
        else:
            rule = GPU_FALLBACK_RULES.get(node.op_type)
            reason = rule(attrs) if rule else None
            if reason:
                record["reason"] = reason
                fallbacks.append(record)
        for sub in subgraphs_of(node):
            scan_graph(sub, supported, "%s/%s[%s]" % (path, node.op_type, node.name or i),
                       unsupported, fallbacks)


def load_model(model_path):
    try:
        import onnx
        from onnx import shape_inference
    except ImportError:
        sys.exit("error: the 'onnx' package is required (pip install onnx)")
    try:
        model = onnx.load(model_path)
    except Exception as e:
        sys.exit("error: failed to load %s: %s" % (model_path, e))
    try:
        model = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except Exception as e:
        sys.stderr.write("warning: shape inference failed (%s); shapes may be missing\n" % e)
    return model


def opset_summary(model):
    return {imp.domain or "ai.onnx": imp.version for imp in model.opset_import}


def _render_instance(L, f):
    L.append("• %s  (node \"%s\", graph %s)" % (f["op_type"], f["node_name"], f["graph"]))
    if f["domain"] != "ai.onnx":
        L.append("    domain: %s" % f["domain"])
    if f.get("reason"):
        L.append("    reason: %s" % f["reason"])
    for label, ios in (("inputs", f["inputs"]), ("outputs", f["outputs"])):
        for io in ios:
            shape = io["shape"] if io["shape"] is not None else "?"
            L.append("    %-7s %-22s shape=%s dtype=%s (%s)"
                     % (label + ":", io["name"] or "<omitted>", shape,
                        io["dtype"] or "?", io["source"]))
    if f["attributes"]:
        for k, v in f["attributes"].items():
            L.append("    attr    %s = %s" % (k, v))
    L.append("")


def render_text(report):
    L = []
    m = report["model"]
    L.append("VKNN unsupported-operator scan")
    L.append("=" * 60)
    L.append("model:        %s" % m["path"])
    L.append("opset:        %s" % ", ".join("%s=%d" % (k, v) for k, v in m["opset"].items()))
    L.append("total nodes:  %d" % m["total_nodes"])
    L.append("supported op types known to VKNN: %d" % m["supported_count"])
    L.append("")

    summary = report["summary"]
    fb_summary = report["gpu_fallback_summary"]
    if not summary and not fb_summary:
        L.append("All operators are supported by VKNN with a GPU path. Nothing to do. ✔")
        return "\n".join(L)

    if not summary:
        L.append("No unsupported operators — every op runs. ✔")
    else:
        L.append("UNSUPPORTED OP TYPES (cannot run — must be implemented): "
                 "%d distinct, %d node(s)" % (len(summary), sum(s["count"] for s in summary)))
        L.append("-" * 60)
        for s in summary:
            dom = "" if s["domain"] == "ai.onnx" else " [domain=%s]" % s["domain"]
            L.append("  %-28s x%-4d%s" % (s["op_type"], s["count"], dom))
        L.append("")
        L.append("UNSUPPORTED — PER-INSTANCE DETAIL")
        L.append("-" * 60)
        for f in report["findings"]:
            _render_instance(L, f)

    if fb_summary:
        L.append("GPU FALLBACK (runs correctly on CPU; add a GPU kernel to optimize): "
                 "%d distinct, %d node(s)" % (len(fb_summary), sum(s["count"] for s in fb_summary)))
        L.append("-" * 60)
        for s in fb_summary:
            L.append("  %-28s x%-4d" % (s["op_type"], s["count"]))
        L.append("")
        L.append("GPU FALLBACK — PER-INSTANCE DETAIL")
        L.append("-" * 60)
        for f in report["gpu_fallbacks"]:
            _render_instance(L, f)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="path to the .onnx model")
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help="VKNN op source to derive the supported set from "
                         "(default: src/core/op.cpp)")
    ap.add_argument("--json", metavar="PATH",
                    help="also write the full machine-readable report to PATH")
    args = ap.parse_args()

    supported = supported_op_names(args.src)
    model = load_model(args.model)
    graph = model.graph

    findings = []
    fallbacks = []
    total = [0]

    def count_nodes(g):
        total[0] += len(g.node)
        for node in g.node:
            for sub in subgraphs_of(node):
                count_nodes(sub)

    count_nodes(graph)
    scan_graph(graph, supported, "main", findings, fallbacks)

    # Aggregate distinct (op_type, domain) with counts, sorted most-frequent first.
    def aggregate(items):
        agg = {}
        for f in items:
            key = (f["op_type"], f["domain"])
            agg[key] = agg.get(key, 0) + 1
        return [{"op_type": k[0], "domain": k[1], "count": c}
                for k, c in sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))]

    summary = aggregate(findings)
    fb_summary = aggregate(fallbacks)

    report = {
        "model": {
            "path": os.path.abspath(args.model),
            "opset": opset_summary(model),
            "total_nodes": total[0],
            "supported_count": len(supported),
        },
        "supported_op_types": sorted(supported),
        "summary": summary,
        "findings": findings,
        "gpu_fallback_summary": fb_summary,
        "gpu_fallbacks": fallbacks,
        "vknn_implementation_guide": {
            "note": "To add each op type below, follow the per-op recipe.",
            "recipe": "skills/add-an-operator.md, docs/ADDING_AN_OPERATOR.md",
            "files_to_touch": [
                "include/vknn/op.h            (add OpType::kFoo)",
                "src/core/op.cpp              (opTypeName + opTypeFromOnnx)",
                "src/import/passes.cpp        (shape rule in inferShapes)",
                "src/backend/cpu/ops/<op>.cpp (CpuOp + VKNN_REGISTER_CPU_OP; one op per file)",
                "shaders/<op>.comp + src/backend/vulkan/ops/<op>.cpp (optional GPU path)",
            ],
        },
    }

    print(render_text(report))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        sys.stderr.write("\nwrote machine-readable report to %s\n" % args.json)

    # Exit non-zero when something is unsupported, so this can gate CI / scripts.
    sys.exit(1 if summary else 0)


if __name__ == "__main__":
    main()
