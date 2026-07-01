#!/usr/bin/env python3
"""Dump EVERYTHING from an ONNX model: metadata, opset imports, graph inputs/outputs, initializers
(weights), every tensor with its dtype + (shape-inferred) shape, and every node with its op type,
name, inputs, outputs, and all attributes. Also an op-type histogram, a dtype summary, and totals.
Recurses into subgraphs (If / Loop / Scan).

Usage:
  python3 onnx_dump.py MODEL.onnx [options]
    --values          print contents of small initializers (<= 16 elements)
    --attrs-full      do not truncate long attribute int/float lists
    --no-infer        skip onnx shape inference (faster; intermediates then have no shape)
    --json OUT.json   also write the whole dump as JSON

Requires: onnx (pip install onnx). Read-only — never modifies the model. Handles models > 2 GB
(external data / file-based shape inference) and missing external weights.
"""
import argparse
import json
import os
import sys
import tempfile
from collections import Counter, OrderedDict

import onnx
from onnx import AttributeProto, TensorProto, numpy_helper, shape_inference


def dname(dt):
    try:
        return TensorProto.DataType.Name(dt)
    except Exception:
        return "dtype(%s)" % dt


def vi_dtype_shape(vi):
    """(dtype_name, shape_list_or_None) for a ValueInfoProto. Dims may be int, a symbolic
    name (dynamic axis), or '?' (unknown)."""
    tt = vi.type.tensor_type
    dt = dname(tt.elem_type)
    if not tt.HasField("shape"):
        return dt, None
    dims = []
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        elif d.HasField("dim_param"):
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return dt, dims


def shape_str(shape):
    return "?" if shape is None else "[" + ",".join(str(x) for x in shape) + "]"


def attr_value(a, truncate=True):
    """AttributeProto -> a readable Python value (lists truncated unless truncate=False)."""
    t = a.type
    if t == AttributeProto.INT:
        return int(a.i)
    if t == AttributeProto.FLOAT:
        return float(a.f)
    if t == AttributeProto.STRING:
        return a.s.decode("utf-8", "replace")
    if t == AttributeProto.TENSOR:
        return "<tensor %s %s>" % (dname(a.t.data_type), list(a.t.dims))
    if t == AttributeProto.TENSORS:
        return "<%d tensors>" % len(a.tensors)
    if t == AttributeProto.GRAPH:
        return "<subgraph '%s': %d nodes>" % (a.g.name, len(a.g.node))
    if t == AttributeProto.GRAPHS:
        return "<%d subgraphs>" % len(a.graphs)
    if t == AttributeProto.INTS:
        v = [int(x) for x in a.ints]
    elif t == AttributeProto.FLOATS:
        v = [float(x) for x in a.floats]
    elif t == AttributeProto.STRINGS:
        v = [s.decode("utf-8", "replace") for s in a.strings]
    else:
        return "<attr type %s>" % t
    if truncate and len(v) > 12:
        return v[:12] + ["...(+%d)" % (len(v) - 12)]
    return v


def infer(path, model, do_infer):
    """Return a model with intermediate value_info filled in by shape inference. Falls back to
    the file-based path for >2 GB models; returns the original model if inference fails."""
    if not do_infer:
        return model
    try:
        return shape_inference.infer_shapes(model)
    except Exception:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False).name
            shape_inference.infer_shapes_path(path, tmp)
            m2 = onnx.load(tmp, load_external_data=False)
            os.unlink(tmp)
            return m2
        except Exception as e:
            print("# shape inference failed (%s); intermediate shapes unavailable" % e, file=sys.stderr)
            return model


def collect_types(graph):
    """name -> (category, dtype, shape) for every tensor known from inputs/outputs/value_info/initializers."""
    m = OrderedDict()
    for vi in graph.input:
        dt, sh = vi_dtype_shape(vi)
        m[vi.name] = ["input", dt, sh]
    for init in graph.initializer:
        m[init.name] = ["initializer", dname(init.data_type), list(init.dims)]
    for vi in graph.value_info:
        dt, sh = vi_dtype_shape(vi)
        m.setdefault(vi.name, ["value", dt, sh])
    for vi in graph.output:
        dt, sh = vi_dtype_shape(vi)
        m[vi.name] = ["output", dt, sh]
    return m


def num_elems(dims):
    n = 1
    for d in dims:
        if not isinstance(d, int):
            return None
        n *= d
    return n


def dump_nodes(nodes, out, args, indent, op_hist):
    pad = "  " * indent
    for i, nd in enumerate(nodes):
        op_hist[nd.op_type] += 1
        ins = ", ".join(x if x else "∅" for x in nd.input) or "-"
        outs = ", ".join(x if x else "∅" for x in nd.output) or "-"
        name = nd.name or "(unnamed)"
        dom = ("" if nd.domain in ("", "ai.onnx") else "[%s]" % nd.domain)
        out.append("%s#%d  %s%s  %s" % (pad, i, nd.op_type, dom, name))
        out.append("%s      in : %s" % (pad, ins))
        out.append("%s      out: %s" % (pad, outs))
        for a in nd.attribute:
            out.append("%s      @%s = %s" % (pad, a.name, attr_value(a, truncate=not args.attrs_full)))
        # recurse into subgraph-valued attributes (If / Loop / Scan bodies)
        for a in nd.attribute:
            if a.type == AttributeProto.GRAPH:
                out.append("%s      -- subgraph @%s '%s' --" % (pad, a.name, a.g.name))
                dump_nodes(a.g.node, out, args, indent + 2, op_hist)
            elif a.type == AttributeProto.GRAPHS:
                for g in a.graphs:
                    out.append("%s      -- subgraph @%s '%s' --" % (pad, a.name, g.name))
                    dump_nodes(g.node, out, args, indent + 2, op_hist)


def main():
    ap = argparse.ArgumentParser(description="Dump everything from an ONNX model.")
    ap.add_argument("model")
    ap.add_argument("--values", action="store_true", help="print small initializer contents")
    ap.add_argument("--attrs-full", action="store_true", help="do not truncate long attribute lists")
    ap.add_argument("--no-infer", action="store_true", help="skip shape inference")
    ap.add_argument("--json", metavar="OUT.json", help="also write the dump as JSON")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit("error: no such file: %s" % args.model)

    # Load structure without pulling multi-GB external weights (values not needed for the dump).
    model = onnx.load(args.model, load_external_data=args.values)
    model = infer(args.model, model, not args.no_infer)
    g = model.graph
    types = collect_types(g)

    # Every tensor name that appears anywhere (edges included), so nothing is missed.
    order = list(types.keys())
    for nd in g.node:
        for nm in list(nd.input) + list(nd.output):
            if nm and nm not in types:
                types[nm] = ["intermediate", "?", None]
                order.append(nm)

    out = []
    out.append("=" * 78)
    out.append("MODEL  %s" % os.path.abspath(args.model))
    out.append("=" * 78)
    out.append("ir_version    : %s" % model.ir_version)
    out.append("producer      : %s %s" % (model.producer_name or "-", model.producer_version or ""))
    out.append("model_version : %s" % model.model_version)
    out.append("opset_import  : " + ", ".join("%s=%d" % (op.domain or "ai.onnx", op.version) for op in model.opset_import))
    if model.doc_string:
        out.append("doc           : %s" % model.doc_string.strip().splitlines()[0][:100])
    if model.metadata_props:
        for p in model.metadata_props:
            out.append("meta[%s]      : %s" % (p.key, p.value[:80]))

    def section(title, rows):
        out.append("")
        out.append("== %s (%d) ==" % (title, len(rows)))
        out.extend(rows)

    section("INPUTS", ["  %-40s %-10s %s" % (vi.name, vi_dtype_shape(vi)[0], shape_str(vi_dtype_shape(vi)[1])) for vi in g.input])
    section("OUTPUTS", ["  %-40s %-10s %s" % (vi.name, vi_dtype_shape(vi)[0], shape_str(vi_dtype_shape(vi)[1])) for vi in g.output])

    total_params = 0
    init_rows = []
    for init in g.initializer:
        dims = list(init.dims)
        n = num_elems(dims)
        if n:
            total_params += n
        row = "  %-40s %-10s %-16s %s elems" % (init.name, dname(init.data_type), shape_str(dims), n if n is not None else "?")
        if args.values and n is not None and n <= 16:
            try:
                row += "  = " + str(numpy_helper.to_array(init).ravel().tolist())
            except Exception:
                pass
        init_rows.append(row)
    section("INITIALIZERS  (weights/constants)", init_rows)
    out.append("  total parameters: %s" % format(total_params, ","))

    # Every tensor, with dtype + shape + where it comes from.
    tensor_rows = []
    for nm in order:
        cat, dt, sh = types[nm]
        tensor_rows.append("  [%-11s] %-44s %-10s %s" % (cat, nm, dt, shape_str(sh)))
    section("TENSORS  (all values, incl. intermediates)", tensor_rows)

    op_hist = Counter()
    node_rows = []
    dump_nodes(g.node, node_rows, args, 0, op_hist)
    out.append("")
    out.append("== NODES (%d) ==" % sum(1 for _ in _walk_nodes(g)))
    out.extend(node_rows)

    section("OP HISTOGRAM", ["  %-24s %d" % (k, v) for k, v in sorted(op_hist.items(), key=lambda kv: (-kv[1], kv[0]))])

    dt_hist = Counter(v[1] for v in types.values())
    section("DTYPE SUMMARY  (over all tensors)", ["  %-12s %d" % (k, v) for k, v in sorted(dt_hist.items(), key=lambda kv: (-kv[1], kv[0]))])

    out.append("")
    out.append("== TOTALS ==")
    out.append("  nodes=%d  inputs=%d  outputs=%d  initializers=%d  tensors=%d  op_types=%d  params=%s"
               % (sum(1 for _ in _walk_nodes(g)), len(g.input), len(g.output), len(g.initializer),
                  len(types), len(op_hist), format(total_params, ",")))

    print("\n".join(out))

    if args.json:
        doc = {
            "model": {"ir_version": model.ir_version, "producer": model.producer_name,
                      "opset_import": {(op.domain or "ai.onnx"): op.version for op in model.opset_import}},
            "inputs": [{"name": vi.name, "dtype": vi_dtype_shape(vi)[0], "shape": vi_dtype_shape(vi)[1]} for vi in g.input],
            "outputs": [{"name": vi.name, "dtype": vi_dtype_shape(vi)[0], "shape": vi_dtype_shape(vi)[1]} for vi in g.output],
            "initializers": [{"name": i.name, "dtype": dname(i.data_type), "shape": list(i.dims)} for i in g.initializer],
            "tensors": [{"name": nm, "category": types[nm][0], "dtype": types[nm][1], "shape": types[nm][2]} for nm in order],
            "nodes": [{"index": k, "op_type": nd.op_type, "name": nd.name, "domain": nd.domain,
                       "inputs": list(nd.input), "outputs": list(nd.output),
                       "attributes": {a.name: attr_value(a, truncate=False) for a in nd.attribute}}
                      for k, nd in enumerate(g.node)],
            "op_histogram": dict(op_hist),
            "dtype_summary": dict(dt_hist),
        }
        with open(args.json, "w") as f:
            json.dump(doc, f, indent=2, default=str)
        print("\n# wrote JSON -> %s" % args.json, file=sys.stderr)


def _walk_nodes(graph):
    for nd in graph.node:
        yield nd
        for a in nd.attribute:
            if a.type == AttributeProto.GRAPH:
                yield from _walk_nodes(a.g)
            elif a.type == AttributeProto.GRAPHS:
                for g in a.graphs:
                    yield from _walk_nodes(g)


if __name__ == "__main__":
    main()
