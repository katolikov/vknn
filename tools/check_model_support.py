#!/usr/bin/env python3
"""Full VKNN support report for an ONNX model: ops, tensors, dtypes, shapes, verdict.

Answers "does VKNN run this model, and if not, why not" in one pass. The engine's
support surface is derived from the sources at run time (never a hardcoded list):

  op recognition   opTypeFromOnnx()/unaryFromOnnx()/binaryFromOnnx()/reduceFromOnnx()
                   in src/core/op.cpp — the importer's single source of truth.
  GPU kernels      VKNN_REGISTER_VK_OP registrations in src/backend/vulkan/ops/.
  CPU kernels      VKNN_REGISTER_CPU_OP registrations in src/backend/cpu/ops/.
  GPU gates        the attribute/shape-checkable subset of vk_backend.cpp::supportsNode
                   (Pad mode/rank, GridSample mode, Cast-from-int64, Einsum equation,
                   grouped Conv, Expand/Tile rank). Deeper gates depend on post-pass
                   internal state and are deliberately not replicated — a node they
                   would catch still RUNS (on the CPU op), it is never a blocker.

Verdict classes:
  SUPPORTED                every node imports and has a kernel; full-GPU expected.
  SUPPORTED, CPU FALLBACKS every node runs, but some on the CPU backend (listed, with
                           the reason) — correct results, GPU-kernel targets.
  NOT SUPPORTED            at least one blocker: an op the importer does not recognize
                           (incl. custom domains and control-flow subgraph ops), an op
                           with no kernel in either backend, or an unusable dtype.

Quantized models (QDQ / QLinear / dynamic quantization) get a dedicated blocker class:
the importer recognizes the family (incl. the com.microsoft members, matched by name),
and support arrives via the import-time dequantize pass — pending in this tree, so a
model carrying these ops still reports NOT SUPPORTED, with the precise reason.

Tensor checks: graph input/output dtypes against the engine's I/O surface
(fp32/fp16 native, uint8 via the declared-format boundary, integer tensors for
shape/index logic), initializer dtypes (DOUBLE narrows to fp32 at import),
symbolic/dynamic dims (shapes are fixed at compile; the batch dim resolves at
load), activation ranks beyond the flat-kernel bound, and external-data files.

Usage:
  tools/check_model_support.py model.onnx
  tools/check_model_support.py model.onnx --json report.json
  tools/check_model_support.py model.onnx --repo /path/to/vknn
  tools/check_model_support.py model.onnx --engine-report report.json

Engine-emitted mode (--engine-report): the node/backend analysis comes from the
engine itself instead of the regex-derived gates above. Generate the JSON with

  build-host/vknn_compile model.onnx /tmp/m.vxm --support-report report.json

which evaluates vkSupportSurvey — the exact gate code VulkanBackend::supportsNode
runs — over the post-pass graph, so tool and engine cannot drift. The report's
per-node rows become the fallback list verbatim (backend "none" rows become
blockers); the tensor/dtype checks still come from the ONNX file. Without the
flag, the regex derivation above is the no-binary fallback.

Exit code: 0 = supported (with or without CPU fallbacks), 1 = not supported,
2 = bad invocation / unreadable model. Requires: onnx (pip install onnx).

Sibling tool: scan_unsupported_ops.py emits an implementation brief (shapes +
attributes per missing op) for the unsupported bucket this tool reports. It
imports _INTERNAL_NAMES and CPU_FALLBACK_GATES from this file, so the gate
definitions here are the single source of truth between the two tools.
"""
import argparse
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Names accepted by op.cpp that never appear in an ONNX export (VKNN-internal
# fusion/layout pseudo-ops, family selectors, the placeholder). Canonical definition —
# scan_unsupported_ops.py imports this set rather than keeping its own copy.
_INTERNAL_NAMES = {"FusedSE", "FusedDwPw", "FusedPointwise", "ConvGemm", "ConvertLayout",
                   "ConvertDtype", "Unknown", "Unary", "Binary", "Reduce"}

# The ONNX quantized operator family (mirrors opTypeIsQuantized in src/core/op.cpp). The importer
# recognizes each as its own OpType, but no kernel exists: support arrives via the import-time
# dequantize pass, which rewrites them to float ops. QGemm/QLinearAdd/QLinearGlobalAveragePool are
# com.microsoft-domain ops the importer matches by name (the wire parser drops NodeProto.domain),
# so the custom-domain blocker does not apply to them.
_QUANTIZED_ONNX = {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear", "QLinearConv",
                   "QLinearMatMul", "QLinearAdd", "QLinearGlobalAveragePool", "MatMulInteger",
                   "ConvInteger", "QGemm"}

# ONNX dtype numbers (TensorProto.DataType) the engine handles, by role.
_NATIVE_IO = {"FLOAT", "FLOAT16", "UINT8"}       # graph I/O incl. the declared-format boundary
_SHAPE_LOGIC = {"INT64", "INT32", "BOOL", "INT8"}  # shape/index/mask tensors; folded or cast
_NARROWED = {"DOUBLE"}                           # import narrows to fp32
_UNUSABLE = {"STRING", "COMPLEX64", "COMPLEX128"}


def _function_body(text, name):
    """The brace-balanced body of `name(...)` in C++ source text, or ''."""
    m = re.search(r"\b%s\s*\([^)]*\)\s*\{" % re.escape(name), text)
    if not m:
        return ""
    depth, i = 1, m.end()
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[m.end():i]


def parse_op_map(op_cpp_path):
    """ONNX op name -> VKNN OpType name, from src/core/op.cpp."""
    try:
        with open(op_cpp_path) as f:
            text = f.read()
    except OSError as e:
        sys.exit("error: cannot read %s: %s (pass --repo)" % (op_cpp_path, e))
    name_to_type = {}
    body = _function_body(text, "OpType opTypeFromOnnx")
    for name, ty in re.findall(r'\{"([A-Za-z][A-Za-z0-9]*)",\s*OpType::(\w+)\}', body):
        name_to_type[name] = ty
    for fn, family in (("UnaryType unaryFromOnnx", "Unary"),
                       ("BinaryType binaryFromOnnx", "Binary")):
        for name in re.findall(r'\{"([A-Za-z][A-Za-z0-9]*)",\s*[UB]::\w+\}', _function_body(text, fn)):
            name_to_type.setdefault(name, family)
    for name in re.findall(r's\s*==\s*"(Reduce\w+)"', _function_body(text, "ReduceType reduceFromOnnx")):
        name_to_type.setdefault(name, "Reduce")
    if not name_to_type:
        sys.exit("error: no op mappings found in %s — wrong file?" % op_cpp_path)
    return name_to_type


def parse_registry(ops_dir, macro):
    """Set of OpType names registered with `macro` under `ops_dir`."""
    types = set()
    if not os.path.isdir(ops_dir):
        return types
    for fn in os.listdir(ops_dir):
        if not fn.endswith((".cpp", ".h")):
            continue
        with open(os.path.join(ops_dir, fn), errors="replace") as f:
            types.update(re.findall(r"%s\(OpType::(\w+)" % macro, f.read()))
    return types


def dtype_name(elem_type):
    try:
        from onnx import TensorProto
        return TensorProto.DataType.Name(elem_type)
    except Exception:
        return str(elem_type)


def dims_of(type_proto):
    """([dim...], dtype) with symbolic dims as strings and unknown dims as '?'."""
    if type_proto is None or not type_proto.HasField("tensor_type"):
        return None, None
    tt = type_proto.tensor_type
    if not tt.HasField("shape"):
        return None, dtype_name(tt.elem_type)
    dims = []
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        elif d.HasField("dim_param") and d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return dims, dtype_name(tt.elem_type)


def attr_map(node):
    from onnx import AttributeProto
    out = {}
    for a in node.attribute:
        t = a.type
        A = AttributeProto
        if t == A.INT:
            out[a.name] = a.i
        elif t == A.FLOAT:
            out[a.name] = a.f
        elif t == A.STRING:
            out[a.name] = a.s.decode("utf-8", "replace")
        elif t == A.INTS:
            out[a.name] = list(a.ints)
        elif t == A.FLOATS:
            out[a.name] = list(a.floats)
    return out


# --------------------------------------------------------------------------
# CPU-fallback gates: the reliable subset of vk_backend.cpp::supportsNode. Each
# returns a reason string when the node runs on the CPU backend instead of the GPU.
def _gate_pad(node, attrs, index):
    shape = (index.get(node.output[0]) or {}).get("shape")
    if shape is not None and len(shape) > 8:
        return "Pad on a rank-%d tensor (flat kernels decode rank <= 8)" % len(shape)
    mode = attrs.get("mode", "constant")
    if mode not in ("constant", "edge", "reflect"):
        return "Pad mode=%r (GPU kernel does constant/edge/reflect)" % mode
    return None


def _gate_gridsample(node, attrs, index):
    mode = attrs.get("mode", "bilinear")
    # Mirrors vkNodeGate: the GPU kernel bakes the mode as a spec constant and covers cubic too.
    if mode not in ("bilinear", "linear", "nearest", "cubic", "bicubic"):
        return "GridSample mode=%r (GPU kernel does bilinear/linear/nearest/cubic/bicubic)" % mode
    return None


def _gate_cast(node, attrs, index):
    src = (index.get(node.input[0]) or {}).get("dtype")
    if src == "INT64":
        return "Cast from int64 (the GPU has no int64 buffers; shape/index casts stay on the CPU op)"
    return None


def _gate_einsum(node, attrs, index):
    eq = "".join(c for c in attrs.get("equation", "") if c not in " \t")
    # 'i,j->ij' has its own GPU kernel; the two batched-matmul forms lower to MatMul chains at
    # compile (src/import/lower_einsum.cpp) and run on the GPU MatMul kernel.
    if eq in ("i,j->ij", "...ab,...b->...a", "bij,bnjk->bnik"):
        return None
    return ("Einsum equation=%r (GPU covers 'i,j->ij' plus the lowered '...ab,...b->...a' and "
            "'bij,bnjk->bnik'; other equations use the CPU op)" % eq)


def _gate_conv(node, attrs, index):
    group = attrs.get("group", 1)
    if group <= 1:
        return None
    xs = (index.get(node.input[0]) or {}).get("shape")
    ws = (index.get(node.input[1]) or {}).get("shape") if len(node.input) > 1 else None
    if not xs or not ws or len(xs) != 4 or len(ws) != 4 \
       or not isinstance(xs[1], int) or not isinstance(ws[0], int):
        return "Conv group=%d with unresolved shapes (grouped conv beyond pure depthwise runs on the CPU op)" % group
    cin, cout = xs[1], ws[0]
    if group == cin and cout == cin:
        return None  # pure depthwise: GPU kernel
    return ("Conv group=%d (Cin=%d, Cout=%d): only group==1 and pure depthwise have GPU kernels; "
            "partial groups and depthwise channel multipliers run on the group-aware CPU op"
            % (group, cin, cout))


def _gate_rank8(node, attrs, index):
    shape = (index.get(node.output[0]) or {}).get("shape")
    if shape is not None and len(shape) > 8:
        return "%s output rank %d (flat broadcast/tile kernels decode rank <= 8)" % (node.op_type, len(shape))
    return None


CPU_FALLBACK_GATES = {
    "Pad": _gate_pad,
    "GridSample": _gate_gridsample,
    "Cast": _gate_cast,
    "Einsum": _gate_einsum,
    "Conv": _gate_conv,
    "Expand": _gate_rank8,
    "Tile": _gate_rank8,
}


def build_value_index(graph, parent_index=None):
    index = dict(parent_index) if parent_index else {}
    for coll in (graph.input, graph.value_info, graph.output):
        for vi in coll:
            shape, dt = dims_of(vi.type)
            index[vi.name] = {"shape": shape, "dtype": dt}
    for init in graph.initializer:
        index[init.name] = {"shape": [int(d) for d in init.dims],
                            "dtype": dtype_name(init.data_type)}
    return index


def subgraphs_of(node):
    from onnx import AttributeProto
    for a in node.attribute:
        if a.type == AttributeProto.GRAPH:
            yield a.g
        elif a.type == AttributeProto.GRAPHS:
            for g in a.graphs:
                yield g


class Report:
    def __init__(self):
        self.blockers = []     # {kind, detail, nodes: [...]} — model cannot run
        self.fallbacks = []    # {op_type, node, reason} — runs on the CPU backend
        self.may_fold = []     # gate hits on shape-logic chains that const-fold at compile
        self.tensor_notes = [] # informational tensor findings
        self.warnings = []     # non-blocking observations
        self.stats = {}

    def blocker(self, kind, detail, node=None):
        for b in self.blockers:
            if b["kind"] == kind and b["detail"] == detail:
                if node:
                    b["nodes"].append(node)
                return
        self.blockers.append({"kind": kind, "detail": detail, "nodes": [node] if node else []})


def _scalar_const(graph, name):
    """First element of initializer or Constant-node output `name` as a float, or None."""
    from onnx import numpy_helper
    try:
        for init in graph.initializer:
            if init.name == name:
                arr = numpy_helper.to_array(init)
                return float(arr.reshape(-1)[0]) if arr.size else None
        for n in graph.node:
            if n.op_type == "Constant" and name in n.output:
                for a in n.attribute:
                    if a.name == "value":
                        arr = numpy_helper.to_array(a.t)
                        return float(arr.reshape(-1)[0]) if arr.size else None
    except Exception:
        return None
    return None


def _constant_names(graph):
    """Names holding compile-time constants: initializers plus Constant-node outputs."""
    names = {i.name for i in graph.initializer}
    for n in graph.node:
        if n.op_type == "Constant":
            names.update(o for o in n.output if o)
    return names


def _instancenorm_reason(node, graph, index):
    """Blocker reason for an InstanceNormalization the importer cannot lower, or None.

    Mirrors src/import/lower_instancenorm.cpp: a node whose scale/B are compile-time constants
    and whose input is rank >= 3 lowers to spatial ReduceMean + Sub/Mul/Add/Sqrt/Div at import;
    anything else keeps the opaque op, which has no kernel in either backend.
    """
    consts = _constant_names(graph)
    for role, name in (("scale", node.input[1] if len(node.input) > 1 else ""),
                       ("B", node.input[2] if len(node.input) > 2 else "")):
        if not name or name not in consts:
            return ("op InstanceNormalization — %s is not a compile-time constant "
                    "(import lowers only the constant-parameter form)" % role)
    shape = (index.get(node.input[0]) or {}).get("shape")
    if shape is not None and len(shape) < 3:
        return ("op InstanceNormalization — input rank %d (import lowers only rank >= 3, "
                "[N,C,spatial...])" % len(shape))
    return None


def _dropout_reason(node, graph, consumed):
    """Blocker reason for a Dropout the importer cannot erase, or None when it erases.

    Mirrors src/import/eliminate_dropout.cpp: inference-mode Dropout (training_mode input absent
    or a constant false, mask output absent or unconsumed) is removed at import with consumers
    rewired to the producer; anything else keeps the node, which has no kernel in either backend.
    """
    if len(node.output) > 1 and node.output[1] and node.output[1] in consumed:
        return ("op Dropout — mask output is consumed (import erases only the identity form; "
                "the mask is never fabricated)")
    if len(node.input) > 2 and node.input[2]:
        val = _scalar_const(graph, node.input[2])
        if val is None:
            return ("op Dropout — training_mode is not a constant "
                    "(import erases only the provably inference-mode form)")
        if val != 0:
            return "op Dropout — training_mode is constant true (training-mode Dropout has no kernel)"
    return None


def scan_nodes(graph, path, name_to_type, vk_ops, cpu_ops, rep, parent_index=None):
    index = build_value_index(graph, parent_index)
    consumed = {i for n in graph.node for i in n.input if i} | {o.name for o in graph.output}
    for i, node in enumerate(graph.node):
        label = "%s '%s' (%s#%d)" % (node.op_type, node.name or "unnamed", path, i)
        attrs = attr_map(node)
        if node.domain not in ("", "ai.onnx") and node.op_type not in _QUANTIZED_ONNX:
            rep.blocker("custom domain",
                        "op %s from domain %r — only the default ONNX domain is implemented"
                        % (node.op_type, node.domain), label)
            continue
        if node.op_type in _QUANTIZED_ONNX:
            rep.blocker("quantized op",
                        "op %s — quantized operator: runs via the import-time dequantize pass "
                        "(pending in this tree); no kernel executes it directly" % node.op_type,
                        label)
            continue
        ty = name_to_type.get(node.op_type)
        if ty is None:
            extra = " (control-flow subgraph op)" if list(subgraphs_of(node)) else ""
            rep.blocker("unrecognized op",
                        "op %s — not recognized by the importer%s" % (node.op_type, extra), label)
        elif node.op_type in ("Constant", "Identity", "Shape"):
            # Erased at import: Constant materializes as an initializer, Identity is eliminated,
            # and Shape of a compile-time-static tensor const-folds (VKNN compiles fixed shapes).
            pass
        elif node.op_type == "Dropout":
            # Erased at import when inference-mode (see _dropout_reason); otherwise kernel-less.
            reason = _dropout_reason(node, graph, consumed)
            if reason:
                rep.blocker("no kernel", reason, label)
        elif node.op_type == "InstanceNormalization":
            # Lowered at import to per-channel normalize ops (see _instancenorm_reason);
            # otherwise kernel-less.
            reason = _instancenorm_reason(node, graph, index)
            if reason:
                rep.blocker("no kernel", reason, label)
        else:
            has_vk = ty in vk_ops
            has_cpu = ty in cpu_ops
            if not has_vk and not has_cpu:
                rep.blocker("no kernel",
                            "op %s (OpType::%s) — recognized but no kernel in either backend"
                            % (node.op_type, ty), label)
            elif not has_vk and has_cpu:
                rep.fallbacks.append({"op_type": node.op_type, "node": label,
                                      "reason": "OpType::%s has no Vulkan kernel (CPU-only op; "
                                                "const-inputs versions fold away at compile)" % ty})
            else:
                gate = CPU_FALLBACK_GATES.get(node.op_type)
                reason = gate(node, attrs, index) if gate else None
                if reason:
                    # int64 sources are shape/index arithmetic; under VKNN's static shapes those
                    # chains const-fold at compile, so the gate hit rarely survives to run time.
                    bucket = rep.may_fold if "int64" in reason else rep.fallbacks
                    bucket.append({"op_type": node.op_type, "node": label, "reason": reason})
        for sub in subgraphs_of(node):
            scan_nodes(sub, "%s/%s" % (path, node.name or node.op_type),
                       name_to_type, vk_ops, cpu_ops, rep, index)


def scan_tensors(model, model_path, rep):
    graph = model.graph
    init_names = {i.name for i in graph.initializer}

    # Graph I/O dtypes and dynamic dims.
    for vi in list(graph.input):
        if vi.name in init_names:
            continue
        shape, dt = dims_of(vi.type)
        rep.stats.setdefault("inputs", []).append({"name": vi.name, "shape": shape, "dtype": dt})
        if dt in _UNUSABLE:
            rep.blocker("dtype", "input '%s' is %s — no such tensors in the engine" % (vi.name, dt))
        elif dt in _SHAPE_LOGIC:
            rep.tensor_notes.append("input '%s' is %s: usable as a shape/index/mask tensor "
                                    "(int64 consumers stay on the CPU op)" % (vi.name, dt))
        elif dt not in _NATIVE_IO and dt not in _NARROWED and dt is not None:
            rep.blocker("dtype", "input '%s' is %s — outside the engine's I/O surface" % (vi.name, dt))
        if shape:
            sym = [d for d in shape if not isinstance(d, int)]
            if sym:
                if not isinstance(shape[0], int) and len(sym) == 1:
                    rep.tensor_notes.append("input '%s' has a symbolic batch dim %r — resolved at "
                                            "load (vknn_compile bakes batch=1 by default)"
                                            % (vi.name, sym[0]))
                else:
                    rep.warnings.append("input '%s' has symbolic dims %s beyond the batch dim — "
                                        "VKNN compiles fixed shapes; re-export with them pinned"
                                        % (vi.name, sym))
    for vi in graph.output:
        shape, dt = dims_of(vi.type)
        rep.stats.setdefault("outputs", []).append({"name": vi.name, "shape": shape, "dtype": dt})
        if dt in _UNUSABLE:
            rep.blocker("dtype", "output '%s' is %s — no such tensors in the engine" % (vi.name, dt))

    # Initializers: dtype spread, total bytes, external data files.
    by_dtype, total_bytes, ext_files = {}, 0, set()
    import numpy as _np  # onnx depends on numpy; used only for element sizes
    for init in graph.initializer:
        dt = dtype_name(init.data_type)
        by_dtype[dt] = by_dtype.get(dt, 0) + 1
        n = 1
        for d in init.dims:
            n *= int(d)
        elem = {"FLOAT": 4, "DOUBLE": 8, "FLOAT16": 2, "BFLOAT16": 2, "INT64": 8,
                "INT32": 4, "INT16": 2, "INT8": 1, "UINT8": 1, "BOOL": 1}.get(dt, 4)
        total_bytes += n * elem
        if init.data_location == 1:  # EXTERNAL
            for kv in init.external_data:
                if kv.key == "location":
                    ext_files.add(kv.value)
        if dt in _UNUSABLE:
            rep.blocker("dtype", "initializer '%s' is %s" % (init.name, dt))
    rep.stats["initializers"] = {"count": len(graph.initializer), "bytes": total_bytes,
                                 "by_dtype": by_dtype}
    if "DOUBLE" in by_dtype:
        rep.tensor_notes.append("%d DOUBLE initializer(s) narrow to fp32 at import" % by_dtype["DOUBLE"])
    if by_dtype.get("INT8") or by_dtype.get("UINT8"):
        rep.warnings.append("int8/uint8 initializers present — if these are quantized weights, the "
                            "quantized ops they feed decide support (see the op report)")
    model_dir = os.path.dirname(os.path.abspath(model_path))
    for f in sorted(ext_files):
        p = os.path.join(model_dir, f)
        if os.path.exists(p):
            rep.tensor_notes.append("external weights: %s (%.1f MB, present)"
                                    % (f, os.path.getsize(p) / 1e6))
        else:
            rep.blocker("external data", "external weights file '%s' not found next to the model" % f)

    # Activation ranks beyond the flat-kernel bound (informational: such nodes take
    # CPU fallbacks or fail the layout pass; rank > 6 is already exotic for VKNN's targets).
    deep = set()
    for vi in graph.value_info:
        shape, _ = dims_of(vi.type)
        if shape is not None and len(shape) > 6:
            deep.add("%s rank %d" % (vi.name, len(shape)))
    if deep:
        rep.warnings.append("rank > 6 intermediate tensor(s): %s — flat kernels decode up to "
                            "rank 6-8 per op; expect CPU fallbacks there" % ", ".join(sorted(deep)[:5]))


def load_model(path):
    try:
        import onnx
        from onnx import shape_inference
    except ImportError:
        sys.exit(2)
    try:
        # Weight payloads are never read (all checks use dims/data_type), so external data stays
        # on disk: multi-GB models scan in seconds and the report checks the files' presence itself.
        model = onnx.load(path, load_external_data=False)
    except Exception as e:
        sys.stderr.write("error: failed to load %s: %s\n" % (path, e))
        sys.exit(2)
    try:
        model = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except Exception as e:
        sys.stderr.write("warning: shape inference failed (%s); shape-dependent checks degrade\n" % e)
    return model


def fmt_shape(shape):
    return "[" + ",".join(str(d) for d in shape) + "]" if shape is not None else "[?]"


def main():
    ap = argparse.ArgumentParser(description="VKNN support report for an ONNX model")
    ap.add_argument("model")
    ap.add_argument("--repo", default=REPO_ROOT, help="VKNN repo root (default: this tree)")
    ap.add_argument("--json", help="also write the full report as JSON")
    ap.add_argument("--engine-report", metavar="JSON",
                    help="per-node assignment from `vknn_compile --support-report`; replaces the "
                         "regex-derived node/gate analysis with the engine's own answer")
    args = ap.parse_args()

    model = load_model(args.model)
    rep = Report()
    rep.stats["opset"] = {imp.domain or "ai.onnx": imp.version for imp in model.opset_import}
    rep.stats["node_count"] = len(model.graph.node)
    scan_tensors(model, args.model, rep)

    if args.engine_report:
        # Engine truth: the report rows are the post-pass node assignment computed by the same
        # gate code supportsNode runs; no regex derivation is involved.
        try:
            with open(args.engine_report) as f:
                engine = json.load(f)
        except (OSError, ValueError) as e:
            sys.exit("error: cannot read engine report %s: %s" % (args.engine_report, e))
        for n in engine.get("nodes", []):
            label = "%s '%s' (engine)" % (n.get("op", "?"), n.get("name") or "unnamed")
            if n.get("backend") == "none":
                rep.blocker("no kernel", "op %s — %s" % (n.get("op", "?"),
                                                         n.get("reason", "no kernel")), label)
            elif n.get("backend") != "vulkan":
                rep.fallbacks.append({"op_type": n.get("op", "?"), "node": label,
                                      "reason": n.get("reason", "")})
        rep.stats["node_source"] = "engine-report:%s" % os.path.abspath(args.engine_report)
        rep.stats["engine_node_count"] = (engine.get("summary") or {}).get("total")
    else:
        name_to_type = parse_op_map(os.path.join(args.repo, "src", "core", "op.cpp"))
        vk_ops = parse_registry(os.path.join(args.repo, "src", "backend", "vulkan", "ops"),
                                "VKNN_REGISTER_VK_OP")
        cpu_ops = parse_registry(os.path.join(args.repo, "src", "backend", "cpu", "ops"),
                                 "VKNN_REGISTER_CPU_OP")
        if not vk_ops or not cpu_ops:
            sys.stderr.write("warning: backend registries not found under %s; kernel checks degrade\n"
                             % args.repo)
        scan_nodes(model.graph, "main", name_to_type, vk_ops, cpu_ops, rep)

    supported = not rep.blockers
    print("== VKNN support report: %s ==" % os.path.basename(args.model))
    if args.engine_report:
        print("(node assignment: engine-emitted, %s)" % args.engine_report)
    ini = rep.stats.get("initializers", {})
    print("opset %s | %d node(s) | %d initializer(s), %.1f MB weights"
          % (rep.stats["opset"], rep.stats["node_count"], ini.get("count", 0),
             ini.get("bytes", 0) / 1e6))
    for io, key in (("input", "inputs"), ("output", "outputs")):
        for t in rep.stats.get(key, []):
            print("  %s %s %s %s" % (io, t["name"], fmt_shape(t["shape"]), t["dtype"]))
    print()

    if supported:
        if rep.fallbacks:
            print("VERDICT: SUPPORTED — runs end to end; %d node(s) fall back to the CPU backend."
                  % len(rep.fallbacks))
        else:
            print("VERDICT: SUPPORTED — every node imports and has a GPU path.")
    else:
        print("VERDICT: NOT SUPPORTED — %d blocking issue(s):" % len(rep.blockers))
        for b in rep.blockers:
            n = len(b["nodes"])
            print("  [%s] %s%s" % (b["kind"], b["detail"],
                                   " — %d node(s), e.g. %s" % (n, b["nodes"][0]) if n else ""))

    if rep.fallbacks:
        print("\n-- CPU fallbacks (correct results; GPU-kernel targets) --")
        print("  (pre-compile upper bound: const-input chains fold away; the engine prints the")
        print("   real fallback count at session load)")
        by_reason = {}
        for f in rep.fallbacks:
            by_reason.setdefault(f["reason"], []).append(f["node"])
        for reason, nodes in sorted(by_reason.items()):
            print("  %d node(s): %s\n      e.g. %s" % (len(nodes), reason, nodes[0]))
    if rep.may_fold:
        print("\n-- shape-logic gate hits (const-fold at compile when shapes are static) --")
        by_reason = {}
        for f in rep.may_fold:
            by_reason.setdefault(f["reason"], []).append(f["node"])
        for reason, nodes in sorted(by_reason.items()):
            print("  %d node(s): %s" % (len(nodes), reason))

    if rep.tensor_notes:
        print("\n-- tensors --")
        for t in rep.tensor_notes:
            print("  " + t)
    if rep.warnings:
        print("\n-- warnings --")
        for w in rep.warnings:
            print("  " + w)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"model": os.path.abspath(args.model), "supported": supported,
                       "stats": rep.stats, "blockers": rep.blockers, "cpu_fallbacks": rep.fallbacks,
                       "shape_logic_gate_hits": rep.may_fold,
                       "tensor_notes": rep.tensor_notes, "warnings": rep.warnings}, f, indent=2)
        print("\nwrote %s" % args.json)

    sys.exit(0 if supported else 1)


if __name__ == "__main__":
    main()
