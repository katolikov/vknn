"""--target vknn: report what the vknn importer will and won't accept.

The supported-op surface is derived at run time from src/core/op.cpp via
tools/check_model_support.py (parse_op_map / _QUANTIZED_ONNX), the same single
source of truth the repo's other support tools share -- nothing is hardcoded
here, so the report cannot drift from the engine.

This mode only REPORTS; it never changes what the optimizer does. The default
passes already produce the shapes vknn prefers: fewer Reshape/Transpose nodes
(NC4HW4 layout churn), pre-folded int64 shape arithmetic (no Cast-from-int64
CPU fallbacks), inference-mode Dropout gone before the importer sees it.
"""
import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_support():
    """(name->OpType map, quantized-op set) from check_model_support.py, or None."""
    if _TOOLS_DIR not in sys.path:
        sys.path.insert(0, _TOOLS_DIR)
    try:
        import check_model_support as cms
    except ImportError:
        return None
    op_cpp = os.path.join(cms.REPO_ROOT, "src", "core", "op.cpp")
    if not os.path.isfile(op_cpp):
        return None
    try:
        return cms.parse_op_map(op_cpp), cms._QUANTIZED_ONNX
    except SystemExit:
        return None


def _subgraphs_of(node):
    from onnx import AttributeProto
    for a in node.attribute:
        if a.type == AttributeProto.GRAPH:
            yield a.g
        elif a.type == AttributeProto.GRAPHS:
            for g in a.graphs:
                yield g


def analyze(model):
    """vknn-importer compatibility summary for the (optimized) model."""
    support = _load_support()
    result = {"available": support is not None, "unsupported": {}, "quantized": {},
              "custom_domain": {}, "control_flow": {}, "dynamic_dims": [], "text": []}
    lines = result["text"]
    lines.append("-- vknn target report --")
    if support is None:
        lines.append("  (op.cpp / check_model_support.py not found; op checks skipped)")
        return result
    name_to_type, quantized = support

    def scan(graph):
        for node in graph.node:
            buckets = []
            if node.op_type in quantized:
                buckets.append(result["quantized"])
            if node.domain not in ("", "ai.onnx") and node.op_type not in quantized:
                buckets.append(result["custom_domain"])
            elif node.op_type not in name_to_type and node.op_type not in quantized:
                buckets.append(result["control_flow"] if any(True for _ in _subgraphs_of(node))
                               else result["unsupported"])
            for bucket in buckets:
                bucket[node.op_type] = bucket.get(node.op_type, 0) + 1
            for sub in _subgraphs_of(node):
                scan(sub)

    scan(model.graph)
    init_names = {t.name for t in model.graph.initializer}
    for vi in model.graph.input:
        if vi.name in init_names or not vi.type.HasField("tensor_type"):
            continue
        dims = vi.type.tensor_type.shape.dim
        sym = [i for i, d in enumerate(dims) if not d.HasField("dim_value")]
        if sym and (len(sym) > 1 or sym[0] != 0):
            result["dynamic_dims"].append(vi.name)

    clean = True
    for label, bucket in (("importer does NOT recognize", result["unsupported"]),
                          ("control-flow (subgraph) op", result["control_flow"]),
                          ("custom-domain op", result["custom_domain"])):
        for op, count in sorted(bucket.items()):
            lines.append("  WARNING: %s: %-24s x%d" % (label, op, count))
            clean = False
    for op, count in sorted(result["quantized"].items()):
        lines.append("  note: quantized op %s x%d -- runs via vknn's import-time "
                     "dequantize pass" % (op, count))
    for name in result["dynamic_dims"]:
        lines.append("  WARNING: input '%s' has symbolic dims beyond the batch dim -- "
                     "vknn compiles fixed shapes; re-export with them pinned" % name)
        clean = False
    if clean and not result["quantized"]:
        lines.append("  every op is recognized by the vknn importer")
    return result
