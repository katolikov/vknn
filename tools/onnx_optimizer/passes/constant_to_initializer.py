"""Convert Constant nodes into initializers (pure structure, no math).

The tensor payload is moved verbatim (the `value` attribute) or materialized
with its ONNX-specified dtype (value_float[s] -> fp32, value_int[s] -> int64).
Sparse and string Constants are left alone, as is a Constant whose output is a
graph output or is captured by a subgraph.
"""
from onnx import numpy_helper

from onnx_optimizer import graph_util as gu
from onnx_optimizer.passes.base import Pass


class ConstantToInitializer(Pass):
    name = "constant-to-initializer"
    description = "materialize Constant nodes as initializers"

    def run(self, model):
        graph = model.graph
        protected = gu.subgraph_ref_names(graph)
        outputs = gu.graph_output_names(graph)
        inits = gu.initializer_names(graph)
        changes = 0
        for node in [n for n in graph.node if n.op_type == "Constant"]:
            out = node.output[0] if node.output else ""
            if not out or out in outputs or out in protected or out in inits:
                continue
            arr = gu.constant_node_array(node)
            if arr is None:
                continue
            graph.initializer.append(numpy_helper.from_array(arr, out))
            graph.node.remove(node)
            gu.remove_value_info(graph, out)
            changes += 1
        return changes
