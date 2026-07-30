"""Remove shape-preserving Reshape / Flatten / Squeeze / Unsqueeze / Expand.

These five ops all preserve the flat element order, so when the statically
inferred output shape equals the statically inferred input shape the node is
an exact identity. Requires both shapes fully static (shape inference runs
before every pass sweep).
"""
from onnx_optimizer import graph_util as gu
from onnx_optimizer.passes.base import Pass

_ORDER_PRESERVING = ("Reshape", "Flatten", "Squeeze", "Unsqueeze", "Expand")


class RemoveNoopReshape(Pass):
    name = "remove-noop-reshape"
    description = "remove Reshape/Flatten/Squeeze/Unsqueeze/Expand with output shape == input shape"

    def run(self, model):
        graph = model.graph
        index = gu.tensor_index(graph)
        protected = gu.subgraph_ref_names(graph)
        changes = 0
        for node in [n for n in graph.node if n.op_type in _ORDER_PRESERVING]:
            if not node.input or not node.input[0]:
                continue
            in_shape = gu.static_shape(index, node.input[0])
            out_shape = gu.static_shape(index, node.output[0])
            if in_shape is None or out_shape is None or in_shape != out_shape:
                continue
            if gu.try_bypass(graph, node, protected):
                changes += 1
        return changes
