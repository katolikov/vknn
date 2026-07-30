"""Merge consecutive Reshape nodes into the second one.

Reshape preserves the flat element order and only depends on the input's
element COUNT, which the first Reshape cannot change -- so the second Reshape
can read the first one's input directly. The only hazard is a 0 entry in the
second target shape with allowzero=0, which copies a dim from the (about to be
bypassed) intermediate tensor; in that case the target is replaced with the
statically-inferred output shape when fully known, else the pair is left
alone. The first Reshape stays for other consumers; DCE sweeps it when dead.
"""
import numpy as np
from onnx import numpy_helper

from onnx_optimizer import graph_util as gu
from onnx_optimizer.passes.base import Pass


class MergeReshapes(Pass):
    name = "merge-reshapes"
    description = "collapse Reshape-of-Reshape chains to a single Reshape"

    def run(self, model):
        graph = model.graph
        index = gu.tensor_index(graph)
        changes = 0
        for node in [n for n in graph.node if n.op_type == "Reshape"]:
            if len(node.input) < 2 or not node.input[0] or not node.input[1]:
                continue
            prod = gu.producer_map(graph)
            upstream = prod.get(node.input[0])
            if upstream is None or upstream.op_type != "Reshape":
                continue
            if not upstream.input or not upstream.input[0]:
                continue
            target = gu.const_array(graph, node.input[1])
            allowzero = gu.get_attr(node, "allowzero", 0)
            if target is None or (not allowzero and (target == 0).any()):
                out_shape = gu.static_shape(index, node.output[0])
                if out_shape is None:
                    continue
                shape_name = gu.unique_name(graph, node.output[0] + "_shape")
                graph.initializer.append(numpy_helper.from_array(
                    np.array(out_shape, dtype=np.int64), shape_name))
                node.input[1] = shape_name
            node.input[0] = upstream.input[0]
            changes += 1
        return changes
