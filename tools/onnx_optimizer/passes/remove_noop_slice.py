"""Remove Slice nodes that keep the whole tensor.

Shape equality alone is NOT sufficient (a step of -1 reverses data at equal
shape), so the proof is: every step is 1 -- steps input absent (defaults to
ones), or a compile-time constant of all ones, or the attribute form (opset
< 10) which has no steps at all -- AND the static output shape equals the
static input shape (with unit steps, equal extent per axis forces start=0 and
end>=dim after clamping).
"""
from onnx_optimizer import graph_util as gu
from onnx_optimizer.passes.base import Pass


class RemoveNoopSlice(Pass):
    name = "remove-noop-slice"
    description = "remove Slice nodes proven to keep the whole tensor (unit steps)"

    def run(self, model):
        graph = model.graph
        index = gu.tensor_index(graph)
        protected = gu.subgraph_ref_names(graph)
        changes = 0
        for node in [n for n in graph.node if n.op_type == "Slice"]:
            if not node.input or not node.input[0]:
                continue
            if len(node.input) > 4 and node.input[4]:
                steps = gu.const_array(graph, node.input[4])
                if steps is None or (steps.reshape(-1) != 1).any():
                    continue
            in_shape = gu.static_shape(index, node.input[0])
            out_shape = gu.static_shape(index, node.output[0])
            if in_shape is None or out_shape is None or in_shape != out_shape:
                continue
            if gu.try_bypass(graph, node, protected):
                changes += 1
        return changes
