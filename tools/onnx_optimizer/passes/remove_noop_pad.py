"""Remove Pad nodes whose pads are all zero.

With all-zero pads the mode (constant/reflect/edge/wrap) and constant_value
are irrelevant: the output is the input. Pads come from the attribute
(opset <= 10) or the second input (opset >= 11, must be a compile-time
constant); the optional axes input does not matter when every pad is zero.
"""
from onnx_optimizer import graph_util as gu
from onnx_optimizer.passes.base import Pass


class RemoveNoopPad(Pass):
    name = "remove-noop-pad"
    description = "remove Pad nodes whose pads are all zero"

    def run(self, model):
        graph = model.graph
        protected = gu.subgraph_ref_names(graph)
        changes = 0
        for node in [n for n in graph.node if n.op_type == "Pad"]:
            if not node.input or not node.input[0]:
                continue
            if len(node.input) > 1 and node.input[1]:
                pads = gu.const_array(graph, node.input[1])
                if pads is None or (pads.reshape(-1) != 0).any():
                    continue
            else:
                pads = gu.get_attr(node, "pads", gu.get_attr(node, "paddings"))
                if pads is None or any(int(p) != 0 for p in pads):
                    continue
            if gu.try_bypass(graph, node, protected):
                changes += 1
        return changes
