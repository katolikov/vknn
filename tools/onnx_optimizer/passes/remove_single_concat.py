"""Remove Concat nodes with a single input (identity by definition)."""
from onnx_optimizer import graph_util as gu
from onnx_optimizer.passes.base import Pass


class RemoveSingleConcat(Pass):
    name = "remove-single-concat"
    description = "remove single-input Concat nodes"

    def run(self, model):
        graph = model.graph
        protected = gu.subgraph_ref_names(graph)
        changes = 0
        for node in [n for n in graph.node if n.op_type == "Concat"]:
            if len([i for i in node.input if i]) != 1:
                continue
            if gu.try_bypass(graph, node, protected, in_name=node.input[0]):
                changes += 1
        return changes
