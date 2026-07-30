"""Remove Identity nodes (consumers rewired to the producer)."""
from onnx_optimizer import graph_util as gu
from onnx_optimizer.passes.base import Pass


class RemoveIdentity(Pass):
    name = "remove-identity"
    description = "remove Identity nodes, rewiring consumers to the source tensor"

    def run(self, model):
        graph = model.graph
        protected = gu.subgraph_ref_names(graph)
        changes = 0
        for node in [n for n in graph.node if n.op_type == "Identity"]:
            if node.input and node.input[0] and gu.try_bypass(graph, node, protected):
                changes += 1
        return changes
