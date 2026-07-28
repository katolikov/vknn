"""Remove Transpose nodes with an identity permutation.

An explicit perm of [0, 1, ..., r-1] is an identity for any input. An ABSENT
perm means "reverse all dims", which is only an identity for rank <= 1 -- and
only removed when the rank is statically known.
"""
from onnx_optimizer import graph_util as gu
from onnx_optimizer.passes.base import Pass


class RemoveNoopTranspose(Pass):
    name = "remove-noop-transpose"
    description = "remove Transpose nodes whose permutation is the identity"

    def run(self, model):
        graph = model.graph
        index = gu.tensor_index(graph)
        protected = gu.subgraph_ref_names(graph)
        changes = 0
        for node in [n for n in graph.node if n.op_type == "Transpose"]:
            if not node.input or not node.input[0]:
                continue
            perm = gu.get_attr(node, "perm")
            if perm is not None:
                if list(perm) != list(range(len(perm))):
                    continue
            else:
                info = index.get(node.input[0])
                if not info or info["shape"] is None or len(info["shape"]) > 1:
                    continue
            if gu.try_bypass(graph, node, protected):
                changes += 1
        return changes
