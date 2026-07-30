"""Remove Cast nodes whose target dtype equals the input dtype.

Strictly same-dtype only: a precision round-trip such as fp32->fp16->fp32 is
NOT an identity (it quantizes the mantissa) and is deliberately left intact.
"""
from onnx_optimizer import graph_util as gu
from onnx_optimizer.passes.base import Pass


class RemoveNoopCast(Pass):
    name = "remove-noop-cast"
    description = "remove Cast nodes whose target dtype equals the source dtype"

    def run(self, model):
        graph = model.graph
        index = gu.tensor_index(graph)
        protected = gu.subgraph_ref_names(graph)
        changes = 0
        for node in [n for n in graph.node if n.op_type == "Cast"]:
            if not node.input or not node.input[0]:
                continue
            to = gu.get_attr(node, "to")
            src = gu.dtype_of(index, node.input[0])
            if to is None or src is None or int(to) != int(src):
                continue
            if gu.try_bypass(graph, node, protected):
                changes += 1
        return changes
