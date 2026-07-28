"""Merge consecutive Transpose nodes; cancel pairs that compose to identity.

Pure data movement: for y = Transpose(x, p) and z = Transpose(y, q), z equals
Transpose(x, r) with r[i] = p[q[i]] element-for-element, so the rewrite is
bit-exact by construction. When r is the identity permutation the second node
is bypassed entirely. The first Transpose is left in place for any other
consumers; dead-code elimination sweeps it when it becomes unused.
"""
from onnx_optimizer import graph_util as gu
from onnx_optimizer.passes.base import Pass


def _perm_of(node, index):
    perm = gu.get_attr(node, "perm")
    if perm is not None:
        return list(perm)
    info = index.get(node.input[0]) if node.input and node.input[0] else None
    if info and info["shape"] is not None:
        return list(reversed(range(len(info["shape"]))))
    return None


class MergeTransposes(Pass):
    name = "merge-transposes"
    description = "compose consecutive Transposes into one; cancel inverse pairs"

    def run(self, model):
        graph = model.graph
        index = gu.tensor_index(graph)
        protected = gu.subgraph_ref_names(graph)
        changes = 0
        for node in [n for n in graph.node if n.op_type == "Transpose"]:
            if not node.input or not node.input[0]:
                continue
            prod = gu.producer_map(graph)
            upstream = prod.get(node.input[0])
            if upstream is None or upstream.op_type != "Transpose":
                continue
            if not upstream.input or not upstream.input[0]:
                continue
            p = _perm_of(upstream, index)
            q = _perm_of(node, index)
            if p is None or q is None or len(p) != len(q):
                continue
            composed = [p[qi] for qi in q]
            if composed == list(range(len(composed))):
                if gu.try_bypass(graph, node, protected, in_name=upstream.input[0]):
                    changes += 1
            else:
                node.input[0] = upstream.input[0]
                gu.set_attr_ints(node, "perm", composed)
                changes += 1
        return changes
