"""Common subexpression elimination.

Two nodes compute the same value when they have the same op_type and domain,
byte-identical attributes, and identical input tensor names -- for a
DETERMINISTIC op. Duplicates are removed and their outputs rewired to the
first occurrence. Nondeterministic ops, nodes with subgraph attributes, and
nodes whose outputs are graph outputs or subgraph-captured are never merged.
Runs in topological order so chains of duplicates collapse in one sweep.
"""
from onnx_optimizer import graph_util as gu
from onnx_optimizer.passes.base import Pass

_NONDETERMINISTIC = {"RandomNormal", "RandomUniform", "RandomNormalLike",
                     "RandomUniformLike", "Multinomial", "Bernoulli", "Dropout"}


class EliminateCommonSubexpr(Pass):
    name = "cse"
    description = "merge duplicate nodes (same op, attributes, and inputs)"

    def run(self, model):
        graph = model.graph
        protected = gu.subgraph_ref_names(graph)
        outputs = gu.graph_output_names(graph)
        gu.toposort(graph)
        changes = 0
        seen = {}
        rewrites = {}  # old tensor name -> canonical tensor name
        dead = []
        for node in graph.node:
            node.input[:] = [rewrites.get(i, i) for i in node.input]
            if node.op_type in _NONDETERMINISTIC or gu.has_subgraph(node):
                continue
            if any(o in outputs or o in protected for o in node.output if o):
                continue
            attrs = tuple(sorted((a.name, a.SerializeToString()) for a in node.attribute))
            key = (node.op_type, node.domain, tuple(node.input), attrs,
                   tuple(bool(o) for o in node.output))
            first = seen.get(key)
            if first is None:
                seen[key] = node
                continue
            for old, new in zip(node.output, first.output):
                if old:
                    rewrites[old] = new
                    gu.remove_value_info(graph, old)
            dead.append(node)
            changes += 1
        for node in dead:
            graph.node.remove(node)
        return changes
