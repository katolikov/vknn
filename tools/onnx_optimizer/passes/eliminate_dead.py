"""Dead-code elimination: nodes, initializers, and stale value_info.

A node is live iff one of its outputs (transitively) feeds a graph output;
the walk counts names captured by a live node's subgraphs as uses, so a Loop
body's outer references keep their producers alive. Dead nodes are removed,
then initializers referenced by nothing (and not declared as graph inputs --
removing those would change the model signature; see prune-unused-inputs),
then value_info entries for names that no longer exist.
"""
from onnx_optimizer import graph_util as gu
from onnx_optimizer.passes.base import Pass


class EliminateDead(Pass):
    name = "dce"
    description = "drop nodes/initializers unreachable from the graph outputs"

    def run(self, model):
        graph = model.graph
        changes = 0

        producers = {}
        for n in graph.node:
            for o in n.output:
                if o:
                    producers.setdefault(o, []).append(n)

        needed = set(gu.graph_output_names(graph))
        queue = list(needed)
        live = set()
        while queue:
            name = queue.pop()
            for n in producers.get(name, ()):
                if id(n) in live:
                    continue
                live.add(id(n))
                uses = set(i for i in n.input if i) | gu.node_subgraph_refs(n)
                for u in uses:
                    if u not in needed:
                        needed.add(u)
                        queue.append(u)

        for n in [n for n in graph.node if id(n) not in live]:
            graph.node.remove(n)
            changes += 1

        graph_inputs = gu.graph_input_names(graph)
        for init in [t for t in graph.initializer
                     if t.name not in needed and t.name not in graph_inputs]:
            graph.initializer.remove(init)
            changes += 1

        existing = set(gu.graph_input_names(graph)) | {t.name for t in graph.initializer}
        for n in graph.node:
            existing.update(o for o in n.output if o)
        for vi in [vi for vi in graph.value_info if vi.name not in existing]:
            graph.value_info.remove(vi)
            # bookkeeping only; not counted as a graph change
        return changes
