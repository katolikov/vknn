"""Prune graph inputs referenced by nothing.

Removes graph.input entries no node (or subgraph) reads and no graph output
re-exports. NOTE: this is the one default pass that changes the model's
SIGNATURE (feeding the pruned name afterwards is an error); outputs for any
feedable input set are unchanged, so bit-exactness holds. Disable with
--disable-pass prune-unused-inputs to keep the interface frozen. For an
initializer-backed input, pruning removes only the override hook -- the
initializer stays and becomes a foldable compile-time constant.
"""
from onnx_optimizer import graph_util as gu
from onnx_optimizer.passes.base import Pass


class PruneUnusedInputs(Pass):
    name = "prune-unused-inputs"
    description = "drop graph inputs nothing reads (changes the input signature)"

    def run(self, model):
        graph = model.graph
        used = gu.subgraph_ref_names(graph) | gu.graph_output_names(graph)
        for n in graph.node:
            used.update(i for i in n.input if i)
        changes = 0
        for vi in [vi for vi in graph.input if vi.name not in used]:
            graph.input.remove(vi)
            gu.remove_value_info(graph, vi.name)
            changes += 1
        return changes
