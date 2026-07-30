"""Deduplicate byte-identical initializers.

Two initializers are duplicates when dtype, dims, and raw payload bytes all
match; consumers of the duplicate are rewired to the first occurrence and the
duplicate is dropped. Names that are graph inputs/outputs or captured by a
subgraph are left alone.
"""
from onnx import numpy_helper

from onnx_optimizer import graph_util as gu
from onnx_optimizer.passes.base import Pass


class DedupInitializers(Pass):
    name = "dedup-initializers"
    description = "merge byte-identical initializers"

    def run(self, model):
        graph = model.graph
        protected = gu.subgraph_ref_names(graph) | gu.graph_input_names(graph) \
            | gu.graph_output_names(graph)
        canonical = {}
        drop = []
        rewrites = {}
        for init in graph.initializer:
            if init.name in protected:
                continue
            key = (init.data_type, tuple(init.dims),
                   numpy_helper.to_array(init).tobytes())
            first = canonical.get(key)
            if first is None:
                canonical[key] = init.name
            else:
                rewrites[init.name] = first
                drop.append(init)
        if not rewrites:
            return 0
        for node in graph.node:
            node.input[:] = [rewrites.get(i, i) for i in node.input]
        for init in drop:
            graph.initializer.remove(init)
            gu.remove_value_info(graph, init.name)
        return len(drop)
