"""Fold Shape and Size nodes over statically-shaped tensors into initializers.

A tensor's shape is metadata, so this is exact whenever the answer is fully
determined at compile time:

  * Shape: every dim inside the node's [start, end) window (opset 15+ attrs,
    default full range) must be static -- dynamic dims OUTSIDE the window
    don't block the fold;
  * Size: every dim must be static.

This is the pass that unlocks constant-folding of the int64 shape-arithmetic
subgraphs transformer exports carry -- chains vknn would otherwise take CPU
fallbacks on (Cast-from-int64 has no GPU path).
"""
import numpy as np
from onnx import numpy_helper

from onnx_optimizer import graph_util as gu
from onnx_optimizer.passes.base import Pass


class FoldShapeOps(Pass):
    name = "fold-shape-ops"
    description = "replace Shape/Size of statically-shaped tensors with int64 initializers"

    def run(self, model):
        graph = model.graph
        index = gu.tensor_index(graph)
        protected = gu.subgraph_ref_names(graph)
        outputs = gu.graph_output_names(graph)
        changes = 0
        for node in [n for n in graph.node if n.op_type in ("Shape", "Size")]:
            out = node.output[0] if node.output else ""
            if not out or out in outputs or out in protected:
                continue
            if not node.input or not node.input[0]:
                continue
            info = index.get(node.input[0])
            if not info or info["shape"] is None:
                continue
            dims = info["shape"]
            if node.op_type == "Size":
                if any(d is None for d in dims):
                    continue
                arr = np.array(int(np.prod(dims, dtype=np.int64)) if dims else 1, dtype=np.int64)
            else:
                rank = len(dims)
                start = int(gu.get_attr(node, "start", 0))
                end = int(gu.get_attr(node, "end", rank))
                start = max(start + rank, 0) if start < 0 else min(start, rank)
                end = max(end + rank, 0) if end < 0 else min(end, rank)
                window = dims[start:end]
                if any(d is None for d in window):
                    continue
                arr = np.array(window, dtype=np.int64)
            graph.initializer.append(numpy_helper.from_array(arr, out))
            graph.node.remove(node)
            gu.remove_value_info(graph, out)
            changes += 1
        return changes
