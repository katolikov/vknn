"""Cancel Squeeze/Unsqueeze pairs that reconstruct the original tensor.

Squeeze and Unsqueeze only add/remove 1-sized dims and preserve the flat
element order, so a pair cancels exactly when the final shape equals the
initial shape. Two proofs are accepted:

  * static shapes: shape(pair input) == shape(pair output), both fully known;
  * equal axes: Squeeze(axes)->Unsqueeze(axes) or Unsqueeze(axes)->Squeeze(axes)
    with identical sorted non-negative axes lists reconstructs the shape by
    construction (no shape knowledge needed).

Axes come from the attribute (opset <= 12) or the second input (opset >= 13,
must be constant). A Squeeze with no axes (drop ALL 1-dims) only cancels via
the static-shape proof.
"""
from onnx_optimizer import graph_util as gu
from onnx_optimizer.passes.base import Pass

_PAIR = {"Squeeze": "Unsqueeze", "Unsqueeze": "Squeeze"}


def _axes_of(graph, node):
    axes = gu.get_attr(node, "axes")
    if axes is None and len(node.input) > 1 and node.input[1]:
        arr = gu.const_array(graph, node.input[1])
        if arr is not None:
            axes = [int(v) for v in arr.reshape(-1)]
    if axes is None:
        return None
    axes = [int(a) for a in axes]
    if any(a < 0 for a in axes):
        return None  # need the rank to normalize; the static-shape proof covers it
    return sorted(axes)


class CancelSqueezeUnsqueeze(Pass):
    name = "cancel-squeeze-unsqueeze"
    description = "cancel Squeeze/Unsqueeze pairs that reconstruct the input shape"

    def run(self, model):
        graph = model.graph
        index = gu.tensor_index(graph)
        protected = gu.subgraph_ref_names(graph)
        changes = 0
        for node in [n for n in graph.node if n.op_type in _PAIR]:
            if not node.input or not node.input[0]:
                continue
            prod = gu.producer_map(graph)
            upstream = prod.get(node.input[0])
            if upstream is None or upstream.op_type != _PAIR[node.op_type]:
                continue
            if not upstream.input or not upstream.input[0]:
                continue
            in_shape = gu.static_shape(index, upstream.input[0])
            out_shape = gu.static_shape(index, node.output[0])
            identity = in_shape is not None and in_shape == out_shape
            if not identity:
                a1 = _axes_of(graph, upstream)
                a2 = _axes_of(graph, node)
                identity = a1 is not None and a1 == a2
            if identity and gu.try_bypass(graph, node, protected, in_name=upstream.input[0]):
                changes += 1
        return changes
