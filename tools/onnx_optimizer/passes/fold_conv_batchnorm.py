"""LOSSY: fold BatchNormalization into a preceding Conv's weights.

NOT bit-exact -- this is floating-point algebra (W' = W * scale/sqrt(var+eps),
B' = (B - mean) * scale/sqrt(var+eps) + bias) and reassociates the rounding
that the separate Conv+BN pair would perform. It therefore only runs under
--allow-lossy, where the byte-equality gate is replaced by a tolerance gate
and the report carries the observed max ULP / abs error.

Requirements: BN in inference form (single output), all four BN parameters
and the Conv weights compile-time constants, Conv output consumed only by the
BN. The arithmetic is done in float64 and rounded back once to the weight
dtype, which is the least-error way to fold but still not the original
rounding sequence.
"""
import numpy as np
from onnx import numpy_helper

from onnx_optimizer import graph_util as gu
from onnx_optimizer.passes.base import Pass


class FoldConvBatchnorm(Pass):
    name = "fold-conv-batchnorm"
    description = "LOSSY: fold BatchNormalization into the preceding Conv's weights"
    lossy = True

    def run(self, model):
        graph = model.graph
        protected = gu.subgraph_ref_names(graph)
        changes = 0
        for node in [n for n in graph.node if n.op_type == "BatchNormalization"]:
            if len(node.output) > 1 and any(o for o in node.output[1:]):
                continue  # training-form BN with live running stats
            uses = gu.use_counts(graph)
            outputs = gu.graph_output_names(graph)
            prod = gu.producer_map(graph)
            conv = prod.get(node.input[0]) if node.input and node.input[0] else None
            if conv is None or conv.op_type != "Conv" or gu.get_attr(conv, "group", 1) != 1:
                continue
            between = conv.output[0]
            if uses.get(between, 0) != 1 or between in outputs or between in protected:
                continue
            if len(node.input) < 5:
                continue
            params = [gu.const_array(graph, name) for name in node.input[1:5]]
            weights = gu.const_array(graph, conv.input[1]) if len(conv.input) > 1 else None
            bias = gu.const_array(graph, conv.input[2]) if len(conv.input) > 2 and conv.input[2] else None
            if weights is None or any(p is None for p in params):
                continue
            scale, shift, mean, var = [p.astype(np.float64) for p in params]
            eps = float(gu.get_attr(node, "epsilon", 1e-5))
            mult = scale / np.sqrt(var + eps)
            w64 = weights.astype(np.float64) * mult.reshape([-1] + [1] * (weights.ndim - 1))
            b64 = (bias.astype(np.float64) if bias is not None
                   else np.zeros(weights.shape[0], dtype=np.float64))
            b64 = (b64 - mean) * mult + shift
            w_name = gu.unique_name(graph, conv.input[1] + "_bnfold")
            b_name = gu.unique_name(graph, conv.output[0] + "_bnfold_bias")
            graph.initializer.append(numpy_helper.from_array(w64.astype(weights.dtype), w_name))
            graph.initializer.append(numpy_helper.from_array(b64.astype(weights.dtype), b_name))
            conv.input[1] = w_name
            if len(conv.input) > 2:
                conv.input[2] = b_name
            else:
                conv.input.append(b_name)
            # The BN becomes an identity over the rewritten Conv.
            if gu.try_bypass(graph, node, protected):
                changes += 1
        return changes
