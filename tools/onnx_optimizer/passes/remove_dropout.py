"""Remove provably inference-mode Dropout nodes.

Only the identity form is erased, mirroring vknn's own import-time rule
(src/import/eliminate_dropout.cpp):

  * opset >= 12: the training_mode input must be absent, empty, or a
    compile-time-constant false;
  * opset 7..11: Dropout is defined as identity outside training;
  * opset <= 6: the is_test attribute must be 1;
  * the mask output, if declared, must be unconsumed -- a consumed mask is
    never fabricated.
"""
from onnx_optimizer import graph_util as gu
from onnx_optimizer.passes.base import Pass


class RemoveDropout(Pass):
    name = "remove-dropout"
    description = "remove inference-mode Dropout (constant-false training_mode, unused mask)"

    def run(self, model):
        graph = model.graph
        opset = gu.opset_of(model) or 0
        protected = gu.subgraph_ref_names(graph)
        changes = 0
        for node in [n for n in graph.node if n.op_type == "Dropout"]:
            if not node.input or not node.input[0]:
                continue
            uses = gu.use_counts(graph)
            outputs = gu.graph_output_names(graph)
            mask = node.output[1] if len(node.output) > 1 else ""
            if mask and (uses.get(mask) or mask in outputs or mask in protected):
                continue
            if opset >= 12:
                if len(node.input) > 2 and node.input[2]:
                    tm = gu.const_array(graph, node.input[2])
                    if tm is None or tm.size != 1 or bool(tm.reshape(-1)[0]):
                        continue
            elif opset <= 6:
                if not gu.get_attr(node, "is_test", 0):
                    continue
            # opset 7..11: identity in inference by definition.
            if gu.try_bypass(graph, node, protected):
                changes += 1
        return changes
