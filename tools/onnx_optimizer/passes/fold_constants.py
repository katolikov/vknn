"""Constant folding, computed by the SAME deterministic reference runtime the
verifier uses (onnxruntime CPU, ORT_DISABLE_ALL, single-threaded).

Every node whose inputs are all compile-time constants (initializers not
overridable via graph.input, or already-folded values) is executed in
isolation as a single-node model on the reference session, and its outputs
become initializers. Because folding and verification share one runtime
configuration, a folded constant is bit-for-bit the value the reference would
have produced at run time -- the fold cannot drift from the gate.

Never folded: nondeterministic ops (Random*, Multinomial, Bernoulli, Dropout),
nodes with subgraphs, calls into model-local functions, nodes whose output is
a graph output or is captured by a subgraph, and nodes whose folded outputs
would exceed --max-fold-bytes (default 16 MiB) -- folding a huge Expand would
trade a cheap op for megabytes of weights. Nodes ORT itself cannot run in
isolation are skipped and remembered, not retried every sweep.
"""
import numpy as np
from onnx import helper, numpy_helper

from onnx_optimizer import graph_util as gu
from onnx_optimizer.passes.base import Pass

_NONDETERMINISTIC = {"RandomNormal", "RandomUniform", "RandomNormalLike",
                     "RandomUniformLike", "Multinomial", "Bernoulli", "Dropout"}

DEFAULT_MAX_FOLD_BYTES = 16 * 1024 * 1024


class FoldConstants(Pass):
    name = "fold-constants"
    description = "fold constant subgraphs via the deterministic reference runtime"

    def __init__(self, max_fold_bytes=DEFAULT_MAX_FOLD_BYTES):
        self.max_fold_bytes = max_fold_bytes
        self._skip = set()  # node keys that failed or exceeded the cap; never retried

    def _node_key(self, node):
        return node.SerializeToString()

    def _fold_one(self, model, node, const_arrays):
        """Run `node` alone on the reference runtime; output name -> array."""
        from onnx_optimizer.verify import ReferenceSession
        index = gu.tensor_index(model.graph)
        inits, seen = [], set()
        for name in node.input:
            if name and name not in seen:
                seen.add(name)
                inits.append(numpy_helper.from_array(const_arrays[name], name))
        outs = []
        for name in node.output:
            if not name:
                continue
            dtype = gu.dtype_of(index, name)
            if dtype is None:
                return None  # dtype unknown even after shape inference: skip
            outs.append(helper.make_tensor_value_info(name, dtype, None))
        if not outs:
            return None
        sub = helper.make_graph([node], "fold_one", [], outs, initializer=inits)
        sub_model = helper.make_model(sub, opset_imports=model.opset_import)
        sub_model.ir_version = model.ir_version
        session = ReferenceSession(sub_model)
        return session.run({})

    def run(self, model):
        graph = model.graph
        protected = gu.subgraph_ref_names(graph)
        outputs = gu.graph_output_names(graph)
        overridable = gu.graph_input_names(graph)
        func_keys = {(f.domain, f.name) for f in model.functions}

        const_arrays = {}
        for init in graph.initializer:
            if init.name not in overridable:
                const_arrays[init.name] = None  # lazy; materialized on demand

        def get_const(name):
            arr = const_arrays.get(name)
            if arr is None and name in const_arrays:
                for init in graph.initializer:
                    if init.name == name:
                        arr = numpy_helper.to_array(init)
                        const_arrays[name] = arr
                        break
            return arr

        changes = 0
        gu.toposort(graph)
        for node in list(graph.node):
            if node.op_type in _NONDETERMINISTIC or node.op_type == "Constant":
                continue
            if gu.has_subgraph(node) or (node.domain, node.op_type) in func_keys:
                continue
            if any(o in outputs or o in protected for o in node.output if o):
                continue
            names = [i for i in node.input if i]
            if not names or not all(i in const_arrays for i in names):
                continue
            key = self._node_key(node)
            if key in self._skip:
                continue
            arrays = {i: get_const(i) for i in names}
            if any(a is None for a in arrays.values()):
                continue
            try:
                folded = self._fold_one(model, node, arrays)
            except Exception:
                folded = None
            if folded is None:
                self._skip.add(key)
                continue
            if sum(a.nbytes for a in folded.values()) > self.max_fold_bytes:
                self._skip.add(key)
                continue
            for name, arr in folded.items():
                if not arr.flags["C_CONTIGUOUS"]:
                    arr = np.ascontiguousarray(arr)  # would promote 0-d to 1-d if unconditional
                graph.initializer.append(numpy_helper.from_array(arr, name))
                const_arrays[name] = arr
                gu.remove_value_info(graph, name)
            graph.node.remove(node)
            changes += 1
        return changes
