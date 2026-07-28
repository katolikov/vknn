"""Shared GraphProto helpers for the optimizer passes.

Everything here mutates or inspects onnx protos directly -- there is no
intermediate IR. The helpers encode the safety rules every pass relies on:

  * a tensor name referenced anywhere inside a subgraph (If/Loop/Scan bodies)
    is PROTECTED: passes never rewrite or remove it, because subgraphs may
    capture outer-scope names and the passes only edit the top-level graph;
  * an initializer that is also listed in graph.input is OVERRIDABLE at run
    time, so it is never treated as a compile-time constant;
  * removing a node whose output is a graph output requires renaming the
    node's input tensor at its producer instead of rewiring consumers, and is
    refused when that rename is not safe (input is a graph input/output,
    an initializer, or protected).
"""
import numpy as np
from onnx import AttributeProto, TensorProto, helper, numpy_helper


def opset_of(model, domain=""):
    """The imported opset version for `domain` (default ai.onnx), or None."""
    for imp in model.opset_import:
        if (imp.domain or "") == (domain or ""):
            return imp.version
    return None


def get_attr(node, name, default=None):
    """Attribute value as a plain Python object, or `default` when absent."""
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


def set_attr_ints(node, name, values):
    """Replace (or add) an ints attribute in place."""
    for a in node.attribute:
        if a.name == name:
            del a.ints[:]
            a.ints.extend(int(v) for v in values)
            a.type = AttributeProto.INTS
            return
    node.attribute.extend([helper.make_attribute(name, [int(v) for v in values])])


def tensor_index(graph):
    """name -> {"dtype": elem_type int or None, "shape": [int|None,...] or None}.

    Built from graph inputs/outputs/value_info (as inferred) and initializers.
    Symbolic or unknown dims are None; a tensor with no shape info at all maps
    shape to None.
    """
    index = {}
    for coll in (graph.input, graph.value_info, graph.output):
        for vi in coll:
            if not vi.type.HasField("tensor_type"):
                index[vi.name] = {"dtype": None, "shape": None}
                continue
            tt = vi.type.tensor_type
            dtype = tt.elem_type if tt.elem_type != TensorProto.UNDEFINED else None
            if not tt.HasField("shape"):
                index[vi.name] = {"dtype": dtype, "shape": None}
                continue
            dims = []
            for d in tt.shape.dim:
                dims.append(int(d.dim_value) if d.HasField("dim_value") else None)
            index[vi.name] = {"dtype": dtype, "shape": dims}
    for init in graph.initializer:
        index[init.name] = {"dtype": init.data_type, "shape": [int(d) for d in init.dims]}
    return index


def static_shape(index, name):
    """Fully-static shape of `name` as a tuple of ints, or None."""
    info = index.get(name)
    if not info or info["shape"] is None:
        return None
    if any(d is None or d < 0 for d in info["shape"]):
        return None
    return tuple(info["shape"])


def dtype_of(index, name):
    info = index.get(name)
    return info["dtype"] if info else None


def graph_input_names(graph):
    return {vi.name for vi in graph.input}


def graph_output_names(graph):
    return {vi.name for vi in graph.output}


def initializer_names(graph):
    return {t.name for t in graph.initializer}


def _iter_subgraphs(node):
    for a in node.attribute:
        if a.type == AttributeProto.GRAPH:
            yield a.g
        elif a.type == AttributeProto.GRAPHS:
            for g in a.graphs:
                yield g


def subgraph_ref_names(graph):
    """Every tensor name referenced anywhere inside any subgraph, recursively.

    Deliberately over-approximate (includes names local to the subgraph): a
    protected name at worst blocks an optimization, never breaks one.
    """
    names = set()

    def collect(g):
        for vi in list(g.input) + list(g.output) + list(g.value_info):
            names.add(vi.name)
        for t in g.initializer:
            names.add(t.name)
        for n in g.node:
            names.update(i for i in n.input if i)
            names.update(o for o in n.output if o)
            for sub in _iter_subgraphs(n):
                collect(sub)

    for n in graph.node:
        for sub in _iter_subgraphs(n):
            collect(sub)
    return names


def node_subgraph_refs(node):
    """Names referenced inside this node's own subgraphs (for dependency edges)."""
    names = set()

    def collect(g):
        for n in g.node:
            names.update(i for i in n.input if i)
            for sub in _iter_subgraphs(n):
                collect(sub)

    for sub in _iter_subgraphs(node):
        collect(sub)
    return names


def has_subgraph(node):
    return any(True for _ in _iter_subgraphs(node))


def producer_map(graph):
    """output tensor name -> producing NodeProto (top-level nodes only)."""
    prod = {}
    for n in graph.node:
        for o in n.output:
            if o:
                prod[o] = n
    return prod


def use_counts(graph):
    """tensor name -> number of top-level node-input references."""
    counts = {}
    for n in graph.node:
        for i in n.input:
            if i:
                counts[i] = counts.get(i, 0) + 1
    return counts


def constant_node_array(node):
    """The value of a Constant node as a numpy array, or None (sparse/string)."""
    for a in node.attribute:
        if a.name == "value":
            return numpy_helper.to_array(a.t)
        if a.name == "value_float":
            return np.array(a.f, dtype=np.float32)
        if a.name == "value_int":
            return np.array(a.i, dtype=np.int64)
        if a.name == "value_floats":
            return np.array(list(a.floats), dtype=np.float32)
        if a.name == "value_ints":
            return np.array(list(a.ints), dtype=np.int64)
    return None


def const_array(graph, name):
    """`name` as a compile-time-constant numpy array, or None.

    Sources: initializers (excluding overridable ones listed in graph.input)
    and Constant-node outputs.
    """
    if not name:
        return None
    inputs = graph_input_names(graph)
    for init in graph.initializer:
        if init.name == name:
            return None if name in inputs else numpy_helper.to_array(init)
    for n in graph.node:
        if n.op_type == "Constant" and name in n.output:
            return constant_node_array(n)
    return None


def unique_name(graph, base):
    """A tensor/node name starting with `base` that collides with nothing."""
    taken = set()
    for coll in (graph.input, graph.output, graph.value_info):
        taken.update(vi.name for vi in coll)
    taken.update(t.name for t in graph.initializer)
    for n in graph.node:
        taken.add(n.name)
        taken.update(n.input)
        taken.update(n.output)
    taken.update(subgraph_ref_names(graph))
    if base not in taken:
        return base
    k = 0
    while "%s_%d" % (base, k) in taken:
        k += 1
    return "%s_%d" % (base, k)


def remove_value_info(graph, name):
    for i, vi in enumerate(graph.value_info):
        if vi.name == name:
            del graph.value_info[i]
            return


def try_bypass(graph, node, protected, in_name=None):
    """Remove a no-op `node`, making its first output an alias of `in_name`.

    `in_name` defaults to the node's first input. Handles the graph-output
    case by renaming `in_name` to the output name at its producer. Returns
    True when the node was removed; False when the rewrite would be unsafe
    (the caller must then leave the node alone).
    """
    in_name = in_name if in_name is not None else node.input[0]
    out = node.output[0]
    if not in_name or not out or in_name == out:
        return False
    if out in protected:
        return False
    outputs = graph_output_names(graph)
    if out in outputs:
        # Rename in_name -> out at the producer and at every other consumer.
        if (in_name in graph_input_names(graph) or in_name in initializer_names(graph)
                or in_name in outputs or in_name in protected):
            return False
        for n in graph.node:
            if n is node:
                continue
            n.input[:] = [out if x == in_name else x for x in n.input]
            n.output[:] = [out if x == in_name else x for x in n.output]
        remove_value_info(graph, in_name)
    else:
        for n in graph.node:
            if n is node:
                continue
            n.input[:] = [in_name if x == out else x for x in n.input]
        remove_value_info(graph, out)
    graph.node.remove(node)
    return True


def toposort(graph):
    """Topologically re-sort graph.node in place (stable; raises on a cycle).

    Dependency edges include names captured by a node's subgraphs, so a Loop
    body reading an outer tensor sorts after that tensor's producer.
    """
    nodes = list(graph.node)
    prod = {}
    for i, n in enumerate(nodes):
        for o in n.output:
            if o:
                prod[o] = i
    indeg = [0] * len(nodes)
    adj = [[] for _ in nodes]
    for i, n in enumerate(nodes):
        deps = set(x for x in n.input if x)
        deps |= node_subgraph_refs(n)
        for d in deps:
            j = prod.get(d)
            if j is not None and j != i:
                adj[j].append(i)
                indeg[i] += 1
    import heapq
    ready = [i for i in range(len(nodes)) if indeg[i] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        i = heapq.heappop(ready)
        order.append(i)
        for j in adj[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(ready, j)
    if len(order) != len(nodes):
        raise ValueError("graph contains a cycle; cannot topologically sort")
    rank = {id(nodes[i]): k for k, i in enumerate(order)}
    graph.node.sort(key=lambda n: rank[id(n)])


def op_histogram(graph):
    """op_type -> node count over the top-level graph."""
    hist = {}
    for n in graph.node:
        hist[n.op_type] = hist.get(n.op_type, 0) + 1
    return hist
