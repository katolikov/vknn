"""Shared test helpers: tiny model builders and bit-exact assertions."""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx_optimizer import graph_util as gu
from onnx_optimizer import report as report_mod
from onnx_optimizer.optimize import Engine
from onnx_optimizer.passes import build_pipeline
from onnx_optimizer.verify import Verifier

OPSET = 17


def make_model(nodes, inputs, outputs, initializers=(), opset=OPSET, ir_version=8):
    graph = helper.make_graph(list(nodes), "test_graph", list(inputs), list(outputs),
                              initializer=list(initializers))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = ir_version
    onnx.checker.check_model(model)
    return model


def vi(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def init(name, array):
    return numpy_helper.from_array(np.asarray(array), name)


def optimize(model, only=None, disable=None, allow_lossy=False, max_iterations=10):
    """Run the real engine (per-pass gating included); (optimized, report)."""
    work = onnx.ModelProto()
    work.CopyFrom(model)
    report = report_mod.Report()
    passes = build_pipeline(only=only, disable=disable, allow_lossy=allow_lossy)
    engine = Engine(work, passes, report, gate_samples=2, seed=0,
                    dyn_sizes=(1, 3), max_iterations=max_iterations)
    return engine.run(), report


def assert_bitexact(reference, candidate, n_random=3, batteries=True):
    verifier = Verifier(reference, n_random=n_random, seed=0, dyn_sizes=(1, 3),
                        batteries=batteries)
    result = verifier.check(candidate)
    assert result["ok"], "not bit-exact: %s" % result


def count_ops(model, op_type):
    return gu.op_histogram(model.graph).get(op_type, 0)
