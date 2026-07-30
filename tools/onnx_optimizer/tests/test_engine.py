"""Engine tests: per-pass auto-revert, fixpoint convergence, metadata
preservation. A deliberately broken pass must be caught by the gate, reverted,
and reported -- never shipped."""
import numpy as np
import onnx
from onnx import helper

from onnx_optimizer import report as report_mod
from onnx_optimizer.optimize import Engine
from onnx_optimizer.passes import build_pipeline
from onnx_optimizer.passes.base import Pass
from onnx_optimizer.tests.helpers import assert_bitexact, init, make_model, optimize, vi


class _EvilPass(Pass):
    """Claims to be structural but flips a weight sign: NOT bit-exact."""
    name = "evil-flip-weight"
    description = "test-only: corrupts the first float initializer"

    def run(self, model):
        for t in model.graph.initializer:
            arr = onnx.numpy_helper.to_array(t)
            if arr.dtype == np.float32:
                t.CopyFrom(onnx.numpy_helper.from_array(-arr, t.name))
                return 1
        return 0


class _CrashPass(Pass):
    name = "evil-crash"
    description = "test-only: raises mid-run after mangling the graph"

    def run(self, model):
        del model.graph.node[:]
        raise RuntimeError("boom")


def _model():
    return make_model(
        [helper.make_node("Identity", ["x"], ["a"]),
         helper.make_node("Add", ["a", "w"], ["y"])],
        [vi("x", [2, 2])], [vi("y", [2, 2])],
        [init("w", np.array([[1, 2], [3, 4]], dtype=np.float32))])


def _run(model, passes):
    work = onnx.ModelProto()
    work.CopyFrom(model)
    report = report_mod.Report()
    engine = Engine(work, passes, report, gate_samples=2, seed=0)
    return engine.run(), report


def test_evil_pass_is_auto_reverted_and_good_passes_still_apply():
    m = _model()
    passes = build_pipeline() + [_EvilPass()]
    opt, report = _run(m, passes)
    assert "evil-flip-weight" in report.reverted
    statuses = {(p["pass"], p["status"]) for p in report.passes}
    assert ("evil-flip-weight", "reverted") in statuses
    assert ("remove-identity", "applied") in statuses
    # The shipped model is the GOOD passes' result, bit-exact vs the input.
    assert not any(n.op_type == "Identity" for n in opt.graph.node)
    assert_bitexact(m, opt)
    # ... and the weight is untouched.
    w = onnx.numpy_helper.to_array(opt.graph.initializer[0])
    assert w.tolist() == [[1, 2], [3, 4]]


def test_crashing_pass_is_reverted():
    m = _model()
    opt, report = _run(m, build_pipeline() + [_CrashPass()])
    assert "evil-crash" in report.reverted
    assert len(opt.graph.node) >= 1
    assert_bitexact(m, opt)


def test_reverted_pass_is_disabled_for_later_iterations():
    m = _model()
    _, report = _run(m, [_EvilPass()] + build_pipeline())
    reverts = [p for p in report.passes if p["status"] == "reverted"]
    assert len(reverts) == 1  # not retried every iteration


def test_fixpoint_terminates_and_is_stable():
    m = _model()
    opt1, report1 = _run(m, build_pipeline())
    opt2, report2 = _run(opt1, build_pipeline())
    assert report1.iterations <= 10
    # A second full run over the optimized model changes nothing.
    assert not [p for p in report2.passes if p["status"] == "applied"]


def test_metadata_preserved():
    m = _model()
    m.producer_name = "unit-test-producer"
    m.metadata_props.add(key="k", value="v")
    ir, opsets = m.ir_version, [(i.domain, i.version) for i in m.opset_import]
    opt, _ = _run(m, build_pipeline())
    assert opt.producer_name == "unit-test-producer"
    assert [(p.key, p.value) for p in opt.metadata_props] == [("k", "v")]
    assert opt.ir_version == ir
    assert [(i.domain, i.version) for i in opt.opset_import] == opsets


def test_only_and_disable_filters():
    m = _model()
    opt, report = optimize(m, only=["remove-identity"])
    applied = {p["pass"] for p in report.passes if p["status"] == "applied"}
    assert applied == {"remove-identity"}
    opt, report = optimize(m, disable=["remove-identity"])
    assert any(n.op_type == "Identity" for n in opt.graph.node)
