"""Verifier tests: the byte gate must catch what tolerances would forgive."""
import numpy as np
from onnx import TensorProto, helper

from onnx_optimizer.tests.helpers import init, make_model, vi
from onnx_optimizer.verify import Verifier, compare_arrays, verify_models

F = TensorProto.FLOAT


def _base_model(weight):
    return make_model(
        [helper.make_node("Add", ["x", "w"], ["y"])],
        [vi("x", [2, 3])], [vi("y", [2, 3])],
        [init("w", weight)])


def test_identical_models_pass():
    w = np.arange(6, dtype=np.float32).reshape(2, 3)
    result = verify_models(_base_model(w), _base_model(w), n_random=2)
    assert result["ok"]


def test_one_ulp_initializer_change_fails():
    w = np.arange(6, dtype=np.float32).reshape(2, 3)
    w2 = w.copy()
    w2.reshape(-1)[4] = np.nextafter(w2.reshape(-1)[4], np.float32(np.inf))
    result = verify_models(_base_model(w), _base_model(w2), n_random=2)
    assert not result["ok"]
    mm = [m for c in result["cases"] for m in c["mismatches"]][0]
    # The WEIGHT changed by 1 ULP; the ULP delta of the SUM depends on the
    # random addend's exponent, so allow a small bound rather than exactly 1.
    assert mm["reason"] == "bytes" and 1 <= mm["max_ulp"] <= 4
    assert mm["first_index"] == [1, 1]


def test_output_dtype_mismatch_fails():
    w = np.ones(3, dtype=np.float32)
    a = make_model([helper.make_node("Add", ["x", "w"], ["y"])],
                   [vi("x", [3])], [vi("y", [3])], [init("w", w)])
    b = make_model([helper.make_node("Add", ["x", "w"], ["s"]),
                    helper.make_node("Cast", ["s"], ["y"], to=TensorProto.DOUBLE)],
                   [vi("x", [3])], [vi("y", [3], TensorProto.DOUBLE)], [init("w", w)])
    result = verify_models(a, b, n_random=1, batteries=False)
    assert not result["ok"]
    assert any(m["reason"] == "dtype" for c in result["cases"] for m in c["mismatches"])


def test_missing_output_fails():
    w = np.ones(3, dtype=np.float32)
    a = make_model([helper.make_node("Add", ["x", "w"], ["y"]),
                    helper.make_node("Relu", ["y"], ["z"])],
                   [vi("x", [3])], [vi("y", [3]), vi("z", [3])], [init("w", w)])
    b = make_model([helper.make_node("Add", ["x", "w"], ["y"])],
                   [vi("x", [3])], [vi("y", [3])], [init("w", w)])
    result = verify_models(a, b, n_random=1, batteries=False)
    assert not result["ok"]
    assert any(m["reason"] == "missing-output" for c in result["cases"] for m in c["mismatches"])


def test_nan_payloads_compare_bitwise():
    # Two different NaN payloads are byte-different even though == and
    # allclose would both call them "equal NaNs".
    a = np.array([np.float32(np.nan)], dtype=np.float32)
    b = np.frombuffer(np.uint32(0x7FC00001).tobytes(), dtype=np.float32).copy()
    assert np.isnan(a[0]) and np.isnan(b[0])
    rec = compare_arrays("o", a, b)
    assert rec is not None and rec["reason"] == "bytes"
    rec_same = compare_arrays("o", a, a.copy())
    assert rec_same is None


def test_dynamic_dims_share_named_size():
    m = make_model(
        [helper.make_node("Add", ["a", "b"], ["y"])],
        [vi("a", ["n", 2]), vi("b", ["n", 2])], [vi("y", ["n", 2])])
    verifier = Verifier(m, n_random=4, seed=0, dyn_sizes=(1, 3), batteries=True)
    result = verifier.check(m)
    assert result["ok"]  # both dims resolved to the same size or Add would fail


def test_nan_inf_battery_present_and_gating():
    m = _base_model(np.ones((2, 3), dtype=np.float32))
    verifier = Verifier(m, n_random=1, seed=0, batteries=True)
    labels = [c[0] for c in verifier.cases]
    assert any("nan-inf" in label for label in labels)
    assert any("large-magnitude" in label for label in labels)
    # A model that flips NaN propagation must be caught by the battery.
    flipped = make_model(
        [helper.make_node("IsNaN", ["x"], ["m"]),
         helper.make_node("Where", ["m", "zero", "x"], ["s"]),
         helper.make_node("Add", ["s", "w"], ["y"])],
        [vi("x", [2, 3])], [vi("y", [2, 3])],
        [init("w", np.ones((2, 3), dtype=np.float32)),
         init("zero", np.zeros((2, 3), dtype=np.float32))])
    result = verifier.check(flipped)
    assert not result["ok"]
    bad_cases = [c["case"] for c in result["cases"] if not c["ok"]]
    assert any("nan-inf" in c for c in bad_cases)
    assert all("nan-inf" in c for c in bad_cases)  # finite cases identical


def test_int64_ulp_not_reported_for_ints():
    a = np.array([1, 2], dtype=np.int64)
    b = np.array([1, 3], dtype=np.int64)
    rec = compare_arrays("o", a, b)
    assert rec["reason"] == "bytes" and "max_ulp" not in rec
