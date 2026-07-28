"""CLI integration: optimize + verify end-to-end as subprocesses, on a small
but realistic CNN-with-shape-logic model (the graph shapes vknn ingests
poorly: movement chains, shape-computation subgraphs, Identity/Dropout)."""
import json
import os
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

from onnx_optimizer.tests.helpers import init, make_model, vi

_TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F = TensorProto.FLOAT


def _cli(module, *args):
    env = dict(os.environ, PYTHONPATH=_TOOLS_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run([sys.executable, "-m", module, *args],
                          capture_output=True, text=True, env=env, cwd=_TOOLS_DIR)


def _realistic_model():
    """Conv -> movement noise -> GlobalAveragePool -> shape-logic Reshape -> Gemm."""
    rng = np.random.default_rng(7)
    nodes = [
        helper.make_node("Identity", ["x"], ["x0"]),
        helper.make_node("Conv", ["x0", "w0"], ["c0"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c0"], ["r0"]),
        helper.make_node("Dropout", ["r0"], ["d0"]),
        helper.make_node("Transpose", ["d0"], ["t0"], perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", ["t0"], ["t1"], perm=[0, 3, 1, 2]),
        helper.make_node("GlobalAveragePool", ["t1"], ["g0"]),
        # shape-computation subgraph: Shape -> Gather -> Unsqueeze -> Concat
        helper.make_node("Shape", ["g0"], ["shp"]),
        helper.make_node("Gather", ["shp", "zero"], ["b_dim"], axis=0),
        helper.make_node("Unsqueeze", ["b_dim", "ax0"], ["b_vec"]),
        helper.make_node("Concat", ["b_vec", "negone"], ["tshape"], axis=0),
        helper.make_node("Reshape", ["g0", "tshape"], ["flat"]),
        helper.make_node("Gemm", ["flat", "w1", "b1"], ["y"]),
    ]
    inits = [
        init("w0", rng.standard_normal((8, 3, 3, 3)).astype(np.float32)),
        init("w1", rng.standard_normal((8, 10)).astype(np.float32)),
        init("b1", rng.standard_normal(10).astype(np.float32)),
        init("zero", np.array(0, dtype=np.int64)),
        init("ax0", np.array([0], dtype=np.int64)),
        init("negone", np.array([-1], dtype=np.int64)),
    ]
    return make_model(nodes, [vi("x", [1, 3, 8, 8])], [vi("y", [1, 10])], inits)


def test_optimize_cli_end_to_end(tmp_path):
    src = tmp_path / "model.onnx"
    dst = tmp_path / "model.opt.onnx"
    rep = tmp_path / "report.json"
    onnx.save(_realistic_model(), str(src))
    r = _cli("onnx_optimizer.optimize", "-i", str(src), "-o", str(dst),
             "--report", str(rep), "--target", "vknn", "--verify-samples", "4")
    assert r.returncode == 0, r.stdout + r.stderr
    assert dst.exists()
    report = json.loads(rep.read_text())
    assert report["verification"]["ok"]
    assert report["auto_reverted_passes"] == []
    before = report["model_before"]["nodes_by_op"]
    after = report["model_after"]["nodes_by_op"]
    for gone in ("Identity", "Dropout", "Transpose", "Shape", "Gather", "Concat"):
        assert after.get(gone, 0) == 0, gone
    assert before["Conv"] == after["Conv"] == 1
    assert report["vknn"]["available"]
    opt = onnx.load(str(dst))
    onnx.checker.check_model(opt)


def test_verify_cli_pass_and_fail(tmp_path):
    m = _realistic_model()
    a = tmp_path / "a.onnx"
    onnx.save(m, str(a))
    r = _cli("onnx_optimizer.verify", str(a), str(a), "--samples", "2")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "BIT-EXACT" in r.stdout

    mutated = onnx.ModelProto()
    mutated.CopyFrom(m)
    for t in mutated.graph.initializer:
        if t.name == "b1":
            arr = onnx.numpy_helper.to_array(t).copy()
            arr[0] = np.nextafter(arr[0], np.float32(np.inf))
            t.CopyFrom(onnx.numpy_helper.from_array(arr, t.name))
    b = tmp_path / "b.onnx"
    onnx.save(mutated, str(b))
    r = _cli("onnx_optimizer.verify", str(a), str(b), "--samples", "2")
    assert r.returncode == 1
    assert "NOT bit-exact" in r.stdout and "ULP" in r.stdout


def test_optimize_cli_rejects_unknown_pass(tmp_path):
    src = tmp_path / "model.onnx"
    onnx.save(_realistic_model(), str(src))
    r = _cli("onnx_optimizer.optimize", "-i", str(src), "-o", str(src) + ".out",
             "--only-pass", "no-such-pass")
    assert r.returncode == 2
    assert "unknown pass" in r.stderr


def test_list_passes():
    r = _cli("onnx_optimizer.optimize", "--list-passes")
    assert r.returncode == 0
    assert "remove-identity" in r.stdout
    assert "LOSSY" in r.stdout  # lossy passes are labeled


def test_external_data_roundtrip(tmp_path):
    """External-data models load, optimize, and re-verify (small stand-in for
    the >2 GiB path, which uses the same onnx external-data machinery)."""
    m = _realistic_model()
    src = tmp_path / "ext.onnx"
    onnx.save_model(m, str(src), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="ext.onnx.data",
                    size_threshold=0)
    assert (tmp_path / "ext.onnx.data").exists()
    dst = tmp_path / "ext.opt.onnx"
    r = _cli("onnx_optimizer.optimize", "-i", str(src), "-o", str(dst),
             "--verify-samples", "2")
    assert r.returncode == 0, r.stdout + r.stderr
    assert onnx.load(str(dst)).graph.node
