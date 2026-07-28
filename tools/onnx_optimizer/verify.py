"""Bit-exact ONNX model equivalence verifier (library + CLI).

Reference runtime: onnxruntime CPU, configured deterministically --
graph_optimization_level=ORT_DISABLE_ALL (so ORT's own rewrites cannot mask a
difference), intra/inter_op_num_threads=1, CPUExecutionProvider only.

Two models are equivalent iff, for every generated input case, they produce
the same output-name set and every output matches in dtype, shape, and RAW
BYTES (a.tobytes() == b.tobytes()). No atol/rtol: NaNs compare bitwise. On a
mismatch the report carries the first differing index, both values, the
mismatch count, and the ULP distance for float dtypes.

Input cases: fixed-seed random tensors per graph input (dtype/shape aware;
symbolic dims resolved to several concrete sizes, same-named dims kept
consistent), plus edge batteries (zeros / ones / negatives / large magnitudes
/ NaN-Inf propagation). Every case that the ORIGINAL model can run is a hard
gate; a case the original itself fails to run is skipped and reported.

Standalone usage (from tools/, or as a plain script):
  python -m onnx_optimizer.verify a.onnx b.onnx [--samples 8] [--seed 0] [--json out.json]

Exit code: 0 = bit-exact on all cases, 1 = mismatch, 2 = bad invocation.
"""
import argparse
import json
import os
import sys
import tempfile

if __package__ in (None, ""):  # running as a plain script: put tools/ on the path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import onnx
from onnx import TensorProto

# Serialized protobufs are hard-capped at 2 GiB; above ~1.9 GiB we spill
# initializers to an external-data file and hand onnxruntime a path instead.
_PROTO_SPILL_BYTES = 1_900_000_000

_NP_DTYPE = {
    TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16,
    TensorProto.DOUBLE: np.float64, TensorProto.INT64: np.int64,
    TensorProto.INT32: np.int32, TensorProto.INT16: np.int16,
    TensorProto.INT8: np.int8, TensorProto.UINT8: np.uint8,
    TensorProto.UINT16: np.uint16, TensorProto.UINT32: np.uint32,
    TensorProto.UINT64: np.uint64, TensorProto.BOOL: np.bool_,
}

_FLOAT_BIG = {np.dtype(np.float16): 6.0e4, np.dtype(np.float32): 1.0e30,
              np.dtype(np.float64): 1.0e300}


def default_session_options():
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return so


def reference_config():
    """The verification config, for reports."""
    import onnxruntime as ort
    return {
        "runtime": "onnxruntime %s" % ort.__version__,
        "providers": ["CPUExecutionProvider"],
        "graph_optimization_level": "ORT_DISABLE_ALL",
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "comparison": "byte-identical (shape + dtype + tobytes); NaN bitwise",
    }


class ReferenceSession:
    """A deterministic ORT CPU session over a ModelProto, bytes, or path.

    Keeps the temporary external-data spill directory (used for >2 GiB
    models) alive as long as the session exists.
    """

    def __init__(self, model):
        import onnxruntime as ort
        self._tmpdir = None
        if isinstance(model, (str, os.PathLike)):
            src = str(model)
        elif isinstance(model, bytes):
            src = model
        else:
            if model.ByteSize() >= _PROTO_SPILL_BYTES:
                self._tmpdir = tempfile.TemporaryDirectory(prefix="onnxopt_big_")
                path = os.path.join(self._tmpdir.name, "model.onnx")
                spill = onnx.ModelProto()
                spill.CopyFrom(model)
                onnx.save_model(spill, path, save_as_external_data=True,
                                all_tensors_to_one_file=True,
                                location="model.onnx.data", size_threshold=1024)
                src = path
            else:
                src = model.SerializeToString()
        self.session = ort.InferenceSession(src, sess_options=default_session_options(),
                                            providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def run(self, feed):
        """Run with the subset of `feed` this model declares; outputs by name."""
        sub = {k: v for k, v in feed.items() if k in self.input_names}
        missing = [k for k in self.input_names if k not in sub]
        if missing:
            raise KeyError("feed is missing required inputs: %s" % ", ".join(missing))
        results = self.session.run(None, sub)
        return {name: np.ascontiguousarray(arr) for name, arr in zip(self.output_names, results)}


def runtime_input_specs(model):
    """[(name, np dtype, dims)] for every runtime input (graph inputs minus
    initializers); dims entries are int, or a str dim_param, or None."""
    init_names = {t.name for t in model.graph.initializer}
    specs = []
    for vi in model.graph.input:
        if vi.name in init_names:
            continue
        if not vi.type.HasField("tensor_type"):
            raise ValueError("input '%s' is not a tensor; cannot generate inputs" % vi.name)
        tt = vi.type.tensor_type
        np_dtype = _NP_DTYPE.get(tt.elem_type)
        if np_dtype is None:
            raise ValueError("input '%s' has dtype %s outside the generator's support"
                             % (vi.name, TensorProto.DataType.Name(tt.elem_type)))
        dims = []
        for d in tt.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            elif d.HasField("dim_param") and d.dim_param:
                dims.append(d.dim_param)
            else:
                dims.append(None)
        specs.append((vi.name, np.dtype(np_dtype), dims))
    return specs


def resolve_shapes(specs, dyn_size):
    """Concrete shape per input, giving every dynamic dim `dyn_size`; dims that
    share a dim_param name share the size."""
    shapes = {}
    for name, _, dims in specs:
        shape = []
        for d in dims:
            shape.append(d if isinstance(d, int) else dyn_size)
        shapes[name] = tuple(shape)
    return shapes


def _random_array(rng, dtype, shape):
    if dtype.kind == "f":
        return rng.standard_normal(shape).astype(dtype)
    if dtype == np.bool_:
        return rng.integers(0, 2, size=shape).astype(np.bool_)
    if dtype == np.uint8:
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    # Signed/unsigned integer inputs are almost always ids/indices/masks:
    # stay in {0, 1} so any downstream Gather/embedding stays in bounds.
    return rng.integers(0, 2, size=shape).astype(dtype)


def random_feed(specs, rng, dyn_size):
    shapes = resolve_shapes(specs, dyn_size)
    return {name: _random_array(rng, dtype, shapes[name]) for name, dtype, _ in specs}


def battery_feeds(specs, rng, dyn_size):
    """[(kind, feed)] edge-case batteries. Float inputs get the pattern; other
    dtypes keep a safe base (zeros/ones where meaningful, else random)."""
    shapes = resolve_shapes(specs, dyn_size)
    base = random_feed(specs, rng, dyn_size)
    out = []

    def build(kind, fill):
        feed = {}
        for name, dtype, _ in specs:
            shape = shapes[name]
            if dtype.kind == "f":
                feed[name] = fill(dtype, shape)
            elif kind == "zeros":
                feed[name] = np.zeros(shape, dtype=dtype)
            elif kind == "ones":
                feed[name] = np.ones(shape, dtype=dtype)
            else:
                feed[name] = base[name]
        out.append((kind, feed))

    build("zeros", lambda dt, sh: np.zeros(sh, dtype=dt))
    build("ones", lambda dt, sh: np.ones(sh, dtype=dt))
    build("negative", lambda dt, sh: (-np.abs(rng.standard_normal(sh)) - 0.5).astype(dt))
    build("large-magnitude", lambda dt, sh: (rng.standard_normal(sh).astype(dt)
                                             * dt.type(_FLOAT_BIG[dt])))

    def naninf(dt, sh):
        arr = rng.standard_normal(sh).astype(dt)
        flat = arr.reshape(-1)
        if flat.size:
            flat[0::5] = dt.type(np.nan)
            if flat.size > 1:
                flat[1::7] = dt.type(np.inf)
            if flat.size > 2:
                flat[2::11] = dt.type(-np.inf)
        return arr

    if any(dtype.kind == "f" for _, dtype, _ in specs):
        build("nan-inf", naninf)
    return out


def _ulp_distance(a, b):
    """Element-wise ULP distance between same-dtype float arrays (int64)."""
    bits = a.dtype.itemsize * 8
    itype = {16: np.int16, 32: np.int32, 64: np.int64}[bits]
    ai = a.view(itype).astype(np.int64)
    bi = b.view(itype).astype(np.int64)
    lo = -(np.int64(1) << np.int64(bits - 1))
    ai = np.where(ai >= 0, ai, lo - ai)
    bi = np.where(bi >= 0, bi, lo - bi)
    return np.abs(ai - bi)


def compare_arrays(name, ref, got):
    """None when byte-identical, else a mismatch record."""
    if ref.dtype != got.dtype:
        return {"output": name, "reason": "dtype",
                "detail": "%s vs %s" % (ref.dtype, got.dtype)}
    if ref.shape != got.shape:
        return {"output": name, "reason": "shape",
                "detail": "%s vs %s" % (list(ref.shape), list(got.shape))}
    ref = np.ascontiguousarray(ref)
    got = np.ascontiguousarray(got)
    if ref.tobytes() == got.tobytes():
        return None
    if ref.dtype.kind == "f":
        bits = ref.dtype.itemsize * 8
        utype = {16: np.uint16, 32: np.uint32, 64: np.uint64}[bits]
        neq = ref.view(utype).reshape(-1) != got.view(utype).reshape(-1)
    else:
        neq = ref.reshape(-1) != got.reshape(-1)
    where = np.flatnonzero(neq)
    first = int(where[0]) if where.size else 0
    rec = {
        "output": name, "reason": "bytes",
        "mismatched_elements": int(where.size),
        "total_elements": int(ref.size),
        "first_index": [int(i) for i in np.unravel_index(first, ref.shape)] if ref.ndim else [],
        "a_value": repr(ref.reshape(-1)[first].item()),
        "b_value": repr(got.reshape(-1)[first].item()),
    }
    if ref.dtype.kind == "f":
        ulp = _ulp_distance(ref.reshape(-1), got.reshape(-1))
        rec["first_ulp"] = int(ulp[first])
        rec["max_ulp"] = int(ulp.max())
        rec["max_abs_diff"] = float(np.nanmax(np.abs(ref.astype(np.float64)
                                                     - got.astype(np.float64))))
    return rec


def compare_runs(ref_outputs, got_outputs):
    """List of mismatch records across two name->array output dicts."""
    mismatches = []
    ref_names, got_names = set(ref_outputs), set(got_outputs)
    for name in sorted(ref_names - got_names):
        mismatches.append({"output": name, "reason": "missing-output", "detail": "absent in candidate"})
    for name in sorted(got_names - ref_names):
        mismatches.append({"output": name, "reason": "extra-output", "detail": "absent in reference"})
    for name in sorted(ref_names & got_names):
        rec = compare_arrays(name, ref_outputs[name], got_outputs[name])
        if rec:
            mismatches.append(rec)
    return mismatches


class Verifier:
    """Precomputes reference outputs for a battery of input cases, then checks
    candidate models against them byte-for-byte.

    Building the verifier runs the reference model once per case (and one extra
    determinism self-check); each `check()` then costs one candidate session
    plus one run per case.
    """

    def __init__(self, reference_model, n_random=8, seed=0, dyn_sizes=(1, 3), batteries=True):
        self.config = dict(reference_config())
        self.config.update({"seed": seed, "random_samples": n_random,
                            "dynamic_dim_sizes": list(dyn_sizes), "batteries": bool(batteries)})
        self.skipped = []  # cases the reference itself cannot run
        self.cases = []    # (label, feed, ref_outputs)
        session = ReferenceSession(reference_model)
        specs = runtime_input_specs(reference_model if isinstance(reference_model, onnx.ModelProto)
                                    else onnx.load(reference_model))
        rng = np.random.default_rng(seed)
        planned = []
        for k in range(n_random):
            size = dyn_sizes[k % len(dyn_sizes)]
            planned.append(("random[%d] dyn=%d" % (k, size), random_feed(specs, rng, size), True))
        if batteries:
            size = dyn_sizes[-1]
            for kind, feed in battery_feeds(specs, rng, size):
                planned.append(("battery:%s dyn=%d" % (kind, size), feed, False))
        for label, feed, hard in planned:
            try:
                outputs = session.run(feed)
            except Exception as e:
                if hard:
                    raise RuntimeError("reference model failed on %s: %s" % (label, e)) from e
                self.skipped.append({"case": label, "reason": str(e).splitlines()[0][:300]})
                continue
            self.cases.append((label, feed, outputs))
        if not self.cases:
            raise RuntimeError("no runnable verification cases for the reference model")
        # Determinism self-check: the same session and feed must reproduce
        # byte-identical outputs, or byte-gating is meaningless in this env.
        label, feed, outputs = self.cases[0]
        again = session.run(feed)
        if compare_runs(outputs, again):
            raise RuntimeError("reference runtime is not deterministic on this host "
                               "(case %s differs on a re-run)" % label)

    def check(self, candidate_model):
        """{"ok": bool, "cases": [...], "skipped": [...], "config": {...}}."""
        result = {"ok": True, "cases": [], "skipped": list(self.skipped), "config": self.config}
        try:
            session = ReferenceSession(candidate_model)
        except Exception as e:
            result["ok"] = False
            result["error"] = "candidate model failed to load: %s" % str(e).splitlines()[0][:300]
            return result
        for label, feed, ref_outputs in self.cases:
            try:
                got = session.run(feed)
                mismatches = compare_runs(ref_outputs, got)
            except Exception as e:
                mismatches = [{"output": "*", "reason": "run-error",
                               "detail": str(e).splitlines()[0][:300]}]
            result["cases"].append({"case": label, "ok": not mismatches, "mismatches": mismatches})
            if mismatches:
                result["ok"] = False
        return result


def verify_models(model_a, model_b, n_random=8, seed=0, dyn_sizes=(1, 3), batteries=True):
    """Bit-exact check of model_b against model_a (paths or ModelProtos)."""
    if isinstance(model_a, (str, os.PathLike)):
        model_a = onnx.load(str(model_a))
    verifier = Verifier(model_a, n_random=n_random, seed=seed,
                        dyn_sizes=dyn_sizes, batteries=batteries)
    return verifier.check(model_b)


def render_result(result):
    lines = []
    for case in result["cases"]:
        status = "PASS" if case["ok"] else "FAIL"
        lines.append("  %-4s %s" % (status, case["case"]))
        for mm in case["mismatches"][:4]:
            if mm["reason"] == "bytes":
                lines.append("       %s: %d/%d elements differ; first at %s: %s vs %s%s"
                             % (mm["output"], mm["mismatched_elements"], mm["total_elements"],
                                mm["first_index"], mm["a_value"], mm["b_value"],
                                " (ULP %d, max ULP %d)" % (mm["first_ulp"], mm["max_ulp"])
                                if "first_ulp" in mm else ""))
            else:
                lines.append("       %s: %s (%s)" % (mm["output"], mm["reason"],
                                                     mm.get("detail", "")))
    for sk in result["skipped"]:
        lines.append("  SKIP %s -- reference cannot run it (%s)" % (sk["case"], sk["reason"]))
    if result.get("error"):
        lines.append("  ERROR %s" % result["error"])
    lines.append("verdict: %s" % ("BIT-EXACT" if result["ok"] else "NOT bit-exact"))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Bit-exact ONNX model equivalence check (deterministic ORT CPU reference)")
    ap.add_argument("model_a", help="reference model (.onnx)")
    ap.add_argument("model_b", help="candidate model (.onnx)")
    ap.add_argument("--samples", type=int, default=8, help="random input cases (default 8)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    ap.add_argument("--dyn-sizes", default="1,3",
                    help="comma-separated sizes tried for symbolic dims (default 1,3)")
    ap.add_argument("--no-batteries", action="store_true",
                    help="skip the zeros/ones/negative/large/nan-inf edge batteries")
    ap.add_argument("--json", metavar="PATH", help="write the full result as JSON")
    args = ap.parse_args(argv)

    try:
        dyn_sizes = tuple(int(x) for x in args.dyn_sizes.split(",") if x)
    except ValueError:
        ap.error("--dyn-sizes must be comma-separated integers")
    try:
        result = verify_models(args.model_a, args.model_b, n_random=args.samples,
                               seed=args.seed, dyn_sizes=dyn_sizes,
                               batteries=not args.no_batteries)
    except (OSError, RuntimeError, ValueError) as e:
        sys.stderr.write("error: %s\n" % e)
        return 2
    print("== bit-exact verify: %s vs %s ==" % (os.path.basename(args.model_a),
                                                os.path.basename(args.model_b)))
    print(render_result(result))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        sys.stderr.write("wrote %s\n" % args.json)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
