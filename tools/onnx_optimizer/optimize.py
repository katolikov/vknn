"""Bit-exact ONNX graph optimizer for VKNN -- CLI entry and engine.

Optimizes an ONNX model with structural, provably lossless passes, and
refuses to write the result unless the optimized model is BYTE-IDENTICAL to
the original on every verification input (deterministic onnxruntime CPU
reference; see verify.py). Two layers of gating:

  1. per-pass: the model is snapshotted before each pass; after a pass makes
     changes, a fast bit-exact check runs and a failure reverts exactly that
     pass (logged in the report, pass disabled for the rest of the run) --
     a buggy pass can never silently poison the output;
  2. final: the full battery (random samples across dynamic-dim sizes plus
     zeros/ones/negative/large/NaN-Inf edge cases) gates the output write.
     On failure nothing is written and the exit code is 1; --force overrides
     with a loud warning.

Float-algebra rewrites (Conv+BN folding & co.) are NOT bit-exact and only run
under --allow-lossy, which swaps the byte gate for a tolerance gate and
reports the observed max ULP / abs error.

Usage (from tools/, or with PYTHONPATH=tools, or as a plain script):
  python -m onnx_optimizer.optimize -i model.onnx -o model.opt.onnx \\
      --report report.json [--disable-pass NAME] [--only-pass NAME] \\
      [--verify-samples 16] [--seed 0] [--target vknn] [--allow-lossy] [--force]

Exit code: 0 = optimized + verified (or nothing to do), 1 = verification
failed, 2 = bad invocation / unreadable model.
"""
import argparse
import os
import sys

if __package__ in (None, ""):  # running as a plain script: put tools/ on the path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnx

from onnx_optimizer import __version__, graph_util as gu, report as report_mod
from onnx_optimizer.passes import PIPELINE, build_pipeline
from onnx_optimizer.verify import Verifier, compare_runs, reference_config

_PROTO_SPILL_BYTES = 1_900_000_000  # keep clear of the 2 GiB protobuf ceiling


def _copy(model):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    return m


def _refresh_shapes(model):
    """Best-effort shape inference; passes degrade gracefully without it."""
    try:
        return onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except Exception:
        return model


class Engine:
    """Fixpoint pass driver with per-pass bit-exact gating and auto-revert."""

    def __init__(self, model, passes, report, gate_samples=2, seed=0,
                 dyn_sizes=(1, 3), max_iterations=10, log=None):
        self.model = model
        self.passes = passes
        self.report = report
        self.max_iterations = max_iterations
        self.log = log or (lambda s: None)
        self.disabled = set()
        self.gate = Verifier(model, n_random=gate_samples, seed=seed,
                             dyn_sizes=dyn_sizes, batteries=False)

    def run(self):
        self.model = _refresh_shapes(self.model)
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            changed = False
            for p in self.passes:
                if p.name in self.disabled:
                    continue
                snapshot = self.model.SerializeToString()
                try:
                    n = p.run(self.model)
                except Exception as e:
                    self.model = onnx.load_from_string(snapshot)
                    self.disabled.add(p.name)
                    detail = "pass raised: %s" % str(e).splitlines()[0][:200]
                    self.report.log_pass(iteration, p.name, 0, "reverted", detail)
                    self.log("  REVERTED %s (%s)" % (p.name, detail))
                    continue
                if n == 0:
                    continue
                try:
                    gu.toposort(self.model.graph)
                    self.model = _refresh_shapes(self.model)
                    gate = self.gate.check(self.model)
                except Exception as e:
                    gate = {"ok": False, "cases": [],
                            "error": str(e).splitlines()[0][:200]}
                if gate["ok"]:
                    changed = True
                    self.report.log_pass(iteration, p.name, n, "applied")
                    self.log("  iter %d: %-26s %d change(s)" % (iteration, p.name, n))
                else:
                    self.model = onnx.load_from_string(snapshot)
                    self.disabled.add(p.name)
                    detail = gate.get("error") or _first_gate_failure(gate)
                    self.report.log_pass(iteration, p.name, n, "reverted", detail)
                    self.log("  REVERTED %s (fast bit-exact gate failed: %s)" % (p.name, detail))
            if not changed:
                break
        self.report.iterations = iteration
        gu.toposort(self.model.graph)
        self.model = _refresh_shapes(self.model)
        return self.model


def _first_gate_failure(gate):
    for case in gate.get("cases", []):
        for mm in case.get("mismatches", []):
            return "case %s, output %s: %s" % (case["case"], mm.get("output"), mm.get("reason"))
    return "mismatch"


def _lossy_tolerance_summary(verification):
    """Max ULP / abs diff across all byte mismatches (lossy-mode reporting)."""
    max_ulp, max_abs = 0, 0.0
    for case in verification["cases"]:
        for mm in case["mismatches"]:
            if mm["reason"] != "bytes":
                return None  # structural mismatch: not a tolerance question
            max_ulp = max(max_ulp, mm.get("max_ulp", 0))
            max_abs = max(max_abs, mm.get("max_abs_diff", float("inf")))
    return {"max_ulp": max_ulp, "max_abs_diff": max_abs}


def _save(model, path):
    if model.ByteSize() >= _PROTO_SPILL_BYTES:
        onnx.save_model(model, path, save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        location=os.path.basename(path) + ".data", size_threshold=1024)
    else:
        onnx.save_model(model, path)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Bit-exact ONNX graph optimizer for VKNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Passes: " + ", ".join(cls.name + (" (lossy)" if cls.lossy else "")
                                      for cls in PIPELINE))
    ap.add_argument("-i", "--input", metavar="MODEL", help="input .onnx model")
    ap.add_argument("-o", "--output", metavar="MODEL", help="output .onnx model")
    ap.add_argument("--report", metavar="PATH", help="write the JSON report here")
    ap.add_argument("--disable-pass", action="append", default=[], metavar="NAME",
                    help="disable a pass (repeatable)")
    ap.add_argument("--only-pass", action="append", default=[], metavar="NAME",
                    help="run only the named pass(es) (repeatable)")
    ap.add_argument("--verify-samples", type=int, default=16,
                    help="random inputs for the final verification (default 16)")
    ap.add_argument("--gate-samples", type=int, default=2,
                    help="random inputs for the fast per-pass gate (default 2)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    ap.add_argument("--dyn-sizes", default="1,3",
                    help="comma-separated sizes tried for symbolic dims (default 1,3)")
    ap.add_argument("--max-iterations", type=int, default=10,
                    help="fixpoint iteration cap (default 10)")
    ap.add_argument("--max-fold-bytes", type=int, default=16 * 1024 * 1024,
                    help="constant-folding per-node output size cap (default 16 MiB)")
    ap.add_argument("--target", choices=["vknn"],
                    help="add a target-specific importer-compatibility report")
    ap.add_argument("--allow-lossy", action="store_true",
                    help="enable float-algebra passes; NOT bit-exact (tolerance-gated)")
    ap.add_argument("--force", action="store_true",
                    help="write the output even if verification FAILS (dangerous)")
    ap.add_argument("--list-passes", action="store_true", help="list passes and exit")
    args = ap.parse_args(argv)

    if args.list_passes:
        for cls in PIPELINE:
            print("%-26s %s%s" % (cls.name, "[LOSSY, --allow-lossy only] " if cls.lossy else "",
                                  cls.description))
        return 0
    if not args.input or not args.output:
        ap.error("-i/--input and -o/--output are required")
    try:
        dyn_sizes = tuple(int(x) for x in args.dyn_sizes.split(",") if x)
    except ValueError:
        ap.error("--dyn-sizes must be comma-separated integers")

    try:
        model = onnx.load(args.input)
    except Exception as e:
        sys.stderr.write("error: failed to load %s: %s\n" % (args.input, e))
        return 2
    try:
        passes = build_pipeline(only=args.only_pass or None, disable=args.disable_pass or None,
                                allow_lossy=args.allow_lossy, max_fold_bytes=args.max_fold_bytes)
    except ValueError as e:
        sys.stderr.write("error: %s\n" % e)
        return 2

    lossy_active = any(p.lossy for p in passes)
    report = report_mod.Report()
    report.tool = {"name": "onnx_optimizer", "version": __version__,
                   "reference": reference_config(),
                   "passes_enabled": [p.name for p in passes],
                   "allow_lossy": args.allow_lossy, "seed": args.seed}
    report.before = report_mod.model_stats(model, os.path.abspath(args.input))
    original = _copy(model)

    log = lambda s: print(s)  # noqa: E731
    print("optimizing %s (%d nodes)..." % (args.input, report.before["nodes"]))
    try:
        engine = Engine(model, passes, report, gate_samples=args.gate_samples,
                        seed=args.seed, dyn_sizes=dyn_sizes,
                        max_iterations=args.max_iterations, log=log)
    except RuntimeError as e:
        sys.stderr.write("error: cannot build the verification gate: %s\n" % e)
        return 2
    optimized = engine.run()

    try:
        onnx.checker.check_model(optimized)
    except Exception as e:
        sys.stderr.write("error: optimized model fails onnx.checker: %s\n"
                         % str(e).splitlines()[0])
        if not args.force:
            sys.stderr.write("refusing to write %s\n" % args.output)
            return 1

    print("final verification (%d random samples + edge batteries)..." % args.verify_samples)
    verifier = Verifier(original, n_random=args.verify_samples, seed=args.seed,
                        dyn_sizes=dyn_sizes, batteries=True)
    verification = verifier.check(optimized)
    report.verification = verification
    report.after = report_mod.model_stats(optimized, os.path.abspath(args.output))

    verified = verification["ok"]
    if lossy_active and not verified:
        report.lossy = _lossy_tolerance_summary(verification)
        if report.lossy is not None:
            verified = True  # lossy mode: tolerance-gated, loudly reported
            print("WARNING: --allow-lossy active; output is NOT bit-exact "
                  "(max ULP %d, max abs diff %g)"
                  % (report.lossy["max_ulp"], report.lossy["max_abs_diff"]))
    elif lossy_active:
        report.lossy = {"max_ulp": 0, "max_abs_diff": 0.0}

    if args.target == "vknn":
        from onnx_optimizer import vknn_target
        report.vknn = vknn_target.analyze(optimized)

    if args.report:
        report.write_json(args.report)
    print(report.render_text())

    if not verified and not args.force:
        sys.stderr.write("\nVERIFICATION FAILED -- %s was NOT written. "
                         "(--force overrides; do not use it lightly.)\n" % args.output)
        return 1
    if not verified:
        sys.stderr.write("\n" + "!" * 72 + "\n"
                         "!! WARNING: verification FAILED but --force was given.\n"
                         "!! The written model is NOT equivalent to the input.\n"
                         + "!" * 72 + "\n")
    _save(optimized, args.output)
    print("wrote %s" % args.output)
    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())
