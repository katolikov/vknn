#!/usr/bin/env python3
"""Self-consistency check for VKNN's op-support tooling — no model, no device.

check_model_support.py and scan_unsupported_ops.py both derive "which ONNX op does VKNN
recognize" from src/core/op.cpp, by different parsers:

  * check_model_support.parse_op_map()  -- precise: matches the opTypeFromOnnx map entries,
    unaryFromOnnx / binaryFromOnnx entries, and the reduceFromOnnx string branch.
  * scan_unsupported_ops.supported_op_names()  -- coarse: every double-quoted identifier in the
    file, minus a hand-listed _INTERNAL_NAMES set.

If the two drift, one tool will claim an op is supported that the other flags as unknown. This
script cross-checks them and asserts:

  1. parse_op_map() yields a non-empty ONNX-name -> OpType map.
  2. every op name from the precise parser is in the coarse recognized set (subset invariant); a
     miss means _INTERNAL_NAMES over-trimmed or the parsers diverged.
  3. every OpType the precise parser maps to has a CPU-kernel registration
     (VKNN_REGISTER_CPU_OP under src/backend/cpu/ops) -- a recognized op with no CPU op cannot run
     even as a fallback. OpTypes with no GPU kernel are fine (they run on the CPU op), so the GPU
     registry is only reported, never required.

Exit 0 = consistent; non-zero = drift, with the offending names printed. Pure source analysis; it
reads the same files the engine ships, imports nothing beyond the two sibling tools, needs no onnx.

Usage:
  tools/check_support_consistency.py [--repo REPO]
"""
import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DEFAULT = os.path.dirname(HERE)

# OpTypes that legitimately have no CPU kernel file (structural / handled inline). Keep small and
# justified; each entry is an op the importer recognizes but the CPU backend never instantiates.
CPU_KERNEL_EXEMPT = {
    "kUnknown",  # the fallthrough sentinel, never a real op
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="VKNN op-support tooling self-consistency check")
    ap.add_argument("--repo", default=REPO_DEFAULT, help="VKNN repo root (default: this tree)")
    args = ap.parse_args()

    cms = _load("check_model_support", os.path.join(HERE, "check_model_support.py"))
    suo = _load("scan_unsupported_ops", os.path.join(HERE, "scan_unsupported_ops.py"))

    op_cpp = os.path.join(args.repo, "src", "core", "op.cpp")
    if not os.path.isfile(op_cpp):
        sys.exit("error: %s not found (pass --repo)" % op_cpp)

    name_to_type = cms.parse_op_map(op_cpp)          # precise ONNX name -> OpType
    recognized = suo.supported_op_names(op_cpp)      # coarse recognized set
    cpu_ops = cms.parse_registry(
        os.path.join(args.repo, "src", "backend", "cpu", "ops"), "VKNN_REGISTER_CPU_OP")
    vk_ops = cms.parse_registry(
        os.path.join(args.repo, "src", "backend", "vulkan", "ops"), "VKNN_REGISTER_VK_OP")

    problems = []

    # 1. non-empty map
    if not name_to_type:
        problems.append("parse_op_map() returned an empty map — op.cpp parse broke")

    # 2. precise names subset of coarse recognized set
    missing = sorted(n for n in name_to_type if n not in recognized)
    if missing:
        problems.append(
            "op names known to check_model_support but NOT to scan_unsupported_ops "
            "(_INTERNAL_NAMES drift?): %s" % ", ".join(missing))

    # 3. every mapped OpType has a CPU kernel
    if cpu_ops:
        no_cpu = sorted({t for t in name_to_type.values()
                         if t not in cpu_ops and t not in CPU_KERNEL_EXEMPT
                         # families are pseudo-types dispatched inside one kernel file
                         and t not in ("Unary", "Binary", "Reduce")})
        if no_cpu:
            problems.append(
                "OpType(s) mapped from ONNX with no VKNN_REGISTER_CPU_OP kernel: %s"
                % ", ".join(no_cpu))
    else:
        sys.stderr.write("warning: no CPU registry found under %s; kernel check skipped\n"
                         % args.repo)

    print("check_support_consistency: %d ONNX names -> %d OpTypes; "
          "%d CPU kernels, %d GPU kernels; recognized set %d names"
          % (len(name_to_type), len(set(name_to_type.values())),
             len(cpu_ops), len(vk_ops), len(recognized)))

    if problems:
        print("FAIL — support tooling drift:")
        for p in problems:
            print("  - %s" % p)
        sys.exit(1)
    print("PASS — check_model_support and scan_unsupported_ops agree on op.cpp")


if __name__ == "__main__":
    main()
