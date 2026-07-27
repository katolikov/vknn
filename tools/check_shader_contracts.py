#!/usr/bin/env python3
"""Build/CI lint: the compute shaders obey the 960-driver correctness contracts (ADR-0011).

The shader tree carries a few invariants that are correctness — not style — but nothing in the build
enforces them; a violation only shows up as a silent accuracy collapse or nondeterminism on device.
This lint reads shaders/*.comp and reports violations, exiting non-zero on a FATAL one so a broken
contract fails CI instead of a device run.

FATAL (correctness):
  1. store16 on fp16 stores. Every shader that stores fp16 (an f16vec*/float16_t writeable buffer,
     packHalf2x16, or the precision.glsl STORE framework) must #include "store16.glsl" (directly or
     via precision.glsl). Missing it lets some mobile drivers round fp16 stores toward zero, biasing
     every activation ~half a ULP smaller — SNR collapses while cosine stays ~1 (ADR-0011; the whole
     reason store16.glsl exists).
  2. Lane-count agreement across an fp32/fp16 pair. `X.comp` and `X_fp16.comp` are the same kernel
     at two storage precisions; the host op computes ONE dispatch lane count and selects between
     them on device fp16 support alone. If their grid-bound expressions disagree, one of the two
     reads a grid that was never launched: the shipped precision silently writes a fraction of its
     output and leaves the rest at whatever the buffer held. (Seen: a per-pixel lane map applied to
     the fp32 kernel and its host dispatch but not to the fp16 twin, which still decoded a channel
     block from the lane index -- every channel block above the first went unwritten, on the default
     precision, with no error anywhere.)

  3. VKNN_NO_RTE ordering. A shader that opts out of the RoundingModeRTE execution mode must
     #define VKNN_NO_RTE BEFORE it includes store16.glsl/precision.glsl; defined after the include it
     is a silent no-op and the execution mode the kernel meant to avoid gets compiled in anyway.

ADVISORY (reported, non-fatal — flags with --strict):
  4. VKNN_NO_RTE on the GEMM/elementwise family. Kernels in that family that store fp16 avoid the
     float-controls execution mode (miscompiled by some drivers -> nondeterminism/faults under load,
     and both float16_t(x) and packHalf2x16 round toward zero there) by defining VKNN_NO_RTE and
     rounding through the integer-exact vknnRte16. A family fp16 kernel without it is suspect.
  5. 2-term gid recovery. A 1-D-dispatched kernel (local_size_x only) that reads
     gl_GlobalInvocationID.x as a flat index must recover the global id as
     `gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x`;
     a bare .x drops every element past the 65535 workgroup cap once the dispatch is 2D-split.

Usage:
  tools/check_shader_contracts.py [--shaders DIR] [--strict]

  --strict   promote the advisory checks (4, 5) to failures as well.
"""
import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHADERS_DEFAULT = os.path.join(os.path.dirname(HERE), "shaders")

# GEMM / elementwise family by stem prefix — the kernels ADR-0011 keeps off the RoundingModeRTE
# execution mode (their fp16 stores round through vknnRte16 to match the fused-unit per-step rounding).
_FAMILY_RE = re.compile(
    r'^(matmul|gemm|fc|conv_gemm|unary|binary|add|clip|prelu|flat_binary'
    r'|quantize_linear|dequantize_linear)')
_BUFFER_ELEM_RE = re.compile(r'\bbuffer\s+\w+\s*\{\s*([A-Za-z0-9_]+)')
_STORE16_INCLUDE_RE = re.compile(r'#include\s+"(store16|precision)\.glsl"')
_PRECISION_INCLUDE_RE = re.compile(r'#include\s+"precision\.glsl"')
_NO_RTE_DEFINE_RE = re.compile(r'#define\s+VKNN_NO_RTE')
_GID_RECOVERY_RE = re.compile(
    r'gl_GlobalInvocationID\.x\s*\+\s*gl_GlobalInvocationID\.y\s*\*\s*'
    r'gl_NumWorkGroups\.x\s*\*\s*gl_WorkGroupSize\.x')


# The grid bound a kernel guards itself with: `if (gid >= uint(EXPR)) return;`. EXPR is the lane
# count the host must dispatch, so it is the one thing an fp32/fp16 pair can never disagree on.
_GRID_BOUND_RE = re.compile(r">=\s*uint\s*\((.+?)\)\s*\)\s*return", re.S)


def _grid_bound(src):
    """The kernel's dispatch-lane-count expression, whitespace-normalized, or None."""
    m = _GRID_BOUND_RE.search(src)
    return re.sub(r"\s+", "", m.group(1)) if m else None


def check_precision_pairs(sources):
    """FATAL: X.comp and X_fp16.comp must guard against the same lane count."""
    fatals = []
    for name, src in sorted(sources.items()):
        if not name.endswith("_fp16.comp"):
            continue
        base = name[: -len("_fp16.comp")] + ".comp"
        if base not in sources:
            continue
        a, b = _grid_bound(sources[base]), _grid_bound(src)
        if a is not None and b is not None and a != b:
            fatals.append("%s and %s disagree on the dispatch lane count (%s vs %s); the host op "
                          "launches one grid for both, so the other reads lanes that were never "
                          "launched and leaves part of its output unwritten" % (base, name, a, b))
    return fatals


def _stores_fp16(src):
    """True when the shader writes fp16: an f16 writeable buffer element, packHalf2x16, or the
    precision.glsl STORE framework (which compiles an fp16 variant under -DUSE_FP16)."""
    for elem in _BUFFER_ELEM_RE.findall(src):
        if elem.startswith("f16vec") or elem == "float16_t":
            return True
    if "packHalf2x16" in src:
        return True
    if _PRECISION_INCLUDE_RE.search(src):
        return True
    return False


def _local_size(src):
    """(x, y, z) local sizes as ints (default 1 when a dimension is unspecified)."""
    def dim(axis):
        m = re.search(r'local_size_%s\s*=\s*(\d+)' % axis, src)
        return int(m.group(1)) if m else 1
    return dim("x"), dim("y"), dim("z")


def check_shader(name, src):
    """Return (fatals, advisories) — lists of message strings for this one shader."""
    fatals, advisories = [], []

    inc_m = _STORE16_INCLUDE_RE.search(src)
    has_store16 = inc_m is not None

    # 1. fp16 stores need the store16/precision rounding framework.
    if _stores_fp16(src) and not has_store16:
        fatals.append("%s: stores fp16 but does not #include store16.glsl or precision.glsl "
                      "(fp16 stores may round toward zero -> SNR collapse)" % name)

    # 2. VKNN_NO_RTE must be defined before the store16/precision include, or it is a no-op.
    nortem = _NO_RTE_DEFINE_RE.search(src)
    if nortem and has_store16 and nortem.start() > inc_m.start():
        fatals.append("%s: #define VKNN_NO_RTE appears after the store16/precision include; "
                      "defined too late it is a no-op and the RTE execution mode is compiled in "
                      "anyway" % name)

    stem = os.path.splitext(name)[0]

    # 4. GEMM/elementwise-family fp16 kernels should carry VKNN_NO_RTE (advisory).
    if _FAMILY_RE.match(stem) and _stores_fp16(src) and not nortem:
        advisories.append("%s: GEMM/elementwise-family kernel stores fp16 but does not #define "
                          "VKNN_NO_RTE (ADR-0011 keeps this family off the RTE execution mode)" % name)

    # 5. 1-D flat kernels need the 2-term gid recovery (advisory).
    lx, ly, lz = _local_size(src)
    one_dim = ly == 1 and lz == 1
    uses_x = "gl_GlobalInvocationID.x" in src
    uses_yz = ("gl_GlobalInvocationID.y" in src) or ("gl_GlobalInvocationID.z" in src)
    # Only a genuinely 1-D kernel that indexes by .x alone (never touches .y/.z as spatial dims) is
    # subject to the 2D-split cap; a kernel using .y/.z runs a real multi-dim grid.
    if one_dim and uses_x and not uses_yz and not _GID_RECOVERY_RE.search(src):
        advisories.append("%s: 1-D kernel indexes gl_GlobalInvocationID.x without the 2-term "
                          "recovery (drops elements past the 65535 dispatch cap when 2D-split)" % name)

    return fatals, advisories


def main():
    ap = argparse.ArgumentParser(description="VKNN shader-contract lint (ADR-0011)")
    ap.add_argument("--shaders", default=SHADERS_DEFAULT, help="shaders dir (default: this tree)")
    ap.add_argument("--strict", action="store_true", help="treat advisory checks as failures too")
    args = ap.parse_args()

    if not os.path.isdir(args.shaders):
        sys.exit("error: %s not found (pass --shaders)" % args.shaders)

    comps = sorted(glob.glob(os.path.join(args.shaders, "*.comp")))
    if not comps:
        sys.exit("error: no *.comp under %s" % args.shaders)

    sources = {}
    all_fatals, all_advisories = [], []
    for comp in comps:
        name = os.path.basename(comp)
        with open(comp, encoding="utf-8") as f:
            src = f.read()
        sources[name] = src
        f_, a_ = check_shader(name, src)
        all_fatals += f_
        all_advisories += a_
    all_fatals += check_precision_pairs(sources)

    print("check_shader_contracts: scanned %d shaders (%d fp32/fp16 pairs)"
          % (len(comps), sum(1 for n in sources if n.endswith("_fp16.comp")
                             and n[: -len("_fp16.comp")] + ".comp" in sources)))
    if all_advisories:
        print("ADVISORY:")
        for m in all_advisories:
            print("  - %s" % m)
    if all_fatals:
        print("FAIL — shader contract violations:")
        for m in all_fatals:
            print("  - %s" % m)
    if all_fatals or (args.strict and all_advisories):
        sys.exit(1)
    if not all_advisories:
        print("PASS — all shaders satisfy the store16 / pair-lane-count / VKNN_NO_RTE / gid-recovery contracts")
    else:
        print("PASS — no fatal contract violations (advisories above; --strict to enforce)")


if __name__ == "__main__":
    main()
