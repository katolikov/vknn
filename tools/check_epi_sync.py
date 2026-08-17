#!/usr/bin/env python3
"""Build/CI check: the pointwise-epilogue machinery is a multi-surface hand sync.

A producer kernel can host a fused pointwise chain at its store (fusePointwiseChains attaches the
chain to the producer instead of leaving a standalone FusedPointwise node). This spans three
independently edited surfaces that must agree, or a mismatch surfaces only as a device-load-time
throw (env.pipeline -> "shader not found: <stem>_epi...", vk_pipeline.cpp) invisible to host builds:

  (a) shader sources that #include "pw_epilogue.glsl"
        -> the _epi/_epi_rx SPIR-V variants CMake generates (the derived stem set; see CMakeLists.txt).
  (b) the op-file kernel-name call sites  shader((std::string("<stem>") + epi.suffix())...)
        -> the stems the C++ actually requests at runtime.
  (c) pwEpilogueCapable(OpType)  (src/import/fuse_pointwise_chains.cpp)
        -> the producer OpTypes the fusion pass will attach a unit to.

This tool re-derives all three from source and FATAL-ERRORs (exit 1) when they disagree, flipping the
failure mode from a device-load crash to a configure/CI diagnostic. On agreement it prints a summary
and exits 0. Pure source analysis: no build, no device, no onnx.

Checks:
  1. every op-file-requested stem has a shader that #includes pw_epilogue.glsl (missing -> load crash).
  2. every shader-derived stem is requested by some op file (extra -> dead SPIR-V embedded in every
     binary + embeddedShadersHash() churn that invalidates device caches).
  3. every pwEpilogueCapable OpType resolves (via the VKNN_REGISTER_VK_OP registry) to an op file that
     requests at least one epi stem (added to the capable set with no kernel able to host it).

Usage:
  tools/check_epi_sync.py [--repo REPO]
"""
import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DEFAULT = os.path.dirname(HERE)

# The include DIRECTIVE, not an incidental comment mention — matches CMakeLists.txt's derivation.
_EPI_INCLUDE_RE = re.compile(r'#include[ \t]+"pw_epilogue\.glsl"')
# A string literal immediately followed by  + epi.suffix()  (the common site and each ternary arm).
_STEM_SITE_RE = re.compile(r'"([A-Za-z0-9_]+)"\s*\)?\s*\+\s*epi\.suffix\(\)')
# matmul.cpp composes the stem in a variable then does  name += epi.suffix();  — trace the literals.
_NAME_SUFFIX_RE = re.compile(r'\b(\w+)\s*\+=\s*epi\.suffix\(\)')
_REGISTER_VK_RE = re.compile(r'VKNN_REGISTER_VK_OP\(\s*OpType::(\w+)')
_PW_CAPABLE_CASE_RE = re.compile(r'case\s+OpType::(\w+)\s*:')
# Data-driven form: op_descriptor.cpp's  set(OpType::X, <layout>, <pwMember>, <pwEpilogue>)  — the
# 4th argument is the epilogue flag. Kept in parallel with the switch form so the check survives the
# op-descriptor refactor without an edit.
_PW_CAPABLE_SET_RE = re.compile(
    r'set\(\s*OpType::(\w+)\s*,[^,]+,\s*(?:true|false)\s*,\s*(true|false)\s*\)')


def derive_shader_stems(shader_dir):
    """Sniff every shaders/*.comp for the epilogue include; return (epi_stems, rx_standalone).

    Mirrors the CMake derivation exactly: fused_pw_* are the standalone-VM _rx pair (never _epi);
    a hand-written <stem>_fp16.comp contributes the stem with the _fp16 suffix stripped.
    """
    epi_stems, rx_standalone = set(), set()
    for comp in sorted(glob.glob(os.path.join(shader_dir, "*.comp"))):
        name = os.path.splitext(os.path.basename(comp))[0]
        with open(comp, encoding="utf-8") as f:
            src = f.read()
        if not _EPI_INCLUDE_RE.search(src):
            continue
        if name.startswith("fused_pw_"):
            rx_standalone.add(name)
        elif name.endswith("_fp16"):
            epi_stems.add(name[: -len("_fp16")])
        else:
            epi_stems.add(name)
    return epi_stems, rx_standalone


def _requested_stems_in_text(text):
    """Extract every stem the C++ requests via '<stem>' + epi.suffix() in one file's text.

    Handles the common single-literal site and both arms of the ternary sites
    (std::string(cond ? "a" : "b") + epi.suffix()). The composed-variable form (matmul's
    name += epi.suffix()) is resolved separately by _matmul_composed_stems.
    """
    stems = set(_STEM_SITE_RE.findall(text))
    # Ternary arms: "b" + epi.suffix() catches the false arm; the true arm "a" precedes the '?'.
    ternary = re.compile(
        r'\?\s*"([A-Za-z0-9_]+)"\s*:\s*"([A-Za-z0-9_]+)"\s*\)\s*\+\s*epi\.suffix\(\)')
    for m in ternary.finditer(text):
        stems.add(m.group(1))
        stems.add(m.group(2))
    return stems


def _matmul_composed_stems(text):
    """Resolve the matmul.cpp composed-name site: base = <ternary of literals>; [name += "_bias";]
    name += epi.suffix(). Returns every base literal plus, when the file composes a bias twin
    (name += "_bias"), the _bias form of each base that does not already carry it.
    """
    if not _NAME_SUFFIX_RE.search(text):
        return set()
    stems = set()
    # base literals: the string literals assigned into the composed variable's ternary(ies).
    for m in re.finditer(r'=\s*[^;]*\?[^;]*;', text):
        seg = m.group(0)
        if "epi.suffix()" in seg:  # not the assignment we want
            continue
        stems.update(re.findall(r'"([A-Za-z0-9_]+)"', seg))
    # the _bias composition (name += "_bias") multiplies each base by a _bias twin, but never
    # doubles a suffix on a literal that already spells the bias form.
    if re.search(r'\+=\s*"_bias"', text):
        stems |= {s + "_bias" for s in list(stems) if not s.endswith("_bias")}
    return stems


def requested_stems(ops_dir, available=frozenset()):
    """Union of every epi stem the op-file kernel-name sites request, over src/backend/vulkan/ops."""
    stems = set()
    files = sorted(glob.glob(os.path.join(ops_dir, "*.cpp"))
                   + glob.glob(os.path.join(ops_dir, "*.h")))
    for path in files:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if "epi.suffix()" not in text:
            continue
        stems |= _requested_stems_in_text(text)
        stems |= _matmul_composed_stems(text)
    # An fp16-accumulator twin (shaders/gen_acc16.py) is selected by precision, not by a name literal:
    # the op file requests the base stem and the kernel picker appends _acc16 at Precision::Low. A
    # requested stem therefore also requests its twin -- but only for the stems that actually have one,
    # so a missing twin still reports as drift rather than being excused.
    stems |= {stem + "_acc16" for stem in stems if stem + "_acc16" in available}
    return stems


def parse_pw_capable(pass_cpp, descriptor_cpp):
    """The pwEpilogueCapable OpType set, from whichever form the tree carries.

    Two forms are supported so the check survives the op-descriptor refactor:
      * the switch in pwEpilogueCapable() (fuse_pointwise_chains.cpp): the case labels above
        `return true`;
      * the data-driven descriptor table (op_descriptor.cpp): the set(OpType::X, ...) rows whose
        4th argument (pwEpilogue) is true.
    """
    with open(pass_cpp, encoding="utf-8") as f:
        pass_text = f.read()
    m = re.search(r'pwEpilogueCapable\s*\([^)]*\)\s*\{', pass_text)
    if not m:
        sys.exit("error: pwEpilogueCapable() not found in %s (parser out of date)" % pass_cpp)
    # Body from the opening brace to the matching close of the function.
    start = m.end() - 1
    depth, i = 0, start
    while i < len(pass_text):
        if pass_text[i] == "{":
            depth += 1
        elif pass_text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    body = pass_text[start:i]

    ret_true = body.find("return true")
    if ret_true >= 0:
        # Switch form: the case labels that fall through to `return true`.
        return set(_PW_CAPABLE_CASE_RE.findall(body[:ret_true]))

    # Data-driven form: the switch was replaced by a descriptor-table lookup; read the table.
    if descriptor_cpp and os.path.isfile(descriptor_cpp):
        with open(descriptor_cpp, encoding="utf-8") as f:
            desc_text = f.read()
        caps = {op for op, flag in _PW_CAPABLE_SET_RE.findall(desc_text) if flag == "true"}
        if caps:
            return caps
    sys.exit("error: pwEpilogueCapable() has no 'return true' path and no descriptor table found "
             "(parser out of date)")


def parse_vk_registry(ops_dir):
    """OpType -> set of op files that register it (VKNN_REGISTER_VK_OP)."""
    reg = {}
    for path in sorted(glob.glob(os.path.join(ops_dir, "*.cpp"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for op in _REGISTER_VK_RE.findall(text):
            reg.setdefault(op, set()).add(os.path.basename(path))
    return reg


def _file_requests_epi(path):
    """True if this op file (or a flat_ops.h it includes) issues an epi.suffix() kernel-name site."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if "epi.suffix()" in text:
        return True
    return bool(re.search(r'#include\s+"flat_ops\.h"', text))


def main():
    ap = argparse.ArgumentParser(description="VKNN pointwise-epilogue multi-surface sync check")
    ap.add_argument("--repo", default=REPO_DEFAULT, help="VKNN repo root (default: this tree)")
    args = ap.parse_args()

    shader_dir = os.path.join(args.repo, "shaders")
    ops_dir = os.path.join(args.repo, "src", "backend", "vulkan", "ops")
    pass_cpp = os.path.join(args.repo, "src", "import", "fuse_pointwise_chains.cpp")
    descriptor_cpp = os.path.join(args.repo, "src", "core", "op_descriptor.cpp")
    for p in (shader_dir, ops_dir, pass_cpp):
        if not os.path.exists(p):
            sys.exit("error: %s not found (pass --repo)" % p)

    epi_stems, rx_standalone = derive_shader_stems(shader_dir)
    req_stems = requested_stems(ops_dir, epi_stems)
    capable = parse_pw_capable(pass_cpp, descriptor_cpp)
    vk_reg = parse_vk_registry(ops_dir)

    problems = []

    # 1. every requested stem has an _epi shader.
    missing_shader = sorted(s for s in req_stems if s not in epi_stems)
    if missing_shader:
        problems.append(
            "op files request _epi kernels with no shader that #includes pw_epilogue.glsl "
            "(device-load crash): %s" % ", ".join(missing_shader))

    # 2. every shader-derived stem is requested by some op file.
    unused_shader = sorted(s for s in epi_stems if s not in req_stems)
    if unused_shader:
        problems.append(
            "shaders #include pw_epilogue.glsl but no op file requests them (dead SPIR-V + cache-hash "
            "churn): %s" % ", ".join(unused_shader))

    # 3. every pwEpilogueCapable OpType resolves to an op file that requests an epi stem.
    no_kernel = []
    for op in sorted(capable):
        files = vk_reg.get(op)
        if not files:
            no_kernel.append("%s (no VKNN_REGISTER_VK_OP)" % op)
            continue
        if not any(_file_requests_epi(os.path.join(ops_dir, f)) for f in files):
            no_kernel.append("%s (%s requests no epi kernel)" % (op, "/".join(sorted(files))))
    if no_kernel:
        problems.append(
            "pwEpilogueCapable OpTypes whose kernel cannot host an epilogue: %s" % ", ".join(no_kernel))

    print("check_epi_sync: %d shader stems (+%d standalone-VM), %d op-file-requested stems, "
          "%d pwEpilogueCapable OpTypes"
          % (len(epi_stems), len(rx_standalone), len(req_stems), len(capable)))

    if problems:
        print("FAIL — epilogue-surface drift:")
        for p in problems:
            print("  - %s" % p)
        sys.exit(1)
    print("PASS — pw_epilogue.glsl shaders, op-file stem sites, and pwEpilogueCapable agree")


if __name__ == "__main__":
    main()
