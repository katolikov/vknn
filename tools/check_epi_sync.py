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
  4. the plan-encoding constants agree by VALUE across their two spellings: the #defines in
     shaders/pw_epilogue.glsl (PW_BCAST_*, PW_KIND_*, PW_REF_*, PW_PACKED_STRIDE_*, PW_STEP_FIELDS,
     PW_EPI_MAXSTEPS/MAXRANK, NC4_LANES) and the constexpr ints in include/vknn/op_type.h +
     include/vknn/nchw.h (kPwBcast*, kPwKind*, kPwRef*, kPwPackedStride*, kPwStepInts, kPwMaxSteps,
     kPwMaxRank, kNC4Block). The plan is encoded host-side and decoded shader-side by these codes, so
     a renumber or an addition on one side alone is a wrong-answer-on-device bug that no host build
     and no compile can see.
  5. a standalone applier does not hand-roll the broadcast-class dispatch (see check_applier_twins).

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


def requested_stems(ops_dir):
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


# --- the plan-encoding constant mirror (check 4) ----------------------------------------------
# The pw plan is written host-side and read shader-side by raw integer codes, so every code lives in
# two files that no compiler compares. These tables define the mirror: a macro under a mirrored
# prefix must have a same-valued host constant, and a host constant under the matching prefix must
# have a macro. Both directions are DERIVED from the names, so a class added or renumbered on one
# side alone fails here rather than on device.
_HOST_CONST_RE = re.compile(
    r'\b(?:inline\s+)?constexpr\s+(?:int|int64_t)\s+(k[A-Za-z0-9_]+)\s*=\s*\(?\s*(-?\d+)\s*\)?\s*;')
_SHADER_DEFINE_RE = re.compile(
    r'^[ \t]*#define[ \t]+([A-Z][A-Z0-9_]*)[ \t]+\(?[ \t]*(-?\d+)[ \t]*\)?[ \t]*(?://.*)?$', re.M)
# (shader prefix, host prefix): every member of the family mirrors, in both directions.
_MIRROR_FAMILIES = (
    ("PW_BCAST_", "kPwBcast"),
    ("PW_KIND_", "kPwKind"),
    ("PW_REF_", "kPwRef"),
    ("PW_PACKED_STRIDE_", "kPwPackedStride"),
)
# Pairs whose two spellings share no derivable stem.
_MIRROR_SINGLETONS = {
    "PW_EPI_MAXSTEPS": "kPwMaxSteps",
    "PW_EPI_MAXRANK": "kPwMaxRank",
    "PW_STEP_FIELDS": "kPwStepInts",
    "NC4_LANES": "kNC4Block",
}
# Host constants whose family prefix is mirrored but which name no shader macro (the shader reads
# the field by its literal offset inside the step record, which PW_STEP_FIELDS already pins).
_HOST_ONLY = ("kPwStepBcastField",)


def _shader_word_to_camel(word):
    """SCREAMING_SNAKE tail -> the CamelCase tail of the host name (ROW_SPLAT -> RowSplat)."""
    return "".join(w[:1].upper() + w[1:].lower() for w in word.split("_") if w)


def _camel_to_shader_word(tail):
    """CamelCase tail -> the SCREAMING_SNAKE tail of the macro (RowSplat -> ROW_SPLAT)."""
    return "_".join(p.upper() for p in re.findall(r'[A-Z][a-z0-9]*|[0-9]+', tail))


def mirrored_host_name(macro):
    """Host constant a shader macro must equal, or None when the macro is outside the mirror."""
    if macro in _MIRROR_SINGLETONS:
        return _MIRROR_SINGLETONS[macro]
    for shader_prefix, host_prefix in _MIRROR_FAMILIES:
        if macro.startswith(shader_prefix):
            return host_prefix + _shader_word_to_camel(macro[len(shader_prefix):])
    return None


def mirrored_macro_name(host):
    """Shader macro a host constant must equal, or None when the constant is outside the mirror."""
    if host in _HOST_ONLY:
        return None
    for macro, name in _MIRROR_SINGLETONS.items():
        if host == name:
            return macro
    for shader_prefix, host_prefix in _MIRROR_FAMILIES:
        if host.startswith(host_prefix) and len(host) > len(host_prefix):
            return shader_prefix + _camel_to_shader_word(host[len(host_prefix):])
    return None


def parse_host_constants(*headers):
    """name -> int for every `constexpr int k... = <literal>;` in the given headers."""
    values = {}
    for path in headers:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for name, value in _HOST_CONST_RE.findall(text):
            values[name] = int(value)
    return values


def parse_shader_defines(path):
    """macro -> int for every `#define NAME <literal>` in one shader/include file."""
    with open(path, encoding="utf-8") as f:
        return {m: int(v) for m, v in _SHADER_DEFINE_RE.findall(f.read())}


def compare_constant_mirror(host_values, shader_values):
    """Return a list of problems: every mirrored code that is missing or differs on either side.

    Pure comparison over two name -> value maps, so it is exercised on synthetic drift by
    verify_mirror_detector() as well as on the tree.
    """
    problems = []
    for macro in sorted(shader_values):
        host = mirrored_host_name(macro)
        if host is None:
            continue
        if host not in host_values:
            problems.append("shaders/pw_epilogue.glsl defines %s but include/ has no %s "
                            "(the class exists only shader-side)" % (macro, host))
        elif host_values[host] != shader_values[macro]:
            problems.append("%s = %d but %s = %d (host encodes what the shader decodes: the plan "
                            "would be misread on device)"
                            % (macro, shader_values[macro], host, host_values[host]))
    for host in sorted(host_values):
        macro = mirrored_macro_name(host)
        if macro is None:
            continue
        if macro not in shader_values:
            problems.append("include/ declares %s but shaders/pw_epilogue.glsl has no %s "
                            "(the class exists only host-side)" % (host, macro))
    return problems


def verify_mirror_detector():
    """Self-check: the comparison above must flag each drift shape, or the check is vacuous.

    A regex that stops matching (a spelling change in either file) empties the maps and every
    comparison trivially agrees, so the detector is run against synthetic drift on every invocation.
    """
    host = {"kPwBcastSame": 0, "kPwBcastScalar": 3, "kPwMaxSteps": 16}
    shader = {"PW_BCAST_SAME": 0, "PW_BCAST_SCALAR": 3, "PW_EPI_MAXSTEPS": 16}
    cases = [
        ("renumbered class", host, dict(shader, PW_BCAST_SCALAR=4)),
        ("shader-only class", host, dict(shader, PW_BCAST_PACKED=9)),
        ("host-only class", dict(host, kPwBcastPacked=9), shader),
        ("renumbered singleton", host, dict(shader, PW_EPI_MAXSTEPS=8)),
    ]
    if compare_constant_mirror(host, shader):
        sys.exit("error: check_epi_sync self-check: the constant mirror reports drift on agreeing "
                 "inputs (comparison out of date)")
    for label, h, s in cases:
        if not compare_constant_mirror(h, s):
            sys.exit("error: check_epi_sync self-check: %s goes undetected (comparison out of date)"
                     % label)


def check_constant_mirror(repo):
    """Compare shaders/pw_epilogue.glsl's plan codes against their include/ counterparts."""
    verify_mirror_detector()
    glsl = os.path.join(repo, "shaders", "pw_epilogue.glsl")
    op_type_h = os.path.join(repo, "include", "vknn", "op_type.h")
    nchw_h = os.path.join(repo, "include", "vknn", "nchw.h")
    for path in (glsl, op_type_h, nchw_h):
        if not os.path.isfile(path):
            sys.exit("error: %s not found (pass --repo)" % path)
    shader_values = parse_shader_defines(glsl)
    host_values = parse_host_constants(op_type_h, nchw_h)
    # Vacuity guard: an empty side would make every comparison trivially agree, so each mirrored
    # family must be populated on both sides before the values are compared at all.
    problems = []
    for shader_prefix, host_prefix in _MIRROR_FAMILIES:
        if not any(m.startswith(shader_prefix) for m in shader_values):
            problems.append("no %s* macro parsed from shaders/pw_epilogue.glsl (parser out of date)"
                            % shader_prefix)
        if not any(h.startswith(host_prefix) for h in host_values):
            problems.append("no %s* constant parsed from include/vknn/op_type.h (parser out of date)"
                            % host_prefix)
    for macro in _MIRROR_SINGLETONS:
        if macro not in shader_values:
            problems.append("no %s macro parsed from shaders/pw_epilogue.glsl (parser out of date)"
                            % macro)
    return problems or compare_constant_mirror(host_values, shader_values)


# --- the standalone appliers' broadcast dispatch (check 5) -------------------------------------
# A standalone applier (fused_pw_flat.comp / fused_pw_flat_v4.comp / fused_pw_nc4.comp) carries a
# monomorphized twin of the interpreter, specialized on the step count for occupancy. The twin must
# reach operands through the SHARED helpers in pw_epilogue.glsl: a second, hand-rolled copy of the
# broadcast-class dispatch diverges from the interpreter the moment a class is added, and the twin
# is the path that runs for every 1..8-step unit, so the divergence is the default rather than the
# exception. A twin MAY fast-path a class it can widen (the quad load of a same-shape operand) as
# long as it falls through to a shared resolver for the rest.
_CLASS_HELPER_PARAM_RE = re.compile(
    r'\b(?:int|vec4)\s+(?:pwLoadBc4|pwFlatIdx|pwNc4Idx)\s*\(([^)]*)\)')
# The shared resolvers a twin must reach the remaining classes through.
_CLASS_RESOLVERS = ("pwFlatIdx", "pwLoadBc4", "pwNc4Idx")


def class_variable_names(glsl_path):
    """Parameter names the shared helpers carry the broadcast class in (derived, not hardcoded).

    The helper signatures are the definition of "this variable holds a class code", so a twin that
    compares one of these names against a class is dispatching on the class.
    """
    with open(glsl_path, encoding="utf-8") as f:
        text = f.read()
    names = set()
    for params in _CLASS_HELPER_PARAM_RE.findall(text):
        for param in params.split(","):
            parts = param.split()
            # The class parameter sits between the step index and the index argument; both helpers
            # spell it right after `int s` / `int slot,int s`, so take every int parameter and keep
            # the ones the bodies compare against a PW_BCAST_* macro (below).
            if len(parts) == 2 and parts[0] == "int":
                names.add(parts[1])
    compared = set()
    for name in names:
        if re.search(r'\b%s\s*==\s*PW_BCAST_' % re.escape(name), text):
            compared.add(name)
    return compared


def check_applier_twins(shader_dir):
    """Return a list of problems: hand-rolled broadcast-class dispatch outside pw_epilogue.glsl."""
    glsl = os.path.join(shader_dir, "pw_epilogue.glsl")
    if not os.path.isfile(glsl):
        sys.exit("error: %s not found (pass --repo)" % glsl)
    class_names = class_variable_names(glsl)
    if not class_names:
        sys.exit("error: no broadcast-class parameter derived from pw_epilogue.glsl's shared "
                 "helpers (parser out of date; the twin check would be vacuous)")
    known_classes = {m for m in parse_shader_defines(glsl) if m.startswith("PW_BCAST_")}
    dispatch = re.compile(r'\b(%s)\s*==\s*(\d+|PW_BCAST_[A-Z0-9_]*)'
                          % "|".join(sorted(re.escape(n) for n in class_names)))
    problems = []
    for name in sorted(os.listdir(shader_dir)):
        if not name.startswith("fused_pw_") or not name.endswith(".comp"):
            continue
        with open(os.path.join(shader_dir, name), encoding="utf-8") as fh:
            text = fh.read()
        dispatches = False
        for lineno, line in enumerate(text.splitlines(), 1):
            for var, operand in dispatch.findall(line):
                dispatches = True
                if operand.isdigit():
                    problems.append(
                        "%s:%d dispatches on a bare class code (%s == %s); compare the named "
                        "PW_BCAST_* constant so the value mirror covers it"
                        % (name, lineno, var, operand))
                elif operand not in known_classes:
                    problems.append("%s:%d compares against %s, which pw_epilogue.glsl does not "
                                    "define" % (name, lineno, operand))
        if dispatches and not any(r in text for r in _CLASS_RESOLVERS):
            problems.append(
                "%s open-codes the broadcast-class dispatch with no fall-through to a shared "
                "resolver (%s); route the classes it does not widen through pw_epilogue.glsl so the "
                "twin cannot diverge from the interpreter" % (name, " / ".join(_CLASS_RESOLVERS)))
    return problems


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
    req_stems = requested_stems(ops_dir)
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

    problems += check_constant_mirror(args.repo)
    problems += check_applier_twins(shader_dir)

    mirrored = len([m for m in parse_shader_defines(os.path.join(shader_dir, "pw_epilogue.glsl"))
                    if mirrored_host_name(m)])
    print("check_epi_sync: %d shader stems (+%d standalone-VM), %d op-file-requested stems, "
          "%d pwEpilogueCapable OpTypes, %d mirrored plan constants"
          % (len(epi_stems), len(rx_standalone), len(req_stems), len(capable), mirrored))

    if problems:
        print("FAIL — epilogue-surface drift:")
        for p in problems:
            print("  - %s" % p)
        sys.exit(1)
    print("PASS — pw_epilogue.glsl shaders, op-file stem sites, pwEpilogueCapable, and the plan "
          "constant mirror agree")


if __name__ == "__main__":
    main()
