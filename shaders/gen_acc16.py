#!/usr/bin/env python3
"""Generate the fp16-accumulator twin of an fp16-storage conv kernel.

Precision::Low carries the reduction in fp16 as well as the storage (VulkanBackend::useFp16Arith):
half the accumulator VGPRs and packed-fp16 math. The twin differs from its source only in the
accumulator/operand types, so it is derived here rather than hand-maintained -- editing the source
kernel and re-running this keeps the pair from drifting.

    python3 shaders/gen_acc16.py <stem>_fp16.comp   ->  <stem>_acc16_fp16.comp
"""
import re
import sys
from pathlib import Path

# (pattern, replacement) applied in order. Each moves one fp32 accumulator/operand to fp16.
RULES = [
    (r"\bvec4 acc\[", "f16vec4 acc["),
    (r"\bvec4 row\[", "f16vec4 row["),
    (r"\bvec4 w\[", "f16vec4 w["),
    (r"vec4 inv = vec4\(src\[([^\]]+)\]\);", r"f16vec4 inv = src[\1];"),
    (r"\bvec4 inv = ", "f16vec4 inv = "),
    (r"vec4 b = \(ocb0 \+ j < Coutb\) \? vec4\(bias\[ocb0 \+ j\]\) : vec4\(0\.0\);",
     "f16vec4 b = (ocb0 + j < Coutb) ? bias[ocb0 + j] : f16vec4(0.0);"),
    (r"vec4 b = vec4\(bias\[cb\]\);", "f16vec4 b = bias[cb];"),
    (r"= ok \? vec4\(src\[([^\]]+)\]\) : vec4\(0\.0\);", r"= ok ? src[\1] : f16vec4(0.0);"),
    (r"w\[j\]\[(\d)\] = vec4\((wt\[.*\])\);", r"w[j][\1] = \2;"),
    (r"vec4 wv = vec4\(wt\[([^\]]+)\]\);", r"f16vec4 wv = wt[\1];"),
    # The activation epilogue stays fp32: vx_act takes floats and the store converts back.
    (r"vx_act\(acc\[([^\]]+)\]\[([^\]]+)\]\.([xyzw])", r"vx_act(float(acc[\1][\2].\3)"),
    (r"vx_act\(acc\[([^\]]+)\]\.([xyzw])", r"vx_act(float(acc[\1].\2)"),
]

HEADER = ("// GENERATED from {src} by shaders/gen_acc16.py -- edit that file, not this one.\n"
          "// fp16-accumulator twin: the reduction rounds to fp16 at every tap instead of carrying\n"
          "// fp32, so it is NOT bit-exact with its source. Selected only at Precision::Low.\n")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    src = Path(sys.argv[1])
    text = src.read_text()
    for pattern, repl in RULES:
        text = re.sub(pattern, repl, text)
    lines = text.split("\n")
    # Keep #version first, then stamp the provenance header above the source's own comment block.
    text = lines[0] + "\n" + HEADER.format(src=src.name) + "\n".join(lines[1:])
    dst = src.with_name(src.name.replace("_fp16.comp", "_acc16_fp16.comp"))
    dst.write_text(text)
    print(f"wrote {dst.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
