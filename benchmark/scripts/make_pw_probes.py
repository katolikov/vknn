#!/usr/bin/env python3
"""Generate one small ONNX probe per producer family for the pw-epilogue device gate.

Each probe: producer -> chain of >=2 pointwise ops (Mul/Add/Sigmoid/Clip), fixed seed.
Writes <name>.onnx + <name>_in.bin (fp32 raw) into the output dir.
"""
import sys, pathlib
import numpy as np
from onnx import TensorProto, helper, numpy_helper

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "pw_probes")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)


def t(name, arr):
    return numpy_helper.from_array(arr.astype(np.float32), name)


def build(name, in_shape, nodes, inits, out_shape, extra_inputs=None):
    graph_inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, in_shape)]
    if extra_inputs:
        graph_inputs += extra_inputs
    g = helper.make_graph(
        nodes, name, graph_inputs,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    (OUT / f"{name}.onnx").write_bytes(m.SerializeToString())
    x = rng.standard_normal(in_shape).astype(np.float32)
    (OUT / f"{name}_in.bin").write_bytes(x.tobytes())
    print(f"{name}: in={in_shape} out={out_shape}")


def chain(after, shape, prefix, cshape=None):
    """Mul(const) -> Add(const) -> Clip chain after tensor `after` of `shape`."""
    cshape = cshape or shape
    scale = t(f"{prefix}_s", rng.standard_normal(cshape) * 0.5 + 1.0)
    bias = t(f"{prefix}_b", rng.standard_normal(cshape) * 0.1)
    nodes = [
        helper.make_node("Mul", [after, f"{prefix}_s"], [f"{prefix}_m"]),
        helper.make_node("Add", [f"{prefix}_m", f"{prefix}_b"], [f"{prefix}_a"]),
        helper.make_node("Clip", [f"{prefix}_a", f"{prefix}_lo", f"{prefix}_hi"], ["y"]),
    ]
    inits = [scale, bias,
             numpy_helper.from_array(np.float32(-4.0), f"{prefix}_lo"),
             numpy_helper.from_array(np.float32(4.0), f"{prefix}_hi")]
    return nodes, inits


# --- conv 3x3 (NC4HW4 world) ---
N, C, H, W, CO = 1, 6, 14, 14, 10
wt = t("cw", rng.standard_normal((CO, C, 3, 3)) * 0.2)
cb = t("cb", rng.standard_normal((CO,)) * 0.1)
cn, ci = chain("c0", [N, CO, H, W], "pc", cshape=[1, CO, 1, 1])
build("p_conv", [N, C, H, W],
      [helper.make_node("Conv", ["x", "cw", "cb"], ["c0"], pads=[1, 1, 1, 1])] + cn,
      [wt, cb] + ci, [N, CO, H, W])

# --- conv 1x1 ---
wt = t("c1w", rng.standard_normal((CO, C, 1, 1)) * 0.3)
cn, ci = chain("c1", [N, CO, H, W], "p1", cshape=[1, CO, 1, 1])
build("p_conv1x1", [N, C, H, W],
      [helper.make_node("Conv", ["x", "c1w"], ["c1"])] + cn,
      [wt] + ci, [N, CO, H, W])

# --- depthwise conv ---
wt = t("dww", rng.standard_normal((C, 1, 3, 3)) * 0.3)
cn, ci = chain("d0", [N, C, H, W], "pd", cshape=[1, C, 1, 1])
build("p_dwconv", [N, C, H, W],
      [helper.make_node("Conv", ["x", "dww"], ["d0"], group=C, pads=[1, 1, 1, 1])] + cn,
      [wt] + ci, [N, C, H, W])

# --- convtranspose ---
wt = t("ctw", rng.standard_normal((C, CO, 3, 3)) * 0.2)
cn, ci = chain("ct0", [N, CO, H * 2, W * 2], "pt", cshape=[1, CO, 1, 1])
build("p_convtr", [N, C, H, W],
      [helper.make_node("ConvTranspose", ["x", "ctw"], ["ct0"], strides=[2, 2],
                        pads=[1, 1, 1, 1], output_padding=[1, 1])] + cn,
      [wt] + ci, [N, CO, H * 2, W * 2])

# --- softmax (flat world, rank 3) ---
sh = [2, 8, 32]
cn, ci = chain("sm0", sh, "ps")
build("p_softmax", sh, [helper.make_node("Softmax", ["x"], ["sm0"], axis=-1)] + cn, ci, sh)

# --- layernorm ---
sh = [2, 16, 24]
lns = t("lns", rng.standard_normal((24,)) * 0.5 + 1.0)
lnb = t("lnb", rng.standard_normal((24,)) * 0.1)
cn, ci = chain("ln0", sh, "pl")
build("p_layernorm", sh,
      [helper.make_node("LayerNormalization", ["x", "lns", "lnb"], ["ln0"], axis=-1)] + cn,
      [lns, lnb] + ci, sh)

# --- reduce (mean over last axis, keepdims) ---
sh = [2, 12, 16]
cn, ci = chain("rd0", [2, 12, 1], "pr")
build("p_reduce", sh,
      [helper.make_node("ReduceMean", ["x", "rd_ax"], ["rd0"], keepdims=1)] + cn,
      [numpy_helper.from_array(np.array([2], np.int64), "rd_ax")] + ci, [2, 12, 1])

# --- gridsample (runtime grid input) ---
gs_in = [1, 4, 12, 12]
grid = helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 10, 10, 2])
cn, ci = chain("gs0", [1, 4, 10, 10], "pg", cshape=[1, 4, 1, 1])
build("p_gridsample", gs_in,
      [helper.make_node("GridSample", ["x", "grid"], ["gs0"], mode="bilinear",
                        padding_mode="zeros", align_corners=1)] + cn,
      ci, [1, 4, 10, 10], extra_inputs=[grid])
gv = rng.uniform(-1, 1, (1, 10, 10, 2)).astype(np.float32)
(OUT / "p_gridsample_grid.bin").write_bytes(gv.tobytes())

# --- resize (2x bilinear) ---
sh = [1, 4, 8, 8]
cn, ci = chain("rz0", [1, 4, 16, 16], "pz", cshape=[1, 4, 1, 1])
build("p_resize", sh,
      [helper.make_node("Resize", ["x", "", "rz_sc"], ["rz0"], mode="linear",
                        coordinate_transformation_mode="half_pixel")] + cn,
      [numpy_helper.from_array(np.array([1, 1, 2, 2], np.float32), "rz_sc")] + ci,
      [1, 4, 16, 16])

# --- avgpool 2x2 ---
sh = [1, 6, 12, 12]
cn, ci = chain("ap0", [1, 6, 6, 6], "pa", cshape=[1, 6, 1, 1])
build("p_avgpool", sh,
      [helper.make_node("AveragePool", ["x"], ["ap0"], kernel_shape=[2, 2], strides=[2, 2])] + cn,
      ci, [1, 6, 6, 6])

# --- maxpool 2x2 ---
cn, ci = chain("mp0", [1, 6, 6, 6], "pm", cshape=[1, 6, 1, 1])
build("p_maxpool", sh,
      [helper.make_node("MaxPool", ["x"], ["mp0"], kernel_shape=[2, 2], strides=[2, 2])] + cn,
      ci, [1, 6, 6, 6])

# --- global average pool ---
cn, ci = chain("gp0", [1, 6, 1, 1], "pp", cshape=[1, 6, 1, 1])
build("p_gap", sh, [helper.make_node("GlobalAveragePool", ["x"], ["gp0"])] + cn, ci, [1, 6, 1, 1])

# --- gemm (classifier shape) ---
gi, go = 32, 20
gw = t("gw", rng.standard_normal((go, gi)) * 0.2)
gb = t("gb", rng.standard_normal((go,)) * 0.1)
cn, ci = chain("gm0", [2, go], "pgm")
build("p_gemm", [2, gi],
      [helper.make_node("Gemm", ["x", "gw", "gb"], ["gm0"], transB=1)] + cn,
      [gw, gb] + ci, [2, go])

# --- matmul (batched, re-probe) ---
a_sh = [2, 24, 16]
mw = t("mw", rng.standard_normal((16, 20)) * 0.3)
cn, ci = chain("mm0", [2, 24, 20], "pmm")
build("p_matmul", a_sh, [helper.make_node("MatMul", ["x", "mw"], ["mm0"])] + cn, [mw] + ci,
      [2, 24, 20])

# --- extra coverage probes ---

# stride-2 1x1 (conv1x1_s2 kernel)
wt = t("s2w", rng.standard_normal((CO, C, 1, 1)) * 0.3)
cn, ci = chain("s2", [N, CO, H // 2, W // 2], "ps2", cshape=[1, CO, 1, 1])
build("p_conv1x1s2", [N, C, H, W],
      [helper.make_node("Conv", ["x", "s2w"], ["s2"], strides=[2, 2])] + cn,
      [wt] + ci, [N, CO, H // 2, W // 2])

# deep small-spatial 1x1 (split-K path in fp16)
dc = 64
wt = t("dpw", rng.standard_normal((dc, dc, 1, 1)) * 0.1)
cn, ci = chain("dp", [1, dc, 7, 7], "pdp", cshape=[1, dc, 1, 1])
build("p_conv1x1deep", [1, dc, 7, 7],
      [helper.make_node("Conv", ["x", "dpw"], ["dp"])] + cn, [wt] + ci, [1, dc, 7, 7])

# 3x3 32ch (Winograd path with --winograd on in fp16)
wc = 32
wt = t("w3w", rng.standard_normal((wc, wc, 3, 3)) * 0.1)
wb = t("w3b", rng.standard_normal((wc,)) * 0.1)
cn, ci = chain("w3", [1, wc, 28, 28], "pw3", cshape=[1, wc, 1, 1])
build("p_conv3x3w", [1, wc, 28, 28],
      [helper.make_node("Conv", ["x", "w3w", "w3b"], ["w3"], pads=[1, 1, 1, 1])] + cn,
      [wt, wb] + ci, [1, wc, 28, 28])

# NC4 channel softmax ([N,C,1,1], axis=1)
sc = 8
cn, ci = chain("sn0", [1, sc, 1, 1], "psn", cshape=[1, sc, 1, 1])
build("p_softmax_nc4", [1, sc, 1, 1],
      [helper.make_node("Softmax", ["x"], ["sn0"], axis=1)] + cn, ci, [1, sc, 1, 1])

# Gemm WITHOUT bias: first appended operand lands at inputs[2] (bias-misread guard)
gw2 = t("gw2", rng.standard_normal((20, 32)) * 0.2)
cn, ci = chain("gnb", [2, 20], "pgn")
build("p_gemm_nobias", [2, 32],
      [helper.make_node("Gemm", ["x", "gw2"], ["gnb"], transB=1)] + cn, [gw2] + ci, [2, 20])

# LayerNorm WITHOUT beta: operand at inputs[2] (beta-misread guard)
lns2 = t("lns2", rng.standard_normal((24,)) * 0.5 + 1.0)
cn, ci = chain("lnb0", [2, 16, 24], "pln")
build("p_layernorm_nobeta", [2, 16, 24],
      [helper.make_node("LayerNormalization", ["x", "lns2"], ["lnb0"], axis=-1)] + cn,
      [lns2] + ci, [2, 16, 24])

# Reduce with axes as ATTR: operand at inputs[1] (axes-misread guard)
cn, ci = chain("rda", [2, 12, 1], "prda")
build("p_reduce_attr", [2, 12, 16],
      [helper.make_node("ReduceMean", ["x"], ["rda"], axes=[2], keepdims=1)] + cn,
      ci, [2, 12, 1])

# maxpool with SAME-SHAPE constant operands (NC4 bc=0 packed-constant path, C%4!=0)
cn, ci = chain("mpf", [1, 6, 6, 6], "pmf")
build("p_maxpool_full", [1, 6, 12, 12],
      [helper.make_node("MaxPool", ["x"], ["mpf"], kernel_shape=[2, 2], strides=[2, 2])] + cn,
      ci, [1, 6, 6, 6])
