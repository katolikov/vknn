"""Per-op correctness: build a tiny ONNX for each new vxrt op (realistic transformer shapes),
run onnxruntime (reference) and vxrt CPU, compare. Confirms the kernels independently."""
import numpy as np, onnx, subprocess, os, tempfile
from onnx import helper as H, TensorProto as T, numpy_helper as NH
import onnxruntime as ort

VX = "/Users/artemkatolikov/DEV/LibTAS/build-host/vx_run_io"
np.random.seed(0)
D = "/tmp/opval"; os.makedirs(D, exist_ok=True)

def run(name, nodes, inputs, outputs, inits=None, feeds=None):
    """inputs: list of (name,shape) fp32 graph inputs; feeds: dict name->np array."""
    g = H.make_graph(nodes, name,
                     [H.make_tensor_value_info(n, T.FLOAT, s) for n, s in inputs],
                     [H.make_tensor_value_info(o, T.FLOAT, None) for o in outputs],
                     initializer=inits or [])
    m = H.make_model(g, opset_imports=[H.make_opsetid("", 17)]); m.ir_version = 10
    p = f"{D}/{name}.onnx"; onnx.save(m, p)
    feeds = feeds or {n: np.random.randn(*s).astype(np.float32) for n, s in inputs}
    try:
        ref = ort.InferenceSession(p, providers=["CPUExecutionProvider"]).run(outputs, feeds)
    except Exception as e:
        print(f"  {name:14} ORT-ERR {str(e).splitlines()[-1][:70]}"); return False
    # vxrt
    od = f"{D}/{name}_out"; os.makedirs(od, exist_ok=True)
    for f in os.listdir(od): os.remove(f"{od}/{f}")
    args = [VX, p, od]
    for n, s in inputs:
        bf = f"{D}/{name}_{n}.bin"; feeds[n].tofile(bf); args.append(bf)
    r = subprocess.run(args, capture_output=True, text=True)
    ok_all = True
    for o, rr in zip(outputs, ref):
        vf = f"{od}/{o}.bin"
        if not os.path.exists(vf):
            print(f"  {name:14} {o:10} VXRT-NO-OUTPUT  (stderr: {r.stderr.strip().splitlines()[-1][:80] if r.stderr.strip() else r.stdout.strip()[-80:]})")
            ok_all = False; continue
        v = np.fromfile(vf, dtype=np.float32).astype(np.float64)
        b = rr.reshape(-1).astype(np.float64)
        m_ = min(len(v), len(b))
        cos = float(np.dot(v[:m_], b[:m_]) / (np.linalg.norm(v[:m_]) * np.linalg.norm(b[:m_]) + 1e-12))
        mad = float(np.max(np.abs(v[:m_] - b[:m_])))
        ok = cos > 0.9999 and len(v) == len(b)
        ok_all &= ok
        print(f"  {name:14} {o:10} cos={cos:.6f} max|d|={mad:.2e} len {len(v)}/{len(b)} {'OK' if ok else 'FAIL'}")
    return ok_all

def ci(name, arr):  # const initializer
    return NH.from_array(arr, name)

print("=== per-op validation: vxrt CPU vs onnxruntime ===")
# LayerNorm [1,256,1024]
run("layernorm", [H.make_node("LayerNormalization", ["x","g","b"], ["y"], axis=-1, epsilon=1e-6)],
    [("x",[1,256,1024])], ["y"],
    inits=[ci("g",np.random.randn(1024).astype(np.float32)), ci("b",np.random.randn(1024).astype(np.float32))])
# MatMul batched attention [1,8,64,16]@[1,8,16,64]
run("matmul_b", [H.make_node("MatMul",["a","b"],["y"])], [("a",[1,8,64,16]),("b",[1,8,16,64])], ["y"])
# MatMul linear [256,1024]@[1024,512]
run("matmul_w", [H.make_node("MatMul",["a","w"],["y"])], [("a",[256,1024])], ["y"],
    inits=[ci("w",np.random.randn(1024,512).astype(np.float32))])
# Softmax last axis [1,8,64,64]
run("softmax", [H.make_node("Softmax",["x"],["y"],axis=-1)], [("x",[1,8,64,64])], ["y"])
# Real pattern Equal->Where: cond = Equal(idx, const), then Where(cond, X, Y). idx is an int-valued
# fp tensor (like the model's shape/index masks). Tests Equal (fp32 1/0) + Where together.
idx=np.tile(np.arange(64,dtype=np.float32),(1,1))   # [1,64] = 0..63
run("equal_where", [H.make_node("Equal",["idx","k"],["m"]), H.make_node("Where",["m","x","y"],["o"])],
    [("x",[1,64]),("y",[1,64])], ["o"],
    inits=[ci("k",np.full((1,64),32.0,np.float32))], feeds={"idx":idx,
        "x":np.random.randn(1,64).astype(np.float32),"y":np.random.randn(1,64).astype(np.float32)})
# Expand [1,1,64]->[1,8,64]
run("expand", [H.make_node("Expand",["x","s"],["y"])], [("x",[1,1,64])], ["y"],
    inits=[ci("s",np.array([1,8,64],np.int64))])
# Tile [1,64]->[2,128]
run("tile", [H.make_node("Tile",["x","r"],["y"])], [("x",[1,64])], ["y"],
    inits=[ci("r",np.array([2,2],np.int64))])
# Einsum i,j->ij
run("einsum_ij", [H.make_node("Einsum",["a","b"],["y"],equation="i,j->ij")], [("a",[64]),("b",[32])], ["y"])
# DepthToSpace [1,8,8,8] block2 DCR
run("d2s", [H.make_node("DepthToSpace",["x"],["y"],blocksize=2,mode="DCR")], [("x",[1,8,8,8])], ["y"])
# ReduceL2 [1,256,16] axis -1 keepdims
run("reducel2", [H.make_node("ReduceL2",["x"],["y"],axes=[-1],keepdims=1)], [("x",[1,256,16])], ["y"])
# Erf/Cos/Sin/Reciprocal/Softplus
run("gelu_ops", [H.make_node("Erf",["x"],["e"]),H.make_node("Cos",["e"],["c"]),
                 H.make_node("Sin",["c"],["s2"]),H.make_node("Reciprocal",["s2"],["r"]),
                 H.make_node("Softplus",["r"],["y"])], [("x",[1,128])], ["y"])
print("done")
