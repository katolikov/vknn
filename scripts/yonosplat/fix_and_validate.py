"""Post-process the exported encoder ONNX:
  1. retype all float64 (DOUBLE) tensors/casts/values -> float32 (the weights were
     originally fp32; const-folding promoted them, so this is lossless. Embedding
     math in fp32 vs fp64 differs ~1e-6, negligible and matches fp16 device target).
  2. consolidate scattered external-data files into a single weights blob.
  3. validate the result numerically in onnxruntime against the PyTorch golden.
"""
import os, numpy as np, onnx
from onnx import TensorProto, numpy_helper

SRC = "/tmp/YoNoSplat/yonosplat_encoder.onnx"
OUT_DIR = "/tmp/YoNoSplat/onnx"
OUT = os.path.join(OUT_DIR, "yonosplat_encoder.onnx")
os.makedirs(OUT_DIR, exist_ok=True)

print("[*] loading exported ONNX (+ external weights)...")
m = onnx.load(SRC, load_external_data=True)
g = m.graph

# 1a. initializers DOUBLE -> FLOAT
n_init = 0
for init in g.initializer:
    if init.data_type == TensorProto.DOUBLE:
        arr = numpy_helper.to_array(init).astype(np.float32)
        init.CopyFrom(numpy_helper.from_array(arr, init.name))
        n_init += 1
print(f"[*] retyped {n_init} DOUBLE initializers -> FLOAT")

# 1b. Cast nodes to=DOUBLE -> FLOAT, and Constant(double) -> float
n_cast = 0
for node in g.node:
    if node.op_type == "Cast":
        for at in node.attribute:
            if at.name == "to" and at.i == TensorProto.DOUBLE:
                at.i = TensorProto.FLOAT
                n_cast += 1
    if node.op_type == "Constant":
        for at in node.attribute:
            if at.name == "value" and at.t.data_type == TensorProto.DOUBLE:
                arr = numpy_helper.to_array(at.t).astype(np.float32)
                at.t.CopyFrom(numpy_helper.from_array(arr))
print(f"[*] retyped {n_cast} Cast->DOUBLE -> Cast->FLOAT")

# 1c. value_info / inputs / outputs DOUBLE -> FLOAT
n_vi = 0
for vi in list(g.value_info) + list(g.input) + list(g.output):
    tt = vi.type.tensor_type
    if tt.elem_type == TensorProto.DOUBLE:
        tt.elem_type = TensorProto.FLOAT
        n_vi += 1
print(f"[*] retyped {n_vi} DOUBLE value-infos -> FLOAT")

print("[*] saving consolidated model (single weights file)...")
for f in os.listdir(OUT_DIR):
    p = os.path.join(OUT_DIR, f)
    if os.path.isfile(p):
        os.remove(p)
onnx.save(m, OUT, save_as_external_data=True, all_tensors_to_one_file=True,
          location="weights.bin", convert_attribute=False)
print(f"[+] wrote {OUT}  (+ weights.bin {os.path.getsize(os.path.join(OUT_DIR,'weights.bin'))/1e6:.0f} MB)")

# 3. validate in onnxruntime against the PyTorch golden
print("[*] validating retyped ONNX in onnxruntime vs PyTorch golden...")
import onnxruntime as ort
ref = "/tmp/YoNoSplat/ref"
image = np.load(f"{ref}/image.npy")
intr = np.load(f"{ref}/intrinsics.npy")
names = ["means", "covariances", "harmonics", "opacities", "rotations", "scales"]
gold = {n: np.load(f"{ref}/{n}.npy") for n in names}

sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
outs = sess.run(names, {"image": image, "intrinsics": intr})
print("[+] ORT ran. per-output cosine / max|Δ| / rel-L2 vs PyTorch:")
allok = True
for n, o in zip(names, outs):
    a = o.reshape(-1).astype(np.float64)
    b = gold[n].reshape(-1).astype(np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mad = float(np.max(np.abs(a - b)))
    rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
    ok = cos > 0.9999
    allok = allok and ok
    print(f"      {n:12} cos={cos:.6f}  max|Δ|={mad:.3e}  relL2={rel:.3e}  {'OK' if ok else 'FAIL'}")
print("\n[+] ONNX MATCHES PYTORCH GOLDEN" if allok else "\n[!] MISMATCH")
