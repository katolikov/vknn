# Generate ORT goldens for the encoder's 6 outputs from the device input bins.
import numpy as np, onnxruntime as ort, os, json
ONNX="/tmp/YoNoSplat/onnx/yonosplat_encoder.onnx"
img=np.fromfile("/tmp/yono_image.bin",dtype=np.float32).reshape(1,2,3,224,224)
intr=np.fromfile("/tmp/yono_intrinsics.bin",dtype=np.float32).reshape(1,2,3,3)
so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess=ort.InferenceSession(ONNX,so,providers=["CPUExecutionProvider"])
outs=[o.name for o in sess.get_outputs()]
res=sess.run(outs,{"image":img,"intrinsics":intr})
meta={}
for name,arr in zip(outs,res):
    arr=np.ascontiguousarray(arr,dtype=np.float32)
    arr.tofile(f"/tmp/yono_{name}_gold.bin")
    meta[name]=list(arr.shape)
    print(name, arr.shape, "mean",float(arr.mean()),"std",float(arr.std()))
json.dump(meta,open("/tmp/yono_gold_meta.json","w"))
print("saved goldens + meta")
