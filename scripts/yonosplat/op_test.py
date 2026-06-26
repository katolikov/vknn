# Build a small ONNX test model for a new GPU op, generate an ORT golden, save input+golden bins.
# Usage: op_test.py <op>   ; then run on device: vknn_run_io <op>.onnx out --backend vulkan <op>_in*.bin
import sys, numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto as TP
op = sys.argv[1]
np.random.seed(0)
def save(name, arr): arr.astype(np.float32).tofile(f"/tmp/optest_{name}.bin")
def build_and_golden(model, inputs:dict):
    onnx.save(model, f"/tmp/optest_{op}.onnx")
    so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess=ort.InferenceSession(f"/tmp/optest_{op}.onnx", so, providers=["CPUExecutionProvider"])
    outs=sess.run(None, inputs)
    for i,(n,v) in enumerate(inputs.items()): save(f"{op}_in{i}", v)
    for o,arr in zip(sess.get_outputs(), outs):
        np.ascontiguousarray(arr,dtype=np.float32).tofile(f"/tmp/optest_{op}_gold.bin")
        print(f"{op}: golden {o.name} shape {arr.shape} mean {arr.mean():.5f}")
        break

if op=="gather":
    X=np.random.randn(2,16,3,8,4).astype(np.float32)
    idx=onnx.helper.make_tensor("idx",TP.INT64,[],[1])   # rank-0 scalar -> drops axis 2
    n=helper.make_node("Gather",["X","idx"],["Y"],axis=2)
    g=helper.make_graph([n],"g",[helper.make_tensor_value_info("X",TP.FLOAT,X.shape)],
        [helper.make_tensor_value_info("Y",TP.FLOAT,[2,16,8,4])],[idx])
    build_and_golden(helper.make_model(g,opset_imports=[helper.make_opsetid("",17)]),{"X":X})

if op=="einsum_mv":  # ...ab,...b->...a with batch broadcast (A batch 1 vs x batch 4) -> ray unprojection
    A=np.random.randn(1,3,3).astype(np.float32); x=np.random.randn(4,3).astype(np.float32)
    n=helper.make_node("Einsum",["A","X"],["Y"],equation="...ab,...b->...a")
    g=helper.make_graph([n],"g",[helper.make_tensor_value_info("A",TP.FLOAT,A.shape),helper.make_tensor_value_info("X",TP.FLOAT,x.shape)],
        [helper.make_tensor_value_info("Y",TP.FLOAT,[4,3])])
    build_and_golden(helper.make_model(g,opset_imports=[helper.make_opsetid("",17)]),{"A":A,"X":x})

if op=="rope_gather":  # data[17,32] gathered by a runtime float index [4,5] along axis 0 -> [4,5,32]
    T=np.random.randn(17,32).astype(np.float32)
    idxv=np.random.randint(0,17,size=(4,5)).astype(np.float32)
    # make index a runtime activation: Identity(idx_input)
    n1=helper.make_node("Gather",["T","I"],["Y"],axis=0)
    g=helper.make_graph([n1],"g",[helper.make_tensor_value_info("T",TP.FLOAT,T.shape),helper.make_tensor_value_info("I",TP.INT64,[4,5])],
        [helper.make_tensor_value_info("Y",TP.FLOAT,[4,5,32])])
    # ORT needs int64 index; VKNN reads it as float at runtime. Save idx as float for VKNN, int64 for ORT golden.
    onnx.save(helper.make_model(g,opset_imports=[helper.make_opsetid("",17)]),"/tmp/optest_rope_gather.onnx")
    so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess=ort.InferenceSession("/tmp/optest_rope_gather.onnx",so,providers=["CPUExecutionProvider"])
    gold=sess.run(None,{"T":T,"I":idxv.astype(np.int64)})[0]
    T.tofile("/tmp/optest_rope_gather_in0.bin"); idxv.tofile("/tmp/optest_rope_gather_in1.bin")
    np.ascontiguousarray(gold,np.float32).tofile("/tmp/optest_rope_gather_gold.bin")
    print("rope_gather golden",gold.shape)

if op=="scatternd":  # data[3,4,5] scattered: idx[6,3] runtime float, updates[6] scalars
    data=np.random.randn(3,4,5).astype(np.float32)
    idx=np.array([[0,0,0],[1,2,3],[2,3,4],[0,1,1],[1,1,1],[2,0,2]],dtype=np.int64)
    upd=np.random.randn(6).astype(np.float32)
    n=helper.make_node("ScatterND",["D","I","U"],["Y"])
    g=helper.make_graph([n],"g",[helper.make_tensor_value_info("D",TP.FLOAT,data.shape),
        helper.make_tensor_value_info("I",TP.INT64,[6,3]),helper.make_tensor_value_info("U",TP.FLOAT,[6])],
        [helper.make_tensor_value_info("Y",TP.FLOAT,[3,4,5])])
    onnx.save(helper.make_model(g,opset_imports=[helper.make_opsetid("",17)]),"/tmp/optest_scatternd.onnx")
    so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess=ort.InferenceSession("/tmp/optest_scatternd.onnx",so,providers=["CPUExecutionProvider"])
    gold=sess.run(None,{"D":data,"I":idx,"U":upd})[0]
    data.tofile("/tmp/optest_scatternd_in0.bin"); idx.astype(np.float32).tofile("/tmp/optest_scatternd_in1.bin"); upd.tofile("/tmp/optest_scatternd_in2.bin")
    np.ascontiguousarray(gold,np.float32).tofile("/tmp/optest_scatternd_gold.bin"); print("scatternd golden",gold.shape)

if op=="matmul7":  # rank-7 batched 3x3 matmul (the intrinsics inverse application)
    A=np.random.randn(1,2,4,1,1,3,3).astype(np.float32); B=np.random.randn(1,2,4,1,1,3,3).astype(np.float32)
    n=helper.make_node("MatMul",["A","B"],["Y"])
    g=helper.make_graph([n],"g",[helper.make_tensor_value_info("A",TP.FLOAT,A.shape),helper.make_tensor_value_info("B",TP.FLOAT,B.shape)],
        [helper.make_tensor_value_info("Y",TP.FLOAT,[1,2,4,1,1,3,3])])
    build_and_golden(helper.make_model(g,opset_imports=[helper.make_opsetid("",17)]),{"A":A,"B":B})
