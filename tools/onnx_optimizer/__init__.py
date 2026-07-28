"""Bit-exact ONNX graph optimizer for VKNN.

Optimizes ONNX models BEFORE they are imported by vknn_compile, with one hard
guarantee: for every tested input, the optimized model's outputs are
byte-identical to the original's under a deterministic onnxruntime CPU
reference (ORT_DISABLE_ALL, single-threaded). Verification gates every write;
a failing model is never written without --force.

Entry points (run from tools/, or with PYTHONPATH=tools, or as plain scripts):

  python -m onnx_optimizer.optimize -i model.onnx -o model.opt.onnx
  python -m onnx_optimizer.verify a.onnx b.onnx

See README.md in this directory for the pass list and the verification
methodology.
"""

__version__ = "1.0.0"
