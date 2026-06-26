# YoNoSplat encoder -> VKNN reproducers (2026-06-25)
See memory [[yonosplat-encoder-onnx]] for the full story. These regenerate the validated encoder ONNX
if /tmp is cleared (the heavy artifacts live in /tmp, not git):

  python3 export_encoder.py --views 2 --ckpt pretrained_weights/re10k.ckpt --export   # -> yonosplat_encoder.onnx (+ scattered weights)
  python3 fix_and_validate.py                                                          # f64->f32, consolidate -> onnx/yonosplat_encoder.onnx, cos vs PyTorch
  # then: vknn_compile onnx/yonosplat_encoder.onnx encoder_fp16.vxm --fp16

Setup: clone https://github.com/cvg/YoNoSplat into /tmp/YoNoSplat; venv /tmp/yono-venv (system torch
2.10 + einops jaxtyping omegaconf scipy dacite); ckpt = HF botaoye/YoNoSplat re10k_224x224_ctx2to32.ckpt
(3.86GB) -> pretrained_weights/re10k.ckpt. Scripts assume /tmp/YoNoSplat paths; adjust if regenerating.
op_validate.py validates each new vknn op (LayerNorm/MatMul/Softmax/Where/Einsum/...) vs onnxruntime.
