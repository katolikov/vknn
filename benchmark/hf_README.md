---
license: mit
library_name: vknn
tags:
  - 3d-gaussian-splatting
  - yonosplat
  - vulkan
  - onnx
---

# YoNoSplat encoder — ONNX + vknn (`.vxm`)

The **YoNoSplat** feed-forward 3D-Gaussian-Splatting *encoder* (8 context views), exported to ONNX and
compiled to a fp16 [vknn](https://github.com/katolikov/vknn) `.vxm` for on-device Vulkan inference.

| file | what |
|---|---|
| `yonosplat_encoder.onnx` | encoder graph, opset 17 (weights external in `weights.bin`) |
| `weights.bin` | fp32 weights for the ONNX (~3.6 GB) |
| `encoder8_fp16.vxm` | compiled fp16 vknn model, ready to run (~2.9 GB) |
| `image8.npy`, `intr8.npy` | sample real 8-frame input (RealEstate10K scene) |
| `*_gold.npy` | onnxruntime goldens for the six outputs |

**Input:** `image [1,8,3,224,224]`, `intrinsics [1,8,3,3]`.
**Output:** `means [1,401408,3]`, `covariances`, `harmonics`, `opacities`, `rotations`, `scales`
(401408 = 8 views × 224×224 Gaussians).

## Use

```sh
pip install huggingface_hub
python benchmark/scripts/fetch_model.py --repo katolikov/yonosplat-vknn --out benchmark/models
# then run + validate on device with the benchmark tool (see vknn/benchmark/USAGE.md)
python benchmark/run.py run benchmark/configs/yonosplat.json
```

On the Samsung Xclipse 960 (Vulkan, fp16) the compiled encoder matches an onnxruntime golden at
cosine ≥ 0.9997 on all six outputs.

## Provenance & license

- Architecture + checkpoint: **YoNoSplat** by Botao Ye et al. (code MIT; checkpoint
  `re10k_224x224_ctx2to32`). The ONNX here is a faithful export (analytic 3×3 inverse, Newton–Schulz
  SVD, baked DINOv2 pos-embed) validated cos = 1.0 against the original.
- The weights are trained on **RealEstate10K**, whose dataset terms are research / non-commercial —
  use accordingly.
- This export/compilation is provided under MIT, with attribution to the original authors.
