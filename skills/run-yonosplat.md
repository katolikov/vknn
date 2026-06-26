# How to run YoNoSplat (3D Gaussian Splatting) end to end

Goal: run the full feed-forward 3DGS pipeline on the GPU — image -> transformer **encoder** (on VKNN
Vulkan) -> Gaussians -> from-scratch **Vulkan rasterizer** -> rendered view. Helper scripts and the
export recipe live in [`../scripts/yonosplat/`](../scripts/yonosplat/) (see its
[README](../scripts/yonosplat/README.md)).

## What runs where

- The **encoder** (DINOv2 ViT-L/14 backbone + RoPE decoders + Gaussian/camera heads, 965M params) is a
  normal VKNN `Session`. It runs 100% on the GPU and produces 6 Gaussian outputs
  (means, covariances, harmonics, opacities, rotations, scales).
- The **rasterizer** is a from-scratch Vulkan compute pipeline (preprocess -> tile-bin -> bitonic sort
  -> per-tile alpha compositing), all in one GPU command buffer. `examples/yonosplat.cpp` wires the two
  together.

## One-shot demo

```sh
./build.sh --android
scripts/yonosplat/run_demo.sh        # pushes binary + encoder .vxm + inputs, runs, pulls the PPM
```

Or run the binary directly on the device:

```sh
adb shell /data/local/tmp/vxrt/vknn_yonosplat \
  encoder_fp16.vxm image.bin intrinsics.bin out.ppm [--extr extr.bin] [--view N]
```

- `image.bin` = fp32 `[1, V, 3, 224, 224]`, `intrinsics.bin` = fp32 `[1, V, 3, 3]` (normalized).
- `extr.bin` (optional) = fp32 `[V, 4, 4]` camera-to-world; the encoder predicts the pose itself
  (dumpable via `VKNN_DUMP_NAMES`), identity if omitted. Renders view `N`.

## Producing the encoder .vxm and goldens

```sh
# 1. Export the encoder to a validated ONNX (faithful monkeypatches; cos=1.0 vs the original).
python3 scripts/yonosplat/export_encoder.py
python3 scripts/yonosplat/fix_and_validate.py

# 2. Compile to an fp16 .vxm.
./build-host/vknn_compile yonosplat_encoder.onnx encoder_fp16.vxm --fp16

# 3. Generate the 6-output onnxruntime golden for validation.
python3 scripts/yonosplat/gen_golden.py
```

## Validate

- Encoder: compare the 6 outputs against the ORT golden (target: bit-identical on the CPU fp32 path;
  fp16 GPU within fp16 noise). To inspect intermediates, set `VKNN_DUMP_NAMES="t1,t2"` — the liveness
  planner aliases buffers, so this forces those tensors into dedicated buffers and dumps them — then
  compare cosine per tensor.
- Rasterizer: `scripts/yonosplat/ref_rasterizer.py` is the CPU reference (the gsplat "classic" math);
  validated cos=1.0 on synthetic and real encoder outputs.

Re-push `build-android/vknn_yonosplat` after every Android rebuild. The project memory / docs cover the
transformer kernels (LayerNorm, batched MatMul, attention Softmax, RoPE, Einsum lowering).
