# How to run YoNoSplat (3D Gaussian Splatting) end to end

Goal: run the full feed-forward 3DGS pipeline on the GPU — image -> transformer **encoder** (on VKNN
Vulkan) -> Gaussians -> from-scratch **Vulkan rasterizer** -> rendered view. Helper scripts and the
export recipe live in [`../scripts/yonosplat/`](../scripts/yonosplat/) (see its
[README](../scripts/yonosplat/README.md)).

## What runs where

- The **encoder** (DINOv2 ViT-L/14 backbone + RoPE decoders + Gaussian/camera heads, 965M params) is a
  normal VKNN `Session`. It runs 100% on the GPU and produces 6 Gaussian outputs
  (means, covariances, harmonics, opacities, rotations, scales).
- The **rasterizer** is a from-scratch Vulkan compute pipeline (preprocess -> tile-bin -> stable radix
  sort -> per-tile alpha compositing). `examples/splatting/yonosplat.cpp` wires the two together.

## One-shot demo

```sh
./build.sh --android
scripts/yonosplat/run_demo.sh        # pushes the binary, runs, pulls the PPM; expects encoder_fp16.vxm + input bins already under /data/local/tmp/vxrt/yono
```

Or run the binary directly on the device:

```sh
adb shell /data/local/tmp/vxrt/vknn_yonosplat \
  encoder_fp16.vxm image.bin intrinsics.bin out.ppm [--extr extr.bin] [--view N] \
  [--render S] [--repeat N] [--raw out.f32] [--packed out.u32]
```

- `image.bin` = fp32 `[1, V, 3, 224, 224]`, `intrinsics.bin` = fp32 `[1, V, 3, 3]` (normalized).
- `scripts/yonosplat/fetch_re10k_test.py` builds real input bins from a RealEstate10K test scene
  (`re10k_image.bin` `[1,2,3,224,224]` + `re10k_intr.bin` `[1,2,3,3]`, plus preview PNGs).
- `--render S` rasterizes at SxS instead of 224 (the normalized intrinsics scale with it). `--repeat N`
  renders N times and requires byte-identical fp32 output (determinism gate). `--raw` dumps the fp32
  render; `--packed` also runs the packed-ARGB path and verifies it equals the round-half-up 8-bit
  quantization of the fp32 render.
- `extr.bin` (optional) = fp32 `[V, 4, 4]` camera-to-world; the encoder predicts the pose itself
  (dumpable via `vknn_run_io --dump <tensor>`, i.e. `Config::dumpTensors` — `run_demo.sh` does exactly
  this), identity if omitted. Renders view `N`.

## Producing the encoder .vxm and goldens

```sh
# 1. Export the encoder to a validated ONNX (faithful monkeypatches; cos=1.0 vs the original).
python3 scripts/yonosplat/export_encoder.py --views 2 --ckpt /tmp/YoNoSplat/pretrained_weights/re10k.ckpt --export
python3 scripts/yonosplat/fix_and_validate.py   # scripts hardcode the /tmp/YoNoSplat checkout paths (see the scripts README)

# 2. Compile to an fp16 .vxm.
./build-host/vknn_compile onnx/yonosplat_encoder.onnx encoder_fp16.vxm --fp16   # the fix_and_validate.py output, not the raw export

# 3. Generate the 6-output onnxruntime golden for validation.
python3 scripts/yonosplat/gen_golden.py
```

## Validate

- Encoder: compare the 6 outputs against the ORT golden (target: bit-identical on the CPU fp32 path;
  fp16 GPU within fp16 noise). To inspect intermediates, set `Config::dumpTensors` (`vknn_run_io --dump "t1,t2"`) — the liveness
  planner aliases buffers, so this forces those tensors into dedicated buffers and dumps them — then
  compare cosine per tensor.
- Rasterizer: `scripts/yonosplat/ref_rasterizer.py` is the CPU reference (the gsplat "classic" math);
  validated cos=1.0 on synthetic and real encoder outputs.

Re-push `build-android/vknn_yonosplat` after every Android rebuild. The project memory / docs cover the
transformer kernels (LayerNorm, batched MatMul, attention Softmax, RoPE, Einsum lowering).
