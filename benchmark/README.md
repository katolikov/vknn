# benchmark

One JSON config, one command: **convert** a model (or use a ready `.vxm`), **run** it on the device
GPU, **save**/**validate** outputs, and collect **timing + profiling**.

```sh
./build.sh && ./build.sh --android
python benchmark/run.py run benchmark/configs/example.json
```

## Model files (HuggingFace)

The YoNoSplat encoder is too large for git, so the ONNX + weights + compiled `.vxm` live in a
HuggingFace model repo. Fetch them with:
```sh
pip install huggingface_hub
python benchmark/scripts/fetch_model.py --repo katolikov/yonosplat-vknn --out benchmark/models
python benchmark/run.py run benchmark/configs/yonosplat.json     # ready-made example (uses the vxm)
```
`fetch_model.py` pulls the model (`yonosplat_encoder.onnx` + `weights.bin` + `encoder8_fp16.vxm`) and
the sample inputs/goldens the example config uses; a private repo needs `hf auth login`. `yonosplat.json`
runs the compiled vxm (validate stage with golden metrics + a runtime-only stage). Publishing the
artifacts is a one-time step via `upload_model.py`.

## Contents
- `run.py` — host driver (per-stage: convert + `adb push` + on-device run + validate + timing).
- `benchmark.cpp` → `vknn_benchmark` — on-device executor: `.npy` or raw (`.bin`/`.raw`) inputs (or none
  for runtime-only), save `.npy`/`.raw`/`.png`, golden compare (cosine / PSNR / SNR / relL2 / max),
  result JSON with timing and optional per-operator profiling.
- `configs/` — JSON configs (`example.json` sample two-stage config, `yonosplat*.json`, …).
- `scripts/` — helper scripts: `make_golden.py` (golden `.npy` via onnxruntime + a config from an ONNX),
  `fetch_model.py`, `upload_model.py`.
- **[USAGE.md](USAGE.md)** — full how-to: the `.npy` mechanism and every config field.

## Commands
| command | does |
|---|---|
| `run CONFIG.json` | run every stage on the device (convert if the stage gives an `onnx`) |
| `convert ONNX OUT.vxm [--fp16/--fp32] [--fuse-se] [--fuse-dwpw] [--no-fuse-swish] [--on host\|device]` | standalone convert |

A config is a list of independent **stages** (each: `model` onnx-or-vxm, `convert` options, `device`
options, `inputs`, `outputs`, `profile`, `bench`) plus an optional `defaults` block. See
[USAGE.md](USAGE.md). Prereqs: a connected `adb` device; runtime is thermally sensitive so `bench`
cools down between runs.
