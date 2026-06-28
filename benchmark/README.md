# benchmark

Unified driver to **convert** a model, **run** it on the device GPU (Vulkan), **validate** every
output against a golden `.npy`, and report **runtime** — all from one JSON config.

## Pieces
- `benchmark.py` — host driver (orchestrates convert + `adb push` + on-device run + validation + timing).
- `validate.cpp` → `vknn_validate` — the on-device executor: loads `.npy` inputs (shape/dtype from the
  npy header), runs the model, checks each named output against a golden `.npy`. Built by CMake.
- `make_golden.py` — generate golden `.npy` (via onnxruntime) + a config from an ONNX and input npys.

## Quick start
```bash
# 1. (optional) generate goldens + a config from an ONNX and input npys
python benchmark/make_golden.py model.onnx out/ image=image.npy intrinsics=intr.npy

# 2. convert + run on device + validate + time, all in one
python benchmark/benchmark.py all benchmark/example.json -n 5
```

## Subcommands
| command | does |
|---|---|
| `convert ONNX OUT.vxm [--fp16/--fp32] [--on host\|device]` | compile onnx → vxm |
| `run CONFIG.json` | push + one run + validate on device |
| `bench CONFIG.json [-n N] [--cooldown S]` | N timed runs, median submit+gpu |
| `all CONFIG.json [-n N]` | convert (cfg.onnx) + push + run + validate + bench |

## Config (`example.json`)
Host paths are pushed to the device; `device_dir` is where everything lands.
```json
{
  "onnx": "/abs/path/encoder.onnx",
  "model": "encoder8_fp16.vxm",
  "fp16": true,
  "backend": "vulkan", "precision": "fp16",
  "device_dir": "/data/local/tmp/vxrt/bench",
  "inputs":  { "image": "image8.npy", "intrinsics": "intr8.npy" },
  "outputs": { "means": "means_gold.npy", "scales": "scales_gold.npy" },
  "tolerance": 0.999, "cooldown": 22, "bench": 5
}
```
`inputs` may also be a positional array (`["image8.npy", "intr8.npy"]`, model input order).

## Direct on-device use (no driver)
`vknn_validate config.json` runs entirely on the device against device-local paths — `benchmark.py`
just stages files and writes that config for you.

Prereqs: `./build.sh --android` (device binaries) and `./build.sh` (host `vknn_compile`); a connected
`adb` device. Runtime is thermally sensitive — `bench` cools down between runs.
