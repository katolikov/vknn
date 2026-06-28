# benchmark

One JSON config, one command: **convert** a model (or use a ready `.vxm`), **run** it on the device
GPU, **save**/**validate** outputs, and collect **timing + profiling**.

```sh
./build.sh && ./build.sh --android
python benchmark/benchmark.py run benchmark/example.json
```

## Contents
- `benchmark.py` — host driver (per-stage: convert + `adb push` + on-device run + validate + timing).
- `validate.cpp` → `vknn_validate` — on-device executor: `.npy`/raw inputs (or none for runtime-only),
  save `.npy`/`.png`, golden compare (cosine / PSNR / SNR / relL2 / max), result JSON with timing and
  optional per-operator profiling.
- `make_golden.py` — generate golden `.npy` (via onnxruntime) + a config from an ONNX.
- `example.json` — sample two-stage config.
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
