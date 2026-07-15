# benchmark

One JSON config, one command: **convert** a model (or use a ready `.vxm`), **run** it on the device
GPU, **validate** outputs against goldens, and collect **timing + fallbacks + profiling** under
`result/<RUN>/<stage>/`.

```sh
./build.sh && ./build.sh --android
python benchmark/run.py run benchmark/configs/example.json
```

Each stage prints its numbers on the host — `load` ms, `run median/min/avg/max` ms, and the ops
that fell back off the requested backend (a release run prints `fallbacks: 0`). A missing
model/input/golden stops the run loudly. `--clean` wipes the device work dir first; `--run NAME`
names the result directory.

`run` rebuilds the Android binaries automatically first (incremental; `--no-build` skips it) and
converts `.onnx` models on the host — `--convert-on device` compiles on the phone instead.

## Model files (HuggingFace)

The YoNoSplat encoder is too large for git, so the ONNX + weights + compiled `.vxm` live in a
HuggingFace model repo. Fetch them with:
```sh
pip install huggingface_hub
python benchmark/scripts/fetch_model.py --repo katolikov/yonosplat-vknn --out benchmark/models
python benchmark/run.py run benchmark/configs/yonosplat.json     # ready-made example (uses the vxm)
```
`fetch_model.py` pulls the model (`yonosplat_encoder.onnx` + `weights.bin` + `encoder8_fp16.vxm`) and
the sample inputs/goldens the example config uses; a private repo needs `hf auth login`. Publishing
the artifacts is a one-time step via `upload_model.py`.

## Contents
- `run.py` — host driver: convert + `adb push` + on-device run + validate; collects
  `result/<RUN>/<stage>/{result.json, outputs/, logcat.txt}` (+ the `.vxm`/`.cache` with `"pull"`).
- `benchmark.cpp` → `vknn_benchmark` — on-device executor: `.npy` or raw (`.bin`/`.raw`) inputs
  (single set or one set per input directory), save `.npy`/`.raw`/`.png`, golden compare
  (cosine / PSNR / SNR / relL2 / max), fallback report, result JSON with timing and optional
  per-operator profiling.
- `configs/` — JSON configs (`example.json` two-stage sample, `yonosplat*.json`, …).
- `scripts/` — helper scripts: `make_golden.py` (golden `.npy` via onnxruntime + a config from an
  ONNX), `fetch_model.py`, `upload_model.py`, the device byte gates — `gate_op.sh` (per-op: compiles a probe
  fused + `--no-fuse-pointwise`, runs both on the device, byte-compares every output, asserts zero
  CPU fallback), `gate_pw_probes.sh` (the same gate over the pointwise-fusion probe suite;
  `make_pw_probes.py` builds one probe model per producer family), `gate_lib.sh` (shared gate
  helpers, sourced) — and `dev_perfab.sh` (cooled, interleaved min-of-N perf A/B of two
  `vknn_run_io` binaries).
- **[USAGE.md](USAGE.md)** — full how-to: results layout, the `.npy` mechanism, every config field.

## Commands
| command | does |
|---|---|
| `run CONFIG.json [--run NAME] [--clean] [-v] [--no-build] [--convert-on host\|device]` | run every stage on the device |
| `convert ONNX OUT.vxm [-O 0..3] [--fp32] [--fuse-se] [--fuse-dwpw] [--on host\|device] [--serial S] [--no-build] [-v]` | standalone convert |

A config is a list of independent **stages** (each: `model`, `convert`, `device`, `run`, `inputs`,
`golden`, `metrics`, `save`, `pull`) plus an optional `defaults` block. See [USAGE.md](USAGE.md).
Prereqs: a connected `adb` device; the runtime is thermally sensitive, so use `run.iters`/`warmup`
(min/median over warm iterations) and `device.cooldown` for cross-run comparability.
