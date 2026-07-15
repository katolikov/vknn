# How to benchmark / validate a model on the device

Goal: from one JSON config, convert a model (or use a ready `.vxm`), run it on the device GPU, save
and/or check its outputs against goldens, and collect timing + per-operator profiling. The tool lives
in [`../benchmark/`](../benchmark/); the full field reference is
[../benchmark/USAGE.md](../benchmark/USAGE.md).

## 1. Build

```sh
./build.sh           # host vknn_compile (host-side convert)
./build.sh --android # device binaries incl. vknn_benchmark (optional — run.py builds and pushes them per run)
```

## 2. Inputs/outputs as `.npy` (or raw `.bin`/`.raw`)

`.npy` carries shape + dtype in its header, so nothing is hand-specified — `numpy.save("image.npy",
arr)`. Reading accepts f32/f16/f64/i64/i32/i8/u8 (→ fp32), C-order. Any input not ending in `.npy`
(`.bin`, `.raw`, …) is a headerless fp32 dump sized to the model input (shape from the model). Outputs
save as `.npy`/`.raw` (exact) and/or `.png` (image-shaped tensors only). **No `inputs` → runtime-only**
(zero-filled, nothing saved/checked).

Generate goldens + a config from an ONNX:
```sh
python benchmark/scripts/make_golden.py model.onnx out/ image=image.npy intrinsics=intr.npy
```

## 3. Write the config

Sectioned and **staged** — each stage is independent; `defaults` is merged into all stages. A
single-stage config may omit `stages` and put the stage fields at the top level:
```jsonc
{
  "defaults": {
    "device": {
      "serial": "",
      "dir": "/data/local/tmp/vknn/bench",
      "cooldown": 22
    },
    "run": {
      "backend": "vulkan",
      "precision": "fp16",
      "max_submit_nodes": 500
    }
  },
  "stages": [
    {
      "name": "m",
      "model": "m.onnx",                  // .onnx is converted; a .vxm runs as-is (no convert)
      "convert": {
        "fp16": true,
        "fuse_se": false,
        "out": "m.vxm"
      },
      "inputs": {                         // or ["a.npy", "b.bin"]; omit -> runtime only
        "image": "image.npy"
      },
      "outputs": {
        "save": ["npy", "png"],
        "golden": {
          "out": "out_gold.npy"
        },
        "metrics": ["cosine", "psnr", "snr", "relL2", "max"]
      },
      "profile": true,
      "run": { "iters": 5, "warmup": 2 },
      "tolerance": 0.999,
      "pull": ["vxm", "cache"]           // also pull the compiled .vxm + its .cache
    }
  ]
}
```
- `model`: a path — an `.onnx` is converted (with the `convert` options), a `.vxm` runs as-is
  (the `{"onnx":…}`/`{"vxm":…}` object form is legacy).
- `convert`: convert-time opts (`fp16` (default true), `opt` (0–3 → `-O0`..`-O3`, default 1; `"s"`
  → `-Os`: all fusion + INT4 weight quantization, fp16 implied), `no_fuse_swish`, `fuse_se`,
  `fuse_dwpw`, `no_fuse_pointwise`, `out`).
- `device`: device placement (`serial` (adb device id — set it when multiple devices are
  attached; `adb devices` lists them), `dir` (default `/data/local/tmp/vknn/bench`), `clean`,
  `cooldown`).
- `run`: runtime opts (`backend`, `precision`, `iters`, `warmup`, `profile`, `fold_islands`,
  `max_submit_nodes`, `winograd`, `tuning`, `tolerance`, `fp32_tensors`, `winogradVariant`,
  `winogradUnit`, `directConv3x3`, `generate_cache`, `cache`).
- `metrics`: any of cosine / psnr / snr / relL2 / max. Pass = `cosine ≥ tolerance` and no NaN.
- `inputs`: a name→file map, a file list (declared input order), or one or more directories — each
  directory is one input set (`<stem>.npy/.bin/.raw` feeds the input named `<stem>`;
  `<name>_gold.npy` in the dir is that set's golden for output `<name>`); each timed round runs
  every set once.

## 4. Run

```sh
python benchmark/run.py run config.json          # all stages (auto-runs ./build.sh --android first)
#   --run NAME (results dir name; default UTC timestamp)   --clean (wipe device dir per stage)
#   --convert-on host|device   --no-build (skip the auto-build)   -v (full device output)
python benchmark/run.py convert m.onnx m.vxm --fp16 -O 2   # standalone convert (--on device runs it on-device)
```
Per stage it prints load + run min/median/avg/max ms (over `run.iters` timed runs), the CPU-fallback
count (a clean release run shows 0), and the per-output metrics, and collects everything under
`result/<RUN>/<stage>/` (RUN = `--run NAME` or a UTC timestamp): `result.json` (timing + per-output
metrics + per-op `profile` when enabled), saved outputs, and the device logcat.

Notes:
- **Profiling forces per-op barriers** (no overlap) → only valid for runs that fit a single submit
  under the GPU watchdog; leave `profile` off for very long runs (e.g. the 8-view YoNoSplat encoder).
- Runtime is thermally sensitive (the device throttles 3–5×); `cooldown` sleeps before each stage's
  device run and `run.iters` > 1 reports the median. See
  [compile-and-run-a-model.md](compile-and-run-a-model.md) for the lower-level `vknn_compile` /
  `vknn_run_io` path and [../docs/config.md](../docs/config.md).
