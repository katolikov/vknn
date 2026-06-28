# How to benchmark / validate a model on the device

Goal: from one JSON config, convert a model (or use a ready `.vxm`), run it on the device GPU, save
and/or check its outputs against goldens, and collect timing + per-operator profiling. The tool lives
in [`../benchmark/`](../benchmark/); the full field reference is
[../benchmark/USAGE.md](../benchmark/USAGE.md).

## 1. Build

```sh
./build.sh           # host vknn_compile (host-side convert)
./build.sh --android # device binaries incl. vknn_benchmark (re-push after every android build)
```

## 2. Inputs/outputs as `.npy` (or raw `.bin`/`.raw`)

`.npy` carries shape + dtype in its header, so nothing is hand-specified — `numpy.save("image.npy",
arr)`. Reading accepts f32/f16/f64/i64/i32/i8/u8 (→ fp32), C-order. Any input not ending in `.npy`
(`.bin`, `.raw`, …) is a headerless fp32 dump sized to the model input (shape from the model). Outputs
save as `.npy`/`.raw` (exact) and/or `.png` (image-shaped tensors only). **No `inputs` → runtime-only**
(zero-filled, nothing saved/checked).

Generate goldens + a config from an ONNX:
```sh
python benchmark/make_golden.py model.onnx out/ image=image.npy intrinsics=intr.npy
```

## 3. Write the config

Sectioned and **staged** — each stage is independent; `defaults` is merged into all stages:
```jsonc
{ "defaults": { "device": { "backend": "vulkan", "precision": "fp16",
                            "dir": "/data/local/tmp/vxrt/bench", "max_submit_nodes": 500, "cooldown": 22 } },
  "stages": [
    { "name": "m",
      "model":   { "onnx": "m.onnx" },                 // or { "vxm": "m.vxm" } to skip convert
      "convert": { "fp16": true, "fuse_se": false, "out": "m.vxm" },
      "inputs":  { "image": "image.npy" },             // or ["a.npy","b.bin"]; omit -> runtime only
      "outputs": { "save": ["npy","png"], "golden": { "out": "out_gold.npy" },
                   "metrics": ["cosine","psnr","snr","relL2","max"] },
      "profile": true, "bench": 5, "tolerance": 0.999, "result": "m.result.json" } ] }
```
- `model`: one of `onnx` (converted with `convert` options) or `vxm` (as-is).
- `convert`: convert-time opts (`fp16`, `no_fuse_swish`, `fuse_se`, `fuse_dwpw`, `out`).
- `device`: runtime opts (`backend`, `precision`, `dir`, `no_weight_cache`, `max_submit_nodes`, `cooldown`).
- `metrics`: any of cosine / psnr / snr / relL2 / max. Pass = `cosine ≥ tolerance` and no NaN.

## 4. Run

```sh
python benchmark/run.py run config.json          # all stages on the device
python benchmark/run.py convert m.onnx m.vxm --fp16   # standalone convert
```
Per stage it prints `submit+gpu` (median over `bench` runs) and the per-output metrics, and writes
`results/<result>` (timing + per-output metrics + per-op `profile` when enabled).

Notes:
- **Profiling forces per-op barriers** (no overlap) → only valid for runs that fit a single submit
  under the GPU watchdog; leave `profile` off for very long runs (e.g. the 8-view YoNoSplat encoder).
- Runtime is thermally sensitive (the device throttles 3–5×); `cooldown` sleeps before each run and
  `bench` reports the median. See [compile-and-run-a-model.md](compile-and-run-a-model.md) for the
  lower-level `vknn_compile` / `vknn_run_io` path and [../docs/CONFIG.md](../docs/CONFIG.md).
