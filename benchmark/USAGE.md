# Benchmark & validate: npy files + config reference

`benchmark.py` runs one or more **stages** on the device from a single JSON config: convert a model
(or use a ready `.vxm`), feed inputs, optionally save outputs, compare against goldens, and collect a
per-stage result JSON with timing and (optional) per-operator profiling. The on-device executor is
`vknn_validate` (built from `benchmark/validate.cpp`); `benchmark.py` stages files over `adb` and runs
it.

```sh
./build.sh && ./build.sh --android        # host vknn_compile + device binaries
python benchmark/benchmark.py run benchmark/example.json
```

## 1. Input / output files

### `.npy` (recommended)
NumPy's array format carries **shape and dtype in its header**, so you never hand-specify them. Write
one with `numpy.save("image.npy", arr)`. Reading supports `float32`, `float16`, `float64`, `int64`,
`int32`, `int8`, `uint8` (all converted to fp32 for the engine); arrays must be C-order.

### raw `.bin` (alternative)
A headerless little-endian **fp32** dump (`arr.astype(np.float32).tofile("in.bin")`). Because it has
no shape, the file must contain exactly the model input's element count — the shape is taken from the
model. Use `.npy` unless you already have `.bin` dumps.

### Outputs
With `"save"` set, each output is written next to the model on the device (pull them yourself), and the
result JSON lists what was written:
- `"npy"` — fp32 `.npy` (always works, exact).
- `"png"` — written when the tensor looks like an image (`[..,C,H,W]` or `[..,H,W,C]`, C∈{1,3,4});
  values are min–max normalised to 0–255. Non-image tensors are skipped.

If a stage has **no `inputs`**, the model is run on zero-filled inputs for a **runtime-only**
measurement: nothing is saved and no goldens are checked.

## 2. Golden comparison metrics

`"golden"` maps an output name to a golden `.npy`. `"metrics"` selects which to report (default: all):

| metric | meaning |
|---|---|
| `cosine` | cosine similarity (pass threshold = `tolerance`, default 0.999) |
| `psnr` | peak signal-to-noise ratio, dB (`20·log10(range / RMSE)`) |
| `snr` | signal-to-noise ratio, dB (`10·log10(Σgolden² / Σerr²)`) |
| `relL2` | relative L2 error `‖a−b‖ / ‖b‖` |
| `max` | max absolute difference |

A pass needs `cosine ≥ tolerance` and zero NaNs.

## 3. Config schema

A config is a list of `stages` (each fully independent) plus an optional `defaults` block merged into
every stage. A single-stage config may drop `stages` and put the fields at the top level.

```jsonc
{
  "defaults": { "device": { "backend": "vulkan", "precision": "fp16",
                            "dir": "/data/local/tmp/vxrt/bench", "no_weight_cache": true,
                            "max_submit_nodes": 500, "cooldown": 22 } },
  "stages": [
    {
      "name": "encoder8",
      "model":   { "onnx": "encoder.onnx" },          // OR { "vxm": "encoder.vxm" } to skip convert
      "convert": { "fp16": true, "fuse_se": false, "fuse_dwpw": false, "no_fuse_swish": false,
                   "out": "encoder8_fp16.vxm" },       // convert-time options (ignored for a vxm)
      "device":  { "backend": "vulkan", "precision": "fp16", "dir": "/data/local/tmp/vxrt/bench",
                   "no_weight_cache": true, "max_submit_nodes": 500, "cooldown": 22 },
      "inputs":  { "image": "image8.npy", "intrinsics": "intr8.bin" },  // by name; or ["a.npy","b.bin"]; omit -> runtime only
      "outputs": { "save": ["npy", "png"],
                   "golden": { "means": "means_gold.npy", "scales": "scales_gold.npy" },
                   "metrics": ["cosine", "psnr", "snr", "relL2", "max"] },
      "profile": true,            // per-operator GPU timing in the result JSON
      "bench": 5,                 // repeat N timed runs (median reported); default 1
      "tolerance": 0.999,
      "result": "encoder8.result.json"
    }
  ]
}
```

### Sections
- **`model`** — exactly one of `onnx` (compiled to `.vxm` with `convert` options) or `vxm` (run as-is).
- **`convert`** — convert-time optimization options (only when `onnx` is given): `fp16`,
  `no_fuse_swish`, `fuse_se`, `fuse_dwpw`, `out` (output `.vxm` name).
- **`device`** — runtime options: `backend` (vulkan/cpu), `precision` (fp16/fp32), `dir` (device
  staging dir), `no_weight_cache`, `max_submit_nodes` (GPU-watchdog chunk size; 0 = single submit),
  `cooldown` (seconds slept before each run — the device throttles).
- **`inputs`** — `.npy`/`.bin` per input, by name (object) or positionally (array). Omit for
  runtime-only.
- **`outputs`** — `save` formats, `golden` map, `metrics` list.
- **`profile`**, **`bench`**, **`tolerance`**, **`result`** — see above.

> Profiling forces a per-op barrier (no overlap), so it only works for models that finish a single
> submit under the GPU watchdog. Leave `profile` off for very long runs (e.g. the 8-view encoder).

## 4. Result JSON

Each stage writes `results/<result>` on the host:
```jsonc
{ "model": "encoder8_fp16.vxm", "backend": "VULKAN",
  "timing_ms": { "load": 12743.3, "run": 9756.9 },
  "profile": [ { "name": "...", "type": "MatMul", "gpu_ms": 1.2, "cpu_ms": 0 }, ... ],  // if profile
  "profile_by_type_ms": { "MatMul": 1856.5, "Conv": 53.4, ... },                        // if profile
  "gpu_total_ms": 10.18,                                                                 // if profile
  "outputs": [ { "name": "means", "shape": [1,401408,3], "saved": ["means.npy"],
                 "metrics": { "cosine": 0.999972, "psnr": 49.1, "snr": 36.0, "relL2": 0.016,
                              "max": 0.29, "nan": 0 }, "pass": true } ] }
```

## 5. Generating goldens

`make_golden.py` runs the model in onnxruntime to produce a golden `.npy` per output and writes a
matching config:
```sh
python benchmark/make_golden.py model.onnx out/ image=image.npy intrinsics=intr.npy
```

## 6. Direct on-device use (no driver)

`vknn_validate config.json` runs entirely on the device against device-local paths — `benchmark.py`
just stages files and writes that flat config. Run `convert` standalone with
`benchmark.py convert model.onnx model.vxm --fp16 [--fuse-se ...] [--on host|device]`.
