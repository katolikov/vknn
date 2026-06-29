# Benchmark & validate: npy files + config reference

`run.py` runs one or more **stages** on the device from a single JSON config: convert a model
(or use a ready `.vxm`), feed inputs, optionally save outputs, compare against goldens, and collect a
per-stage result JSON with timing and (optional) per-operator profiling. The on-device executor is
`vknn_benchmark` (built from `benchmark/benchmark.cpp`); `run.py` stages files over `adb` and runs it.

```sh
./build.sh                                 # host vknn_compile (for host-side convert)
python benchmark/run.py run benchmark/configs/example.json       # auto-builds the device binaries first
python benchmark/run.py run benchmark/configs/example.json -v    # also print device stdout/stderr + the staged config
python benchmark/run.py run benchmark/configs/example.json --no-build   # reuse existing build-android/ binaries
```

`run` first runs `./build.sh --android` for you (incremental Ninja — a near-no-op when nothing changed;
`--no-build` skips it), so you never have to build the device binaries by hand before a run.

`run.py` logs each stage as it goes: the device serial/dir, every file pushed (or `MISSING` and skipped —
the device may already hold a copy), the inputs/goldens the device run will use, the run command and its
timing, and whether a `result.json` came back. The on-device executor writes its timing **and errors** to
stderr; `run.py` surfaces that stderr when a run produces no timing, and `-v` prints all of it plus the
generated config. A run that writes no `result.json` is reported as a failure (exit 3), not a silent pass.

Each stage's Android **logcat** is cleared before the run and saved to `results/<stage>.logcat.txt`
afterwards — that is where the GPU driver, thermal throttling, OOM-killer, and watchdog-reset messages
land (the executor's own stdout/stderr does not capture them).

## 1. Input / output files

### `.npy` (recommended)
NumPy's array format carries **shape and dtype in its header**, so you never hand-specify them. Write
one with `numpy.save("image.npy", arr)`. Reading supports `float32`, `float16`, `float64`, `int64`,
`int32`, `int8`, `uint8` (all converted to fp32 for the engine); arrays must be C-order.

### raw `.bin` / `.raw` (alternative)
A headerless little-endian **fp32** dump (`arr.astype(np.float32).tofile("in.raw")`). Any input whose
name does not end in `.npy` is read as raw — `.bin`, `.raw`, or any extension. Because it has no shape,
the file must contain exactly the model input's element count (the shape is taken from the model). Use
`.npy` unless you already have raw dumps.

### Outputs
With `"save"` set, each output is written next to the model on the device (pull them yourself), and the
result JSON lists what was written:
- `"npy"` — fp32 `.npy` (exact); `"raw"` — headerless fp32 `.raw` (always works, exact).
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
  "defaults": {
    "device": {
      "backend": "vulkan",
      "serial": "",
      "precision": "fp16",
      "dir": "/data/local/tmp/vxrt/bench",
      "cache_mode": "tune",
      "max_submit_nodes": 500,
      "cooldown": 22,
      "cache": "encoder8_fp16.cache",
      "generate_cache": false
    }
  },
  "stages": [
    {
      "name": "encoder8",

      "model": {
        "onnx": "encoder.onnx"            // OR  "vxm": "encoder.vxm"  to skip convert
      },
      "convert": {                        // convert-time options (ignored when a vxm is given)
        "fp16": true,
        "fuse_se": false,
        "fuse_dwpw": false,
        "no_fuse_swish": false,
        "out": "encoder8_fp16.vxm"
      },
      "device": {
        "backend": "vulkan",
        "precision": "fp16",
        "dir": "/data/local/tmp/vxrt/bench",
        "cache_mode": "tune",
        "max_submit_nodes": 500,
        "cooldown": 22,
        "cache": "encoder8_fp16.cache",   // unified per-model cache; default <model>.cache
        "generate_cache": false           // true -> warm the cache in an untimed load first
      },

      "inputs": {                         // by name; or an array ["a.npy", "b.bin"]; omit -> runtime only
        "image": "image8.npy",
        "intrinsics": "intr8.bin"
      },
      "outputs": {
        "save": ["npy", "png"],
        "golden": {
          "means": "means_gold.npy",
          "scales": "scales_gold.npy"
        },
        "metrics": ["cosine", "psnr", "snr", "relL2", "max"]
      },

      "profile": true,                    // per-operator GPU timing in the result JSON
      "bench": 5,                         // repeat N timed runs (median reported); default 1
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
- **`device`** — runtime options: `backend` (vulkan/cpu), `serial` (adb device serial/id; empty = the
  single attached device — **required when several devices are attached**), `precision` (fp16/fp32),
  `dir` (device staging dir), `cache_mode` (`off`/`tune`/`full`), `max_submit_nodes` (GPU-watchdog chunk size; 0 =
  single submit), `cooldown` (seconds slept before each run — the device throttles), `cache` (the
  unified per-model cache file; default `<model>.cache`), `generate_cache` (bool: when `true`, populate
  the cache in an **untimed throwaway load** first, so the timed load is warm and the cache-build cost
  is excluded from `timing_ms`). Each stage may target a different `serial`. Find serials with
  `adb devices`.
- **`inputs`** — `.npy`/`.bin` per input, by name (object) or positionally (array). Omit for
  runtime-only.
- **`outputs`** — `save` formats, `golden` map, `metrics` list.
- **`profile`**, **`bench`**, **`tolerance`**, **`result`** — see above.

> Profiling forces a per-op barrier (no overlap), so it only works for models that finish a single
> submit under the GPU watchdog. Leave `profile` off for very long runs (e.g. the 8-view encoder).

## 4. Result JSON

Each stage writes `results/<result>` on the host:
```jsonc
{
  "model": "encoder8_fp16.vxm",
  "backend": "VULKAN",
  "timing_ms": {
    "load": 12743.3,                       // cold load (compile + autotune + write <model>.cache);
                                           // with "generate_cache": true this is the warm load instead
    "run": 9756.9
  },

  // the next three keys appear only when "profile" is true
  "profile": [
    { "name": "/enc/.../MatMul", "type": "MatMul", "gpu_ms": 1.2, "cpu_ms": 0 }
    // ... one entry per operator ...
  ],
  "profile_by_type_ms": {
    "MatMul": 1856.5,
    "Conv": 53.4
  },
  "gpu_total_ms": 10.18,

  "outputs": [
    {
      "name": "means",
      "shape": [1, 401408, 3],
      "saved": ["means.npy"],
      "metrics": {
        "cosine": 0.999972,
        "psnr": 49.1,
        "snr": 36.0,
        "relL2": 0.016,
        "max": 0.29,
        "nan": 0
      },
      "pass": true
    }
  ]
}
```

## 5. Generating goldens

`make_golden.py` runs the model in onnxruntime to produce a golden `.npy` per output and writes a
matching config:
```sh
python benchmark/scripts/make_golden.py model.onnx out/ image=image.npy intrinsics=intr.npy
```

## 6. Direct on-device use (no driver)

`vknn_benchmark config.json` runs entirely on the device against device-local paths — `run.py`
just stages files and writes that flat config. Run `convert` standalone with
`run.py convert model.onnx model.vxm --fp16 [--fuse-se ...] [--on host|device]`.
