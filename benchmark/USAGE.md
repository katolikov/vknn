# Benchmark & validate: npy files + config reference

`run.py` runs one or more **stages** on the device from a single JSON config: convert a model (or
use a ready `.vxm`), feed inputs, save outputs, compare against goldens, and collect everything under
`result/<RUN>/<stage>/`. The on-device executor is `vknn_benchmark` (built from
`benchmark/benchmark.cpp`); `run.py` stages files over `adb` and runs it.

```sh
./build.sh                                                        # host vknn_compile (host-side convert)
python benchmark/run.py run benchmark/configs/example.json        # auto-builds the device binaries first
python benchmark/run.py run benchmark/configs/example.json --run baseline   # name the result dir
python benchmark/run.py run benchmark/configs/example.json --clean          # wipe the device dir first
python benchmark/run.py run benchmark/configs/example.json -v               # + device stdout/stderr
python benchmark/run.py run benchmark/configs/example.json --no-build       # reuse build-android/
```

Per stage, the host prints:
- **timing** — `load` ms and `run median/min/avg/max` ms over the timed iterations,
- **fallbacks** — how many ops did NOT run on the requested backend, and which (a clean release
  run prints `0`),
- every file pushed, the validation lines (`cos= ... PASS/FAIL`), and where results were collected.

A missing model / input / golden file **stops the run immediately** — nothing is silently skipped.
A run that writes no `result.json` is a failure (exit 3), not a silent pass.

## 1. Results layout

Every invocation gets a run directory (`--run NAME`, default = UTC timestamp):

```
benchmark/result/<RUN>/<stage>/
  result.json      # timing, fallbacks, per-output metrics, optional per-op profile
  outputs/         # saved outputs pulled from the device ("save": ["npy", ...])
  logcat.txt       # the run's Android logcat (GPU driver / thermal / OOM messages land here)
  <model>.vxm      # with "pull": ["vxm"]
  <model>.cache    # with "pull": ["cache"] — the unified pipeline/autotune/weight cache
```

## 2. Input / output files

### `.npy` (recommended)
NumPy's array format carries **shape and dtype in its header**. Write one with
`numpy.save("image.npy", arr)`. Reading supports `float32/16/64`, `int64/32/8`, `uint8` (decoded,
then fed in the model input's declared dtype — a uint8 or fp16 input stays native); arrays must be
C-order.

### raw `.bin` / `.raw`
A headerless little-endian **fp32** dump. Because it has no shape, the file must contain exactly the
model input's element count (the shape comes from the model).

### Input forms
```jsonc
"inputs": { "image": "image.npy" }     // map: model-input name -> file
"inputs": [ "in0.npy", "in1.bin" ]     // list: files in model-input order
"inputs": "models/dl3dv"               // DIRECTORY: one input SET, files map by stem name
"inputs": [ "clip1/", "clip2/" ]       // several directories: one input set each
```
A **directory** maps `<stem>.npy|.bin|.raw` to the model input named `<stem>`, and
`<name>_gold.npy` inside it is that set's golden for the output named `<name>`. With several
directories every set runs each timed iteration and is validated against its own goldens
(outputs/metrics are suffixed `_set<i>`).

If a stage has **no `inputs`**, the model runs on zero-filled inputs for a **runtime-only**
measurement: nothing is saved and no goldens are checked.

### Outputs
With `"save"` set, outputs are written on the device and pulled to `result/<RUN>/<stage>/outputs/`:
- `"npy"` — native-dtype `.npy` (uint8 stays uint8, fp16 stays fp16); `"raw"` — native bytes.
- `"png"` — written when the tensor looks like an image (`[..,C,H,W]` or `[..,H,W,C]`, C∈{1,3,4});
  min–max normalised to 0–255. Non-image tensors are skipped.

## 3. Golden comparison metrics

`"golden"` maps an output name to a golden `.npy` (directory inputs carry their own, see above).
`"metrics"` selects which to report (default: all):

| metric | meaning |
|---|---|
| `cosine` | cosine similarity (pass threshold = `tolerance`, default 0.999) |
| `psnr` | peak signal-to-noise ratio, dB (`20·log10(range / RMSE)`) |
| `snr` | signal-to-noise ratio, dB (`10·log10(Σgolden² / Σerr²)`) |
| `relL2` | relative L2 error `‖a−b‖ / ‖b‖` |
| `max` | max absolute difference |

A pass needs `cosine ≥ tolerance` and zero NaNs. A missing or unreadable golden file fails the
stage immediately.

## 4. Config schema

A config is a list of `stages` (each fully independent) plus an optional `defaults` block merged
into every stage. A single-stage config may drop `stages` and put the fields at the top level.

```jsonc
{
  "defaults": {                                // merged into every stage
    "device": { "serial": "" },
    "run":    { "precision": "low", "iters": 10, "warmup": 2 }
  },
  "stages": [
    {
      "name": "resnet50",                      // stage + result-directory name
      "model": "models/resnet50.onnx",         // .onnx (converted first) or .vxm (as-is)

      "convert": {                             // only used for an .onnx model
        "fp16": true,                          // store weights fp16 (default true)
        "opt": 1,                              // optimization level -O0..-O3 (default 1)
        "no_fuse_swish": false, "fuse_se": false,
        "fuse_dwpw": false, "no_fuse_pointwise": false,   // per-fusion overrides
        "out": "resnet50_fp16.vxm"             // device .vxm name (default: <onnx-stem>.vxm)
      },

      "device": {
        "serial": "",                          // adb serial (required with several phones attached)
        "dir": "/data/local/tmp/vknn/bench",   // device work dir
        "clean": false,                        // rm -rf the dir before this stage (or --clean)
        "cooldown": 0                          // seconds to idle before the run (thermal)
      },

      "run": {                                 // on-device execution options
        "backend": "vulkan",                   // vulkan | cpu
        "precision": "low",                    // low | normal (fp16 + selective fp32) | high (fp32)
        "cache": "model.cache",                // unified cache file (default <model>.cache)
        "generate_cache": false,               // untimed warm-up load to populate it first
        "iters": 10,                           // timed iterations -> min/median/avg/max
        "warmup": 2,                           // untimed warm-up runs (default 1 when iters>1)
        "profile": false,                      // per-operator GPU timing into result.json
        "fold_islands": true,                  // false = keep every supported op on the GPU (verification)
        "max_submit_nodes": 500,               // GPU-watchdog submit chunking (0 = single submit)
        "winograd": "auto",                    // auto | on | off (deterministic kernel choice)
        "tuning": "fast",                      // none | fast | heavy (autotune effort)
        "tolerance": 0.999                     // cosine pass threshold
      },

      "inputs":  { "image": "image.npy" },     // see "Input forms" above
      "golden":  { "prob": "prob_gold.npy" },  // for file inputs; directories carry their own
      "metrics": ["cosine", "psnr", "snr"],
      "save":    ["npy"],                      // formats pulled to result/<RUN>/<stage>/outputs/
      "pull":    ["vxm", "cache"]              // also pull the compiled model / its cache
    }
  ]
}
```

## 5. Standalone convert

```sh
python benchmark/run.py convert model.onnx model.vxm [-O 0..3] [--fp32] \
    [--fuse-se] [--fuse-dwpw] [--no-fuse-swish] [--on host|device]
```
`-O` is the optimization level (see `docs/op-coverage.md` § Fusions); `--on device` runs
`vknn_compile` on the phone for models too big to convert on the host.

## 6. Making goldens

`scripts/make_golden.py` runs an ONNX model with onnxruntime on given inputs and writes
`<output>_gold.npy` files + a starter config.

## 7. Troubleshooting

**`bad magic` / `incompatible vknn version` when loading a `.vxm`.** The `.vxm` container format
carries a version word (`VXM3` / `VXM4`). A ready `.vxm` used **as-is** (a fetched or leftover file,
not reconverted) can be stale relative to the runner you built, so the runner refuses it. The `.onnx`
path never hits this — `run.py` rebuilds the host and device binaries and reconverts each run, so the
compiler and runner always match. Fix: point the config at the `.onnx` (drop the ready `.vxm`), or
delete the stale `<model>.vxm` and `<model>.cache` on the device and let it recompile. Always convert
and run with the same build.
