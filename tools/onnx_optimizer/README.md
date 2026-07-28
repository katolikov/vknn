# onnx_optimizer — bit-exact ONNX graph optimization for VKNN

Optimizes ONNX models **before** they reach `vknn_compile`, with one hard
guarantee: **the optimized model is bit-exact equivalent to the original.**
For every verification input, outputs must match in name set, dtype, shape,
and **raw bytes** (`a.tobytes() == b.tobytes()`; NaNs compare bitwise). There
is no `atol`/`rtol` in the default mode, and verification gates every write —
a model that fails the gate is never written (short of an explicit `--force`,
which prints a loud warning and still exits non-zero).

## Why a pre-import optimizer

VKNN's importer runs its own in-engine passes (`src/import/`), but it ingests
some graph shapes poorly, and every node the importer never sees is work the
engine never has to undo:

- **Reshape/Transpose churn** — the Vulkan backend runs NC4HW4 packed layout;
  every redundant movement op is a layout conversion on the GPU. Movement
  simplification is this tool's priority target.
- **int64 shape arithmetic** — `Shape → Gather → Concat → Reshape` chains from
  framework exporters hit the Cast-from-int64 CPU fallback (`vk_backend.cpp`).
  Folding them to initializers removes the chains entirely.
- **Exporter noise** — `Identity`, inference-mode `Dropout`, no-op
  `Cast`/`Pad`/`Slice`, inline `Constant` nodes, duplicated weight scalars.

## Usage

```sh
pip install -r tools/onnx_optimizer/requirements.txt

# from tools/ (or with PYTHONPATH=tools, or as plain scripts):
python -m onnx_optimizer.optimize \
    -i model.onnx -o model.opt.onnx \
    --report report.json \
    [--disable-pass NAME] [--only-pass NAME] \
    [--verify-samples 16] [--seed 0] \
    [--dim height=224] [--dyn-sizes 1,3] \
    [--target vknn] [--allow-lossy] [--force]

# standalone bit-exact verifier (also usable as a library):
python -m onnx_optimizer.verify a.onnx b.onnx [--samples 8] [--json out.json]
```

Exit codes: `0` verified, `1` verification failed (output not written unless
`--force`), `2` bad invocation / unreadable model. `--list-passes` prints the
pass table. `--dim NAME=N` pins a named symbolic dim for input generation
(e.g. `--dim num_channels=3` for a model exported with a symbolic channel
dim); unpinned symbolic dims cycle through `--dyn-sizes`, with same-named
dims kept consistent.

## Passes (all structural, all on by default)

| pass | what it removes / rewrites |
|---|---|
| `constant-to-initializer` | `Constant` nodes → initializers (payload moved verbatim) |
| `fold-shape-ops` | `Shape`/`Size` over statically-known dims → int64 initializers (dims outside a `Shape` `start:end` window may stay dynamic) |
| `fold-constants` | nodes with all-constant inputs, executed by the **same** deterministic reference runtime used for verification, so folded values cannot drift from the gate; per-node output cap `--max-fold-bytes` (default 16 MiB) |
| `remove-identity` | `Identity` |
| `remove-dropout` | provably inference-mode `Dropout` (constant-false `training_mode`, unconsumed mask) — mirrors `src/import/eliminate_dropout.cpp` |
| `remove-noop-cast` | `Cast` to the same dtype. **Never** collapses precision round-trips: `fp32→fp16→fp32` is not an identity and survives |
| `remove-noop-transpose` | identity-permutation `Transpose` |
| `merge-transposes` | consecutive `Transpose` composed into one; inverse pairs cancel |
| `merge-reshapes` | `Reshape`-of-`Reshape` → one `Reshape` (0-entries in the target resolved from static shapes, never blindly) |
| `remove-noop-reshape` | `Reshape`/`Flatten`/`Squeeze`/`Unsqueeze`/`Expand` whose static output shape equals the input shape |
| `cancel-squeeze-unsqueeze` | `Squeeze`↔`Unsqueeze` pairs that provably reconstruct the input (static shapes equal, or identical non-negative axes) |
| `remove-noop-slice` | `Slice` proven full-range with unit steps (shape equality alone is NOT enough — a `step=-1` slice keeps the shape but reverses data, and survives) |
| `remove-noop-pad` | `Pad` with all-zero pads |
| `remove-single-concat` | single-input `Concat` |
| `cse` | duplicate nodes (same op/attrs/inputs, deterministic ops only) |
| `dedup-initializers` | byte-identical initializers (dtype + dims + payload) |
| `prune-unused-inputs` | graph inputs nothing reads (the one default pass that changes the **signature**; disable to freeze the interface) |
| `dce` | nodes/initializers unreachable from graph outputs, stale value_info |
| `fold-conv-batchnorm` | **LOSSY**, `--allow-lossy` only: BN folded into Conv weights (float64 arithmetic, one final rounding) |

Safety rules shared by every pass: tensor names captured by `If`/`Loop`/`Scan`
subgraphs are never rewritten; initializers listed in `graph.input` are
runtime-overridable and never treated as constants; a node whose output is a
graph output is only bypassed when the producer rename is provably safe.
IR version, opset imports, producer metadata, and `metadata_props` are
preserved; models over the 2 GiB protobuf limit round-trip through ONNX
external data (both when saving and when building verification sessions).

## The engine: nothing unverified ever ships

1. **Per-pass gating with auto-revert.** The model is snapshotted before each
   pass. After any pass that reports changes, a fast bit-exact check
   (`--gate-samples`, default 2) runs against the ORIGINAL model's cached
   outputs. A mismatch — or a crash, or a checker failure — reverts exactly
   that pass, disables it for the rest of the run, and logs it in the report
   under `auto_reverted_passes`. A buggy pass can never silently poison the
   output.
2. **Fixpoint iteration.** The pipeline repeats until a full sweep changes
   nothing (`--max-iterations` guard, default 10), with topological re-sort +
   `onnx.shape_inference` between passes and `onnx.checker` on the result.
3. **Final full verification** gates the write: `--verify-samples` random
   inputs (fixed seed, spread across `--dyn-sizes` for symbolic dims) plus
   edge batteries — zeros, ones, negatives, large magnitudes
   (dtype-aware: 6e4 for fp16), and a NaN/Inf propagation case. Every case the
   original model can run is a **hard gate**, NaN/Inf included; a case the
   original itself cannot run is skipped and reported.

Reference runtime configuration (also what `fold-constants` computes with):

```
onnxruntime CPU, providers=["CPUExecutionProvider"]
graph_optimization_level = ORT_DISABLE_ALL   # ORT must not mask differences
intra_op_num_threads = 1, inter_op_num_threads = 1
```

On any mismatch the report carries the first differing index, both values,
the mismatch count, and the ULP distance (plus max ULP / max abs diff) for
float outputs. The verifier also self-checks determinism (same session, same
feed, twice) and refuses to gate models that are nondeterministic under the
reference config (e.g. training-mode Dropout).

`--allow-lossy` swaps the byte gate for a tolerance gate **for lossy passes
only**: structural breakage still reverts, value drift is allowed and the
report/stdout state the observed max ULP and max abs error. The output is
then explicitly NOT bit-exact and is labeled as such.

## `--target vknn`

Adds an importer-compatibility section to the report, derived at run time
from `src/core/op.cpp` via `tools/check_model_support.py` (the same single
source of truth the repo's other support tools share): ops the importer does
not recognize, custom-domain and control-flow ops, quantized-family notes,
and inputs with symbolic dims beyond the batch dim (vknn compiles fixed
shapes). Reporting only — the transforms stay bit-exact-only; the default
passes already produce the graph shapes the importer prefers.

## Validation on real graphs

All runs: seed 0, 16 random samples + full edge batteries, deterministic ORT
reference as above (onnx 1.17.0 / onnxruntime 1.20.1 / numpy 2.4.6). No pass
was auto-reverted on any of these models.

| model (source) | opset | nodes | changes applied | verification |
|---|---|---|---|---|
| super-resolution-10, sub-pixel CNN ([onnxmodelzoo/super-resolution-10](https://huggingface.co/onnxmodelzoo/super-resolution-10)) | 10 | 12 → 10 | 2× Constant→initializer; the pixel-shuffle Reshape/Transpose chain is real data movement and correctly survives | **PASS** 21/21 byte-identical |
| mobilenetv2_100 ([onnxmodelzoo/mobilenetv2_100_Opset16](https://huggingface.co/onnxmodelzoo/mobilenetv2_100_Opset16)) | 16 | 170 → 100 | 70× Constant→initializer, 68 duplicate initializers (Clip bounds) deduped | **PASS** 21/21 byte-identical |
| mobilenetv4_conv_small ([onnx-community](https://huggingface.co/onnx-community/mobilenetv4_conv_small.e2400_r224_in1k), `--dim num_channels=3 --dim height=224 --dim width=224`) | 12 | 89 → 89 | already-clean export; zero changes, still fully verified | **PASS** 21/21 byte-identical |
| all-MiniLM-L6-v2 transformer encoder ([sentence-transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), 90 MB) | 14 | **780 → 519** | 218× Constant→initializer, 32 CSE merges, 199 initializer dedups, 7 no-op Casts, Shape/ConstantOfShape folds, DCE | **PASS** 20/20 byte-identical (int inputs → no NaN battery) |

Model byte size is dominated by weights and stays ~unchanged; the win is
node count and graph shape. On MiniLM most `Shape` chains depend on the
dynamic batch/sequence dims and are correctly NOT folded (dims pinned at
export time would fold them); the `--target vknn` report flags exactly those
inputs. All four optimized models also pass the standalone
`python -m onnx_optimizer.verify original.onnx optimized.onnx` gate.

## Tests

```sh
python3 -m pytest tools/onnx_optimizer/tests -q
```

Per-pass unit tests (pattern removed + bit-exact through the real engine),
negative tests pinning the strict-mode boundaries (`fp32→fp16→fp32` Cast
chain survives, non-inverse Transpose pairs keep their movement, reversing
`Slice` survives, training-mode/consumed-mask `Dropout` survives, overridable
initializers never fold), engine tests injecting corrupting/crashing passes
and asserting auto-revert, and CLI integration (optimize + verify as
subprocesses, external-data round-trip). Wired into `scripts/ci_host.sh`
(skips cleanly when the Python deps are absent).

## Known limitations

- Only the top-level graph is optimized; `If`/`Loop`/`Scan` bodies are left
  intact and every name they capture is protected from rewriting.
- Models the reference runtime cannot run (unsupported ops, nondeterministic
  nodes) cannot be verified, so they are not optimized.
- Random integer inputs are drawn from {0, 1} so ids/indices stay in range
  for arbitrary models; pass real test vectors through the library API
  (`verify.Verifier`) if you need domain-specific coverage.
- `prune-unused-inputs` changes the input signature (outputs are unaffected);
  disable it if downstream tooling feeds by position.
- Lossy mode currently contains one pass (`fold-conv-batchnorm`); it exists
  to prove the tolerance-gating machinery, not as a performance feature.
