# ADR-0013: automatic input-shape resolution from symbolic dim bindings

## Status
Accepted (2026-07-08). Extends ADR-0012 §1 (dynamic shapes as declared plan buckets): the per-tensor
`--shape` mechanism stays, and this adds a symbol-level way to declare the same shapes.

## Context
ADR-0012 resolves a graph input's dynamic (negative) dims from a per-tensor declared shape
(`--shape NAME=D0xD1x...` / `Config::inputShapes`), falling back to the batch axis, with an
undeclared dynamic non-batch axis a hard compile error. That contract is correct but scales poorly on
a model with many inputs that all share a few dimensions. A with-past transformer decoder
(Qwen2.5-Coder-0.5B, exported by optimum) has 51 inputs — `input_ids`, `attention_mask`,
`position_ids`, and 24 layers × {key, value} — every one of which is a function of just three symbolic
dimensions: `batch_size`, `sequence_length`, and `past_sequence_length`. Declaring a decode plan meant
writing 51 `--shape` flags, each a value re-derived by hand from those three numbers (including the
compound `attention_mask` axis `past_sequence_length + sequence_length`).

The ONNX graph already carries this structure: each input axis is either a concrete `dim_value` or a
symbolic `dim_param` (a name, or a small expression the exporter emits as a string). The importer was
discarding the `dim_param`, collapsing every symbolic axis to `-1` and losing the name. Reconstructing
the per-tensor shapes by hand is exactly the arithmetic the graph already encodes.

## Decisions

### 1. Retain the symbolic dim names on the input descriptor
`parseValueInfo` keeps each graph-input axis's `dim_param` string, and the graph builder stores it on
the input `TensorDesc::dimParams` (parallel to `shape`, empty where the axis is concrete). The field is
compile-time only and is **not** serialized into a `.vxm` — a compiled model already has concrete
shapes, so nothing downstream or on-device changes.

### 2. Resolve dynamic axes by evaluating dim_param expressions against a few bindings
`inferShapes` takes a `bindings` map (`Config::dimBindings`, `PassOptions::dimBindings`,
`vknn_compile --dim NAME=VALUE`) keyed by `dim_param` name. For each still-dynamic axis it evaluates
the axis's `dim_param` — a bare symbol, an integer literal, or a `+`/`-`/`*` combination (a small
recursive-descent evaluator, `src/import/dim_expr.h`) — against the bindings. The compound
`past_sequence_length + sequence_length` resolves from its two bound symbols; binding three symbols
resolves all 51 Qwen inputs. Precedence per axis: a per-tensor `--shape` declaration overrides a
binding; else the binding evaluates; else the leading (batch) axis falls back to `batch`; else it is an
error.

### 3. Aggregate the unresolved-axis error and list the free symbols
A dynamic non-batch axis that no declaration or binding resolves is still a hard error (never a silent
1×1 plan, per ADR-0012), but the message is now aggregated across every input and lists the unbound
`dim_param` symbol names to bind, instead of failing per tensor. `vknn_compile --list-dims` prints the
free symbols without compiling.

## Consequences
- A dynamic decoder compiles with a couple of `--dim` bindings instead of one `--shape` per tensor; the
  `.vxm` is **byte-identical** to the per-tensor form (auto resolution reconstructs the same shapes).
  Verified on Qwen2.5-Coder-0.5B: `--dim sequence_length=1 --dim past_sequence_length=256` reproduces
  the 51-`--shape` plan byte-for-byte, 932/932 nodes Vulkan, and the device greedy stream matches the
  HF reference token-for-token.
- A fully-static model (e.g. the YoNoSplat 8-view encoder) and a dynamic-batch CNN are unchanged and
  need no `--dim`/`--shape`; their compiled bytes are identical to before.
- `--bucket` segments accept `dim:NAME=VALUE` bindings alongside `NAME=DxD` shapes, so a prefill and a
  decode bucket differ only by their bound `sequence_length` / `past_sequence_length`.
- The expression grammar is deliberately small (`+ - *` over symbols and literals). An exporter that
  emits a richer expression (division, functions) falls back to the aggregated error, and the tensor
  can still be pinned with `--shape`.
