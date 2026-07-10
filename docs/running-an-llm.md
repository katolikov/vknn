# Running an LLM on VKNN

VKNN runs **Qwen2.5-Coder-0.5B** — a qwen2-architecture autoregressive decoder
(24 layers, GQA with 2 key/value heads, head dimension 64, 151936-wide vocabulary)
— **entirely on the GPU with zero CPU compute fallbacks**. Every tensor-compute op
(the embedding gather, all projection and MLP matmuls, RMSNorm, RoPE, GQA attention,
SwiGLU, softmax, residual adds, the KV-cache concats, and the final logits matmul)
runs on the Vulkan backend. The only host code is the BPE tokenizer, the sampling argmax over the
logits readback (greedy runs an engine-side GPU argmax by default), and the token loop.

The decoder is exported **with a KV cache** (`optimum-cli export onnx --task
text-generation-with-past`), which decomposes RMSNorm, RoPE, and the GQA `repeat_kv`
into ops VKNN already runs on the GPU. Keep the external-data weight file
(`model.onnx_data`) beside the `.onnx` — the importer resolves it relative to the
`.onnx` directory.

A **vision-language** decoder runs the same way, with the vision tower, the token
embedding, and the decoder's prefill + decode plans compiled into **one multi-graph
`.vxm`** — see [running-a-vlm.md](running-a-vlm.md).

## 1. Compile a fixed-context decode plan

The decode loop is host-driven around a single **fixed-max-context (fixed-C)** plan:
one static bucket, compiled at a fixed past length `C`, that serves every step. The KV
cache lives in the `past_key_values` buffers, not in a graph re-plan.

Bind the symbolic dimensions with `--dim` — the compiler resolves every one of the 51
inputs from the graph's own `dim_param` structure, so a fixed context of `C = 256`
(`sequence_length = 1`, `past_sequence_length = 256`, `batch_size` defaulted to 1) needs
just two bindings. The compound `attention_mask` axis `past_sequence_length +
sequence_length` evaluates to `257` (the `C` past slots plus the one new token), the past
key/value tensors resolve to `1 x 2 x 256 x 64`, and `input_ids` / `position_ids` to
`1 x 1`. Write a support report — the 0-CPU-fallback oracle:

```sh
./build.sh --convert    # builds build-host/vknn_compile

build-host/vknn_compile qwen-onnx/model.onnx qwen_chat.vxm --fp16 -O1 \
  --dim sequence_length=1 --dim past_sequence_length=256 \
  --support-report qwen_chat_support.json
```

Run `vknn_compile qwen-onnx/model.onnx x.vxm --list-dims` first to print the free
symbolic dims to bind. `--dim` replaces the older per-tensor form (still supported, and
overriding `--dim` for an oddball tensor); the two produce a byte-identical `.vxm`:

```sh
# equivalent, one --shape per tensor (an undeclared dynamic non-batch axis is a hard error)
shapes=( --shape input_ids=1x1 --shape attention_mask=1x257 --shape position_ids=1x1 )
for i in $(seq 0 23); do
  shapes+=( --shape past_key_values.$i.key=1x2x256x64 --shape past_key_values.$i.value=1x2x256x64 )
done
build-host/vknn_compile qwen-onnx/model.onnx qwen_chat.vxm --fp16 -O1 "${shapes[@]}"
```

The full decoder compiles to **100% Vulkan** — the support report's `summary` reads
`{"total": 932, "vulkan": 932, "reasons": {}}` (no `cpu` / `none` node). `examples/llm/chat.cpp`
reads `kv_heads`, `C`, and `head_dim` back from `past_key_values.0.key` at load, so the
decode loop stays model-agnostic.

To benchmark separately, compile a prefill-shaped plan (time-to-first-token) and a
decode-step plan (time-per-output-token) — each is just a different `past_sequence_length`
(and `sequence_length`) binding:

```sh
# prefill: --dim sequence_length=32 --dim past_sequence_length=0   -> mask 1x32,  past 1x2x0x64
# decode : --dim sequence_length=1  --dim past_sequence_length=64  -> mask 1x65,  past 1x2x64x64
```

To prefill the whole prompt in one forward instead of one host step per prompt token,
compile a **multi-bucket** plan whose second bucket takes `input_ids [1, S]` with
`S > 1`. `vknn_chat` selects that prefill bucket automatically and folds the entire
prompt in a single `S`-token pass, cutting time-to-first-token by an order of magnitude;
`--no-prefill` forces the token-by-token path for A/B. Compile a decode bucket (`S = 1`)
and a prefill bucket (`S = 256`) with the `--graph` multi-bucket form:

```sh
vknn_compile out.vxm --fp16 \
  --graph "model.onnx;dim:sequence_length=256;dim:past_sequence_length=1024;dim:total_sequence_length=1280" \
  --graph "model.onnx;dim:sequence_length=1;dim:past_sequence_length=1024;dim:total_sequence_length=1025"
```

`-Os` compiles the same fusion set plus **calibration-free int4 weight quantization** (a
native int4 GPU MatMul) in place of `--fp16`: the instruct model is ~2.4x smaller and
stays coherent. `--quant-samples 0` selects weight-only quantization, which a multi-bucket
compile requires — pair it with the `--graph` form above for an int4 prefill model. The
published `katolikov/qwen-vknn` repo hosts exactly that — an int4 instruct model with the
256-token prefill bucket (~517 MB) — alongside the fp16 single-bucket and fp16-prefill
files.

## 2. Build the runner and push it to a device

`examples/llm/chat.cpp` is on the explicit `_vknn_examples` list in `CMakeLists.txt`, so it
builds as `vknn_chat`. GPU validation is Android-only (the host build has no Vulkan
backend). Re-push the binary after every Android rebuild.

```sh
./build.sh --android          # builds build-android/vknn_chat

SERIAL=<adb-serial> ; DDIR=/data/local/tmp/vknn/qwen
adb -s $SERIAL shell "mkdir -p $DDIR"
adb -s $SERIAL push build-android/vknn_chat qwen_chat.vxm $DDIR/
```

## 3. Run

**`vknn_chat`** reads whitespace-separated prompt token ids on stdin (one turn per line),
runs prefill + decode, and streams generated token ids to stdout (one integer per line,
flushed), then an `END` sentinel per turn. The conversation KV cache and the absolute
token position persist across turns. Sampling is greedy at `--temp 0`, otherwise
temperature + top-k + top-p.

```
vknn_chat model.vxm [--backend vulkan|cpu] [--precision low|normal|high] [--fp32-tensors CSV]
          [--max-tokens N] [--temp T] [--top-k K] [--top-p P] [--eos ID] [--seed S]
          [--chain N] [--no-kv-link] [--no-prefill] [--no-gpu-argmax]
          [--no-rope-fusion] [--no-fused-attention] [--no-matmul-view-fold]
```

The KV cache is engine-resident by default: `vknn_chat` links every `present.*` output to
its `past_key_values.*` input (`Session::linkOutputToInput`), so each token binds only
`input_ids`/`attention_mask`/`position_ids` and the engine folds the new key/value row
into the cache in place — on the Vulkan backend entirely on-device, with no host copy of
the cache in either direction. `--no-kv-link` selects the host-side cache loop instead;
both paths produce the same token stream.

Greedy decode (the default `--temp 0`) reads the next token off the GPU. `vknn_chat`
registers the decode bucket's `logits` output for an engine-side argmax
(`Session::setOutputArgMax`), so the winning id comes back as 8 bytes from a GPU reduction
instead of downloading and scanning the full 151936-wide logits row — the token stream is
identical (first-occurrence argmax). `--no-gpu-argmax` forces the host scan for A/B.
Temperature sampling (`--temp > 0`) keeps the full-row readback.

`--chain N` decodes in **device-resident chains** of N tokens
(`Session::configureDecodeChain`, ADR-0015): the engine records N decode iterations into
one command-buffer sequence — one `vkQueueSubmit` and one fence per N tokens — and feeds
each iteration's token id / position / attention-mask slot forward on-GPU from the
previous iteration's argmax, with the KV fold slots for all N iterations precomputed per
chain. The per-token host tax (input pack, link update, submit call, fence wake)
amortizes by N; the token stream is bit-identical to `--chain 1`. An EOS inside a chain
trims the overshoot (the discarded iterations never print or count in tok/s), and near
the context edge the chain shortens so the fold-slot clamp stays exactly the single-step
loop's. Chaining needs the greedy path with linked KV and the engine argmax; any other
combination (sampling, `--no-kv-link`, `--no-gpu-argmax`) keeps the single-step loop with
one stderr notice.

A one-shot completion, feeding token ids directly:

```sh
echo "750 75698 1445 1648" | adb -s $SERIAL shell \
  "cd $DDIR && ./vknn_chat qwen_chat.vxm --backend vulkan --precision low --max-tokens 32 --temp 0"
```

**`examples/llm/chat_host.py`** is the interactive front-end. It tokenizes with a
HuggingFace `AutoTokenizer`, drives the device binary over `adb` (feeding prompt ids on
stdin, reading generated ids from stdout), detokenizes and streams the completion to the
terminal with a live tokens/s counter, and holds the REPL. All model compute stays on the
device GPU; this process only tokenizes and displays.

```sh
python3 examples/llm/chat_host.py --serial $SERIAL --ddir $DDIR --model qwen_chat.vxm \
    --tokenizer qwen-onnx --precision low --max-tokens 128
```

## 4. Decode fusions

At load, the runner re-collapses the decode step's decomposed chains into fused kernels.
Both passes are **load-time only and never rewrite the compiled `.vxm`** — an existing
model simply runs faster once it loads. Each is on by default.

- **RoPE chain fusion.** The rotate-half chain around each q/k site — the last-axis half
  `Slice`s, the cos/sin table `Gather`s, the rotate products, and the final `Concat` —
  collapses into one `Rope` dispatch per site. `--no-rope-fusion` keeps the decomposed
  chain for A/B.
- **Fused decode attention.** The single-query attention core — `MatMul` scores →
  scale/mask → `Softmax` → `MatMul` context — fuses into one `FusedAttention` kernel per
  layer that reads the GQA key/value cache through per-axis operand-view strides (the
  MatMul operand-view fold), so the `repeat_kv` broadcast is never materialized.
  `--no-fused-attention` restores the decomposed core, and `--no-matmul-view-fold`
  disables the operand-view fold.

Both kernels compute scores and softmax in fp32, so they are **numerically finer** than
the decomposed chain rather than byte-identical — the greedy token stream is unchanged.

## 5. Precision

The plan is compiled `--fp16`, so weights are stored fp16; every kernel **accumulates in
fp32** regardless of the `--precision` storage tier. RMSNorm is a native fused op (the
`lower_rmsnorm` pass folds the decomposed mean/rsqrt/mul chain into one `RMSNorm` node)
that accumulates the variance reduction in fp32, so the fp16-activation path (`--precision
low`) stays numerically faithful through the norm — the prefill-logits argmax matches the
HuggingFace reference.

`--precision low` stores activations fp16; `--precision high` stores them fp32. Because a
greedy token is the argmax over a 151936-wide logits row, two near-tied top logits can flip
under fp16 rounding at a close decision, so a long greedy stream may diverge from an fp32
reference even when each individual forward is correct. Pin the fragile tail to fp32 with
`--fp32-tensors` (the final norm output and the wide `lm_head` logits) — or compile an fp32
plan — when exact long-horizon greedy parity is required. fp16 stores round to nearest even
and saturate a finite result to `±65504` rather than overflowing to `±inf`.
