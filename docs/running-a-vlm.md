# Running a VLM on VKNN

VKNN runs **SmolVLM2-2.2B** — a SigLIP vision tower + pixel-shuffle connector feeding a
24-layer Llama-style text decoder (MHA, 32 query and 32 key/value heads, head dimension 64,
49280-wide vocabulary, `rope_theta = 130000`) — **entirely on the GPU with zero CPU compute
fallbacks**, shipped as **one multi-graph `.vxm`**. The file carries four plan buckets over a
single content-deduped weight pool — the token-embedding lookup is fused into the decoders at
compile time, so the decoder buckets take `input_ids` directly:

| Bucket | Graph | Shapes |
|---|---|---|
| 0 | vision encoder | `pixel_values 1x3x384x384` → `image_hidden_states 1x81x2048` |
| 1 | decoder, text prefill | `input_ids 1x128`, `seq = 128`, `kv_len = 640` (512 cache slots + 128 new) |
| 2 | decoder, image prefill | the text-prefill inputs + `image_hidden_states 1x81x2048` + `image_positions 1x81x2`; an on-GPU `ScatterND` splices the feature rows over the image-token rows |
| 3 | decoder, decode | `input_ids 1x1`, `seq = 1`, `kv_len = 513` (512 cache slots + 1 new) |

`Session::run` dispatches each call to the bucket matching its **bound input names + shapes**,
so one session serves the whole pipeline: encode the image once, prefill the prompt in one
pass (the fused embedding lookup runs inside the decoder; an image prompt binds the vision
features and their row positions, landing on the image-prefill bucket whose on-GPU `ScatterND`
splices the 81 image-embedding rows over the prompt's image tokens), then decode token by
token. The compiled model is published at
[hf.co/katolikov/SmolVLM2-2.2B-vknn](https://huggingface.co/katolikov/SmolVLM2-2.2B-vknn) (4.5 GB fp16;
running it needs **~4.6 GB of GPU-addressable memory**).

## 1. Export three graphs

The official optimum export is unusable here because of a data-dependent `NonZero` (its
`com.microsoft` GroupQueryAttention / RotaryEmbedding contrib ops, by contrast, lower to
primitive subgraphs at import — `src/import/lower_ort_contrib.cpp`). The working recipe exports three
plain-`ai.onnx` graphs from the HF checkpoint with `torch.onnx.export(dynamo=False)`, opset 17,
and `attn_implementation="eager"` (SDPA exports as a fused op; eager gives the plain
MatMul/Softmax attention VKNN already runs full-GPU):

- **`vision.onnx`** — the SigLIP tower + connector at a fixed single 384×384 tile.
  `Idefics3VisionEmbeddings` computes NaViT variable-resolution position ids with
  `torch.bucketize` + boolean indexing — the source of the export's only `NonZero`. Every
  patch of a fully-valid 384×384 tile is real, so the position ids collapse to
  `arange(num_patches)` and the attention mask to all-ones; exporting with a constant
  `arange` position-id buffer is **bit-identical** for this input and removes the data-dependent
  ops entirely. Output: 81 image-embedding rows per tile.
- **`embed.onnx`** — the token-embedding gather alone, `input_ids [1, seq]` →
  `inputs_embeds [1, seq, 2048]`.
- **`decoder.onnx`** — the text decoder on **`inputs_embeds`** (not `input_ids`), which
  removes the data-dependent `NonZero`/`ScatterND` splice from the export; `vknn_compile`
  fuses the embedding lookup back into the decoder and reinstates the image/text merge as a
  shape-static on-GPU `ScatterND` (see below). The 4-D additive `attention_mask [1, 1, seq, kv_len]` is a **graph input** the host
  builds (see below). The KV cache is a fixed 512-slot window baked into the past shapes
  (`past_key_values.*.{key,value} [1, 32, 512, 64]`); `seq` and `kv_len` stay symbolic
  `dim_param`s, so **one traced graph** serves both the prefill and the decode plan through
  compile-time `--dim` bindings.

## 2. Compile one multi-graph `.vxm`

The `--graph` form of `vknn_compile` compiles each source graph into its own bucket — the same
file may repeat at different shapes — over one shared weight pool, then fuses the cross-bucket
hand-offs before saving (see below)
([config.md](config.md#the-vknn_compile-flags)):

```sh
build-host/vknn_compile smolvlm2-2.2b-fp16.vxm --fp16 \
  --graph "export/vision.onnx" \
  --graph "export/embed.onnx;dim:seq=128" \
  --graph "export/embed.onnx;dim:seq=1" \
  --graph "export/decoder.onnx;dim:seq=128;dim:kv_len=640" \
  --graph "export/decoder.onnx;dim:seq=1;dim:kv_len=513" \
  --support-report smolvlm_support.json
```

`--support-report` writes one report per bucket (`smolvlm_support.json` for bucket 0, then
`smolvlm_support.bucketN.json`) — the 0-CPU-fallback oracle. Every summary reads
`{"vulkan": <total>, "reasons": {}}`: 632 + 1 + 1 + 852 + 756 nodes, all Vulkan (one report per
source graph, written before boundary fusion). The compiler then fuses the cross-bucket
hand-offs (`fuseBucketBoundaries`): each embedding graph is merged into the decoder that
consumes its `inputs_embeds` — the decoders take `input_ids` directly and the standalone embed
buckets disappear — and an image-capable copy of the prefill decoder is added whose on-GPU
`ScatterND` splices the vision features in, so the saved file carries the four buckets above.
The three decoder buckets share one ~3.4 GB weight set on disk and one GPU copy at run time: the device
weight pool keys uploads by **payload content digest**, so the prefill and decode plans —
whose fused node lists differ — still deduplicate every unchanged tensor
([ADR-0014](adr/0014-multi-graph-vxm.md)).

A calibration-free **int4** build sits next to the fp16 file. Re-quantizing the compiled fp16
`.vxm` with `vknn_compile smolvlm2-2.2b-fp16.vxm smolvlm2-2.2b-i4.vxm -Os --quant-samples 0`
rewrites every MatMul weight to int4 under a per-layer error guard — the fp16 embedding table is
left as-is — across all four buckets over the shared initializer pool. The result,
`smolvlm2-2.2b-i4.vxm` at ~1.35 GB (about a third of the 4.5 GB fp16 file), runs the same GPU
pipeline with 0 CPU fallbacks and needs roughly **one-third the GPU-addressable memory** of the
fp16 build. Answers stay grounded in the image, with slightly less detail than fp16 — the
expected weight-only-int4 envelope. It is published alongside the fp16 file at
[hf.co/katolikov/SmolVLM2-2.2B-vknn](https://huggingface.co/katolikov/SmolVLM2-2.2B-vknn).

## 3. Build the runner and push everything

`examples/llm/vlm.cpp` builds as `vknn_vlm` (Android build — the host build has no Vulkan
backend):

```sh
./build.sh --android          # builds build-android/vknn_vlm

SERIAL=<adb-serial> ; DDIR=/data/local/tmp/vknn/smolvlm2
adb -s $SERIAL shell "mkdir -p $DDIR"
adb -s $SERIAL push build-android/vknn_vlm smolvlm2-2.2b-fp16.vxm $DDIR/
```

## 4. Run

**`vknn_vlm`** reads commands on stdin: `i <path>` loads a raw fp32 `[1,3,384,384]` pixel file
and runs the vision bucket; a line of whitespace-separated token ids is one prompt turn. The
decoder buckets take `input_ids` directly (the embedding lookup is fused in at compile time);
an image prompt additionally binds the vision features and their row positions, dispatching to
the image-prefill bucket whose on-GPU `ScatterND` overwrites each image-token row with its
feature row. The prompt prefills in one pass (right-padded to the S = 128 window with the EOS
id; pad rows are masked and never folded into the KV cache), and greedy/sampled decode streams
token ids to stdout, `END` per turn. During decode the KV cache is engine-resident by default
(`Session::linkOutputToInput`): each step binds only `input_ids`, the mask, and the positions,
and the engine folds the new row in place; `--no-kv-link` keeps the host cache loop everywhere
with the same token stream. The KV cache and the absolute position persist across turns.

```
vknn_vlm model.vxm [--backend vulkan|cpu] [--precision low|normal|high] [--max-tokens N]
         [--temp T] [--top-k K] [--top-p P] [--eos ID] [--image-token ID] [--seed S]
         [--debug-stats] [--no-kv-link] [--timing] [--no-matmul-view-fold]
         [--no-rope-fusion] [--no-fused-attention] [--no-kv-concat-fold]
```

The decoder picks up the same load-time decode fusions as the text-only LLM path — MatMul
view folding, RoPE chain fusion, KV-concat folding, and single-query fused attention that
reads the KV cache through operand-view strides. These are applied at load time and never
change the compiled `.vxm`; `vknn_vlm` carries the matching `--no-matmul-view-fold` /
`--no-rope-fusion` / `--no-fused-attention` / `--no-kv-concat-fold` flags to A/B them. Greedy
selection and sampling run host-side in `vknn_vlm` (the engine-side GPU argmax is a
`vknn_chat` path).

**`examples/llm/vlm_host.py`** is the host front-end: it owns the HF processor (tokenizer +
chat template, and the image-token expansion so the ids match the device splice exactly) and
the image preprocessing (LANCZOS resize to 384×384, rescale 1/255, normalize mean = std = 0.5,
CHW fp32), drives `vknn_vlm` over adb, and streams the reply with TTFT / tokens-per-second.
All model compute stays on the device GPU.

```sh
python3 examples/llm/vlm_host.py --serial $SERIAL --tokenizer smolvlm2-checkpoint/ \
    --image photo.jpg --question "What is in this picture?"
```

Interactive mode (no `--image`/`--question`): each stdin line is a question, and
`/img <path>` loads a new image for the following questions.

## 5. Measured behavior

Measured on device (a current flagship phone GPU), `--precision low`:

- **Correctness:** 80/80 greedy tokens match the fp32 onnxruntime reference of the same
  export, image + text, on device.
- **TTFT 0.85 s** for the 128-token prefill; the one-time vision encode is 0.98 s.
- **Decode 6–7.5 tokens/s.** During decode the KV cache is engine-resident
  (`Session::linkOutputToInput`): each step binds only `input_ids`, the mask, and the
  positions, and the engine folds the new present row into its cache slot in place — the 48
  past tensors never round-trip across the host boundary per token. The prefill bucket keeps
  the host cache flow once per turn, so a turn boundary materializes the device state back
  into the host cache (`Session::readResident`); `--no-kv-link` restores the host loop
  everywhere, with the same token stream. The cache itself is MHA-sized: 192 KiB per token
  (24 layers × 2 × 32 heads × 64 × fp16), 16× a comparable GQA decoder's.

## 6. Precision

The weights are fp16 (`--fp16`); every kernel accumulates in fp32. Two fp16 boundaries matter
in a VLM specifically:

- **The additive mask fill is −1e4 — never `-FLT_MAX`/`finfo.min` or −65504.** The mask
  crosses the fp16 boundary, where `-FLT_MAX` overflows to −inf and `0 * inf = NaN` poisons
  the softmax, and −65504 (the fp16 max) overflows as soon as any score adds to it.
  `exp(-1e4)` underflows to exactly 0, so masked positions contribute nothing and no inf ever
  exists. `vknn_vlm` builds the mask host-side with this fill.
- **`tanh` is evaluated range-safe.** SigLIP's tanh-GELU feeds `tanh` arguments up to ±230,
  and the driver's built-in is `inf/inf = NaN` past |x| ≈ 88.7. The engine clamps the argument
  to ±10 before the built-in — bit-identical for every finite-result argument
  (`shaders/common.glsl`).

The text-only decode path is the same fixed-context loop as an LLM —
[running-an-llm.md](running-an-llm.md) covers it, including the greedy-argmax fp16 caveats.
