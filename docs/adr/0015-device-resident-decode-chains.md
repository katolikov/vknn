# 0015. Device-resident decode chains

Date: 2026-07-10. Status: accepted.

## Context

LLM decode runs one engine forward per generated token. With command buffers pre-recorded,
push descriptors baked, engine-resident KV, and all chunks of a run submitted as one
vkQueueSubmit (one fence), the per-token cost still carries a fixed host tax on top of the
GPU span: packing the token id / position / attention-mask inputs (an int64 decode plus
memcpy of a context-plus-one-row mask), the per-token KV-link range update, session
bind/collect, the vkQueueSubmit call, and the fence wake. Measured on the 0.5B int4 decode
bucket this tax is ~4 ms/token against a ~18.5 ms GPU span — the largest remaining
non-GPU cost, and it scales with token count, not model size.

Every input of the next decode step is derivable on-device from state the GPU already
holds: the greedy next-token id is the argmax epilogue's result, the position is the
previous position plus one, the mask gains one valid slot, and the KV append ranges
advance by one row.

## Decision

The decode bucket's GPU segment records a **chain of K decode iterations into one
command-buffer sequence**, submitted with one vkQueueSubmit and one fence per K tokens.

- **Iteration state feedback** runs on-GPU between iterations as head dispatches of the
  next iteration (the same pattern as the resident-link `link_copy` fold):
  - `chain_feedback`: writes the previous iteration's argmax index into the `input_ids`
    boundary buffer, increments the position buffer, and marks the newly valid mask slot.
    Iteration 0 consumes the host-provided inputs unchanged (or, mid-stream, the previous
    chain's last argmax via the same dispatch).
- **KV-link ranges** are precomputed by the host for all K iterations at chain start (the
  host knows the base position; the slot clamp at the context edge is reproduced exactly)
  and indexed per iteration by a push constant into an enlarged ranges SSBO.
- **Argmax results** land in a K-slot result buffer (per-iteration push-constant offset);
  the host reads all K ids after the single fence and trims post-EOS overshoot (bounded
  by K−1 discarded iterations, each a wasted-but-harmless forward).
- The chain length is a Config field (`decodeChainSteps`), default 1 (identical to the
  single-step loop). The chat driver enables chaining only on the greedy path; sampling
  keeps the single-step loop because the sampled token cannot be produced on-device.
- The chunking caps (`maxSubmitNodes` / `maxSubmitBindings`) apply across the whole chain
  recording, so the watchdog / descriptor-cap batching contract is unchanged; the batches
  of one chain ride one submission like any chunked run.

## Consequences

- Greedy decode's host cost amortizes by K; the stream is bit-identical to the
  single-step loop because every per-iteration input is computed by the same rules,
  only on-device.
- A chain crossing the context-window edge keeps the exact clamp semantics via the
  precomputed ranges; the mask update dispatch writes nothing once the window is full.
- EOS is detected K tokens late in the worst case; the discarded iterations cost GPU
  time but never surface in the output stream.
- The full-logits row is overwritten K−1 times inside a chain, so chained decode is
  argmax-only by construction; any consumer needing per-token logits (sampling,
  logprobs) uses chain length 1.
- Re-record triggers (link capacity growth, argmax registration, boundary rebinds)
  re-record the whole chain, unchanged in kind from the single-step segment.
