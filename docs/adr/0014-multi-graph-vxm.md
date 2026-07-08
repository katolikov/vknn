# ADR-0014: multi-graph `.vxm` over a content-keyed device weight pool

## Status
Accepted (2026-07-08). Extends ADR-0012 §1 (dynamic shapes as declared plan buckets) and
ADR-0013 (symbolic dim bindings): a bucket can now come from its **own source graph**, not only
from re-shaping one graph.

## Context
A vision-language model is several graphs sharing one weight set: a vision encoder, a
token-embedding gather, and a text decoder that must be planned at both a prefill shape
(S = 128) and a decode shape (S = 1). Before this ADR a `.vxm` carried exactly one source
graph; the multi-bucket form (`--bucket`, ADR-0012/0013) only re-shaped that one graph.
Shipping a VLM meant several `.vxm` files — duplicating the shared weights on disk and as
separate GPU uploads — and several sessions for the caller to route between.

The device weight pool keyed uploads by `modelTag` (a graph-derived hash) + tensor name. Both
halves of that key break under multi-graph: shape-gated pointwise fusion diverges the node
lists of the S = 128 and the S = 1 plan of the *same* decoder, so any graph-derived tag splits
the pool exactly for the prefill + decode case it exists for (two ~3.4 GB decoder weight copies
exceed a phone's GPU-addressable memory); dropping the tag and keying by name alone would alias
same-named tensors whose stored bytes were shape-baked differently.

## Decisions

### 1. `--graph`: one bucket per source graph, one `.vxm`
`vknn_compile <out.vxm> --graph "FILE.onnx[;NAME=D0x...;dim:NAME2=VALUE;...]" [--graph ...]`
compiles each occurrence into one bucket from its own file with its own shape/dim segments
(the `--bucket` segment syntax), layered over the shared `--batch`/`--shape`/`--dim` fallback.
Occurrences may repeat a file at different shapes or name different files; every bucket lands
in one `.vxm` whose initializer pool is content-deduped (the writer interns pool blobs by
digest). `--graph` and `--bucket` are mutually exclusive, the form takes a single positional
argument (the output — a mixed `model.onnx out.vxm --graph` form would silently overwrite the
source), and a `.vxm` cannot be a graph source (its passes are baked at one shape).

### 2. Dispatch by bound input names + shapes
`Session::run` builds a key from the caller's bound inputs and evaluates it per candidate
bucket. A **homogeneous** multi-shape session (one graph, several shapes) evaluates one key
over the default bucket's inputs — the pre-multi-graph semantics verbatim, including the
positional single-input forgiveness. A **multi-graph** session dispatches named inputs
strictly, and a candidate must bind at least one caller entry, so an all-unnamed run cannot
match an unrelated bucket through its own defaults. `bucketKeys()` and the indexed
`inputInfo(bucket)` / `outputInfo(bucket)` describe each bucket.

### 3. Streamed bucket loading
`loadGraphBin` streams buckets from the file one at a time, so host memory peaks at a single
bucket's weights rather than the file total — loading a 4.5 GB five-bucket VLM costs the same
host peak as its largest bucket. The load walks the whole file (a corrupt later bucket fails
the load rather than surfacing at first dispatch).

### 4. Content-keyed device weight pool
`uploadInit` keys the pool by a 128-bit **content digest of the stored payload** (plus dtype
and bound element count) instead of `modelTag` + name. Content identity survives the
fusion-diverged prefill/decode node lists (one GPU copy for N buckets referencing one
initializer) and refuses the same-name-different-bytes alias that a name key would accept.

### 5. Per-bucket support reports
`--support-report out.json` writes one report per bucket (`out.json` for bucket 0, then
`out.bucketN.json`), keeping the 0-CPU-fallback oracle per plan.

## Consequences
- SmolVLM2-2.2B ships as **one** 4.5 GB five-bucket `.vxm` (vision, embed ×2, decoder
  prefill + decode) and runs full-GPU from a single session
  ([running-a-vlm.md](../running-a-vlm.md)).
- A single-graph `.vxm` is unchanged: one bucket, byte-identical bytes, the same fast run
  path, and the legacy positional forgiveness preserved for one-graph multi-shape sessions.
- The pool digest costs one hash per initializer at upload; in exchange N same-weight buckets
  cost one GPU copy with no reliance on graph identity.
- Cross-bucket weight sharing is by **byte content**, so it survives only transforms applied
  identically in every bucket; a pass that rewrites a tensor's bytes for one shape (prepacking
  differences and the like) correctly forfeits sharing for that tensor rather than aliasing.
