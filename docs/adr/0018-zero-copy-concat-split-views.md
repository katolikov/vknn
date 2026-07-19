# ADR-0018: Zero-copy Concat/Split/Slice through sub-buffer views

## Status
Accepted (2026-07-19).

## Context
Concat, Split, and Slice move bytes without computing anything, yet each costs real GPU work:
Concat records one dispatch per concatenated part, Split one copy (NC4HW4) or gather dispatch
(flat) per output, Slice one gather dispatch. On concat/split-shaped models this is a double loss —
the dispatches themselves, plus a full read+write round-trip of the activation through device
memory. A ShuffleNet-v2 block runs Concat, ChannelShuffle, and Split as separate passes over the
block's whole activation; a DenseNet-style network re-concatenates a growing feature stack every
layer; a YOLO C2f block splits and re-concatenates around every bottleneck.

Whenever the parts tile the whole CONTIGUOUSLY in the stored byte layout, the copy is pure
addressing: part `i`'s bytes inside the concatenated buffer are exactly the bytes the producer
would have written to a standalone buffer, at a fixed offset.

## Decision
The segment planner links eligible Concat parts, Split outputs, and contiguous Slice outputs as
**sub-buffer views**: a view is a real `VkBuffer` of the slice's size bound into the whole's
`VkDeviceMemory` at the slice's byte offset (`vk::Buffer` view constructor). Producers then write
their slice of the concatenation in place, split/slice consumers read theirs in place, and the
Concat/Split/Slice node records nothing — `record()` skips a slice exactly when the buffer
identity proves the planner created its view (same `hazardRoot()`, offset equal to the slice's
computed byte offset), so a refused view degrades to the dispatching path, never to an error.

Bit-exactness is by construction: the bytes and the arithmetic that produced them are unchanged;
only the copies disappear. The transform is structural and model-agnostic — it keys on stored-
layout contiguity, never on model or node names.

### Eligibility (all conditions checked per node, refusal keeps today's path)
- Contiguity. NC4HW4: rank 4, `N == 1`, axis 1, every slice's channel count a multiple of 4
  (slices tile whole channel blocks). Flat: every dim before the concat/split axis is 1. Slice:
  unit steps, leading dims select one index, at most one partial axis, trailing dims full.
- Members (the tensors that become views) must be produced by a GPU node of the same segment, not
  host-read (readback keeps its dedicated `HOST_CACHED` buffer), not fp32-pinned, and in the same
  layout world as the whole.
- A node carrying a fused pointwise epilogue (`pw_steps`) is never folded — its kernel computes at
  the stores, so the copy is not pure (DenseNet's BN+ReLU-riding concats stay dispatched by this
  rule, correctly: their cost is compute, not movement).
- Links compose through a relative-offset union: a Split of a Concat output views straight into
  the concat arena, and a concat-of-prefixes chain (DenseNet shape) collapses into one arena per
  chain, each smaller concat output re-rooted as a prefix view. Inconsistent placements (a tensor
  demanded at two different offsets), cycles, and duplicate parts refuse the site.

### Memory and synchronization rules
- A view-hosting allocation skips `VkMemoryDedicatedAllocateInfo` (dedicated memory may bind only
  its own buffer), and arenas never reuse a pooled slot allocated with the hint. The view binds
  with `vkBindBufferMemory` at the offset; alignment, allocation-size, and memory-type-bit
  requirements are checked at creation and any violation falls back to a plain buffer (the target
  GPUs report 4-byte buffer alignment, so none fires in practice).
- Liveness: a member's uses extend the ARENA's lifetime span, so pool reuse of the arena begins
  only after the last view read.
- Barrier hazard tracking compares `(hazardRoot, byte range)` overlap instead of buffer handles:
  a write through a view and a read through its arena are one hazard, while disjoint slices of one
  arena (parallel Inception branches writing their slots) still record no barrier between them.
  The emitted barriers are the existing global compute/transfer barriers, which are sufficient
  availability/visibility operations for aliased buffer memory.

## Consequences
- Concat/Split/Slice dispatches and their memory round-trips disappear wherever the tiling is
  contiguous; outputs are byte-identical to the dispatching path on every model (gated at
  `--tuning none` and `fast` on both devices, plus adversarial synthetic graphs covering nested
  arenas, duplicate parts, batch>1 refusal, readback edges, and strided-slice refusal).
- The optimization is load-time planning only: nothing is serialized to the `.vxm`, no cache-codec
  or Hint changes, no new knobs.
- Views hold a `shared_ptr` to their arena buffer, so an arena's memory outlives every view; a
  view allocates no `VkDeviceMemory` of its own and does not count against
  `maxMemoryAllocationCount`.
