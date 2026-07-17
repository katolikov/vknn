# ADR-0017: Capability-gated cooperative-matrix GEMM; synchronization2 barrier scoping

## Status
Accepted (2026-07-17).

## Context
The engine's GEMM/conv kernels are plain SSBO compute shaders. Desktop-class GPUs expose
`VK_KHR_cooperative_matrix` (hardware matrix units, 16x16x16 tiles on current implementations),
`VK_EXT_shader_float8` (e4m3/e5m2 operands) and accelerated integer dot products; the mobile
targets the engine ships on expose none of these. Separately, every inter-dispatch barrier used
the synchronization1 `SHADER_READ | SHADER_WRITE` access classes, which over-scope the cache
maintenance a pure-SSBO pipeline needs, and every write-after-read hazard on a reused liveness
pool slot paid a full memory barrier.

## Decision

### Cooperative-matrix MatMul path (`Hint::CoopmatGemm`, id 11)
1. **Routing is a deterministic rule, never a race** (per ADR-0009: the coopmat kernels regroup
   the K reduction, so a timing race would let thermal state flip output bits). The rule
   (`core/lowp_gemm.h`) requires: an enumerated subgroup-scope 16x16x16 row for the operand
   types, pinnable 32-wide subgroups (`VK_EXT_subgroup_size_control` with the COMPUTE stage
   bit), the Vulkan memory model feature (coopmat SPIR-V declares `OpMemoryModel Logical
   Vulkan`), fp16 storage, a dense 2-D batch-1 MatMul with no view/bias/epilogue, and
   M, N multiples of 32 with K a multiple of 16 (the kernels carry no edge masking).
2. **A one-time on-device self-check guards the first use** (`coopmat_check.cpp`): an
   asymmetric 32x64x48 exact-integer GEMM must byte-match the host oracle, or the path
   disables for the session. This converts the documented fragment-lane-mapping footgun
   (column-distributed output on some hardware) from a silent-corruption risk into a logged
   fallback, on hardware the maintainers never touched.
3. **The default kind keeps the fp32 accumulator** (fp16 operands). `Mode::Fp8` (e4m3) and
   `Mode::Int8Coop` are opt-in only: they quantize the weight operand on the host (per-tensor
   symmetric scales, codecs in `core/lowp_gemm.h`) and the activation operand on-device per
   dispatch (`lowp_absmax` + `lowp_quant_*`), and are never selected by `Auto`.
4. The hint value joins the cache-variant key (codec map 17 -> 18, `coopmatGemm`), following
   the `SplitKConv` precedent; an older cache file decodes to the same key.
5. A device lacking any prerequisite runs the SSBO kernels byte-identically to the previous
   release at every hint value.

### synchronization2 barrier scoping and WAR elision
6. When the `synchronization2` feature exists, the compute-to-compute barrier narrows its
   access scopes to `SHADER_STORAGE_WRITE -> SHADER_STORAGE_READ | SHADER_STORAGE_WRITE`
   (every kernel operand is an SSBO; the sync1 `SHADER_READ` class also implies sampled-image
   and uniform cache maintenance). Without the feature the sync1 barrier is emitted unchanged.
7. A write-after-read hazard on a reused pool slot emits an **execution-only** barrier
   (`VK_ACCESS_2_NONE` on both sides): ordering without availability/visibility operations.
   Earlier unflushed writes stay tracked, so a later reader still triggers a full barrier.
8. **The sync1 fallback for the WAR case is the full barrier, not the spec-equivalent empty
   `vkCmdPipelineBarrier`**: the target mobile driver drops a zero-memory-barrier sync1
   barrier outright (measured: reused-buffer outputs corrupt on branchy graphs; the sync2
   form with explicit stage masks and `VK_ACCESS_2_NONE` is honored). A sync1-only device
   therefore keeps the pre-elision behavior.

## Consequences
- The coopmat/fp8/int8 kernels compile, validate and unit-test (codecs, routing rule) on every
  build, but had no conforming execution environment when written: **no cooperative-matrix
  device was available, so the kernels are unverified on hardware**. The self-check is the
  first line of defense when one appears; per-model accuracy gates on such a device remain
  future work before any perf or accuracy claim.
- Barrier scoping and WAR elision are byte-neutral (verified per model against the prior
  release on both test devices) and reduce inter-dispatch stalls on dispatch-bound graphs.
- New shaders: `coopmat_gemm{,_fp8,_i8}.comp`, `lowp_absmax.comp`, `lowp_quant_{fp8,i8}.comp`.
  New caps: sync2, subgroup-size-control range/stages, Vulkan memory model, coopmat feature +
  enumerated rows, fp8 features, integer-dot-product acceleration bits (the acceleration bits
  gate any future int8-dot kernel per the standing rule: the feature bit alone only proves the
  opcodes exist, not that they are fast).
