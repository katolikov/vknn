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

## Amendment (2026-07-30): global tile core, native-wave pinning, conv consumer

### Native subgroup width instead of a fixed wave32
The routing prerequisite "pinnable 32-wide subgroups" becomes "the device's native compute
subgroup width (32 or 64), pinnable". `CoopmatGemmCaps` carries `subgroupWidth` (from
`VkPhysicalDeviceSubgroupProperties`) and `widthPinnable`; `coopmatSubgroupWidth()` returns the
width every coopmat pipeline and self-check must create with - as specialization constant 0
(`local_size_x_id = 0`) and as `requiredSubgroupSize`. The device's own width is pinned directly;
there is no preference ladder between 32 and 64. RDNA-class drivers run compute at either width,
and the coopmat fragment ops are subgroup-scope, so the tile math is width-independent; staged
kernels stride their panel-fill and scatter loops by `gl_WorkGroupSize.x`.

### Shared staged-tile core: shaders/coopmat_tile.glsl
A shared include provides the 32x32x16 staged tile machinery for kernels whose operands are not
dense row-major fp16 in global memory: fp16 LDS panels `cmA[k][m]` (k-major, so NC4HW4/im2col/
dequant gathers land without a transpose) and `cmB[k][n]`, an fp32 accumulator tile `cmD[m][n]`
(bias/act apply before the single TO_STORE narrowing), and the fragment loop
(`coopTileAccClear` / `coopTileStep` / `coopTileStoreAcc`) as a 2x2 grid of 16x16x16 fragments
per pinned subgroup. Out-of-range panel entries stage as zero, so ragged edges mask only at the
scatter - staged consumers need no %32 shape alignment. The direct-load `coopmat_gemm.comp`
keeps its own body.

### Conv consumer: conv_gemm_cm.comp
The implicit-GEMM conv (`out[m, oc] = sum_k patch(m, k) * Wt[k, oc]`) on the staged core: the
conv_gemm A-panel gather (aligned channel-block fast path, per-element edge decode) fills cmA,
the B panel reads the same `[K][Cout]` "#gemmw" weight cache entry conv_gemm binds. One kernel
serves every routed group-1 conv - 1x1 included, where the gather degenerates to a channel read.
Routing (`coopmatConvRoute`, core/lowp_gemm.h) is deterministic - caps + Hint::CoopmatGemm +
GEMM-view floors (M, N >= 32, K >= 16) - and preempts the Winograd decision and every SSBO race;
fused-residual and pointwise-epilogue chains keep the SSBO kernels. Gated by its own on-device
self-check (`coopmatConvGemmSelfCheckPassed`: ragged 36x22x54 integer conv with a C % 4 != 0
channel count so both A-gather bodies run live, byte-compared to a
host NC4HW4 oracle) memoized per (VkDevice, kernel) alongside the GEMM verdict; the capability
mirror is filled once in `fillCoopmatGemmCaps` (coopmat_check.cpp), shared by MatMul and Conv.

### Survey: remaining coopmat candidates (from the kernel census)
- `matmul_tiled*` prefill GEMMs: DIRECT fit for the aligned subset (already served); ragged
  edges/bias/epilogue need the staged core - candidate follow-up.
- `matmul_tiled_i4/_i8/_lut4` (quantized prefill): dequant-to-LDS already produces a dense fp16
  B panel; the post-barrier micro-kernel is a drop-in `coopTileStep` target - candidate.
- `wino_gemm_fp16`: dense per-position GEMM, but K packs 4 input channels per vec4 lane;
  requires k-scalar staging (or a k-scalar V/U emission) - candidate.
- POOR fits (stay SSBO): every GEMV/decode kernel (M == 1), fused attention decode (M = G <= 8),
  the bit-exact direct-race conv kernels (tap-skip accumulation order is contractual),
  `wino_fused2`/`wino_full` (per-tile granularity), split-K reduces (not contractions).

### Consequences
- Still no cooperative-matrix device in the fleet (both Xclipse phones re-probed 2026-07-30:
  extension absent); both new kernels compile and their routing unit-tests, and the per-kernel
  self-checks are the first-use gate on real hardware. Byte-neutrality on non-coopmat devices
  verified per model against main on both test devices.
- New shader + include: `conv_gemm_cm.comp`, `coopmat_tile.glsl`. New routing surface:
  `coopmatConvRoute`, `coopmatSubgroupWidth`, `CoopmatGemmCaps.subgroupWidth/widthPinnable`
  (replacing `wave32Pinnable`).
