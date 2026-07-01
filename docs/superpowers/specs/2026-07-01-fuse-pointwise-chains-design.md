# Design: generic pointwise-chain **epilogue fusion** into any producer

Status: **proposed (v2 — epilogue fusion)** — supersedes the v1 standalone-kernel design
Date: 2026-07-01
Author: katolikov

## 1. Goal

Reduce inference runtime by fusing a *synchronizable* pointwise chain — a run of per-element-independent
ops (Mul/Add/Sub/Div/Clip/activation/…) — into the **epilogue of whatever op produces the chain's
input**, so the producer applies the chain in registers before its single store. This generalizes the
existing single-activation epilogue (`fuseActivations`/`fuseResidualAdd` already fold one Relu/Clip/Add
into a Conv/Gemm store) into an **N-step chain epilogue that any producer kernel can carry** —
i.e. *any* operator can be the "head" of a fused chain.

When a chain has no fusable producer (its input is a graph input, or the producer feeds more than one
consumer, or the producer already carries an epilogue), the chain is emitted as one **standalone
`FusedPointwise` kernel** — the degenerate "producer = plain load" case. So the same chain-detection
pass yields either an epilogue on a real producer or a standalone fused kernel, and nothing is left
unfused.

Hard requirements (unchanged, standing principles):

- **Generic, chain-shape driven.** Any eligible chain, any model, out of the box. No per-model naming.
  (`[[generic-fix-not-per-model]]`) The epilogue mechanism is one shared code path every kernel opts
  into — not per-op special cases.
- **Bit-exact.** fp32 `maxErr == 0`; fp16 `maxErr == 0` vs the unfused graph, in both precisions.
- **Default on.** `PassOptions::fusePointwiseChains = true`.  (`[[no-env-vars-use-config]]`)
- **0 fallback.** GPU-resident; CPU path exists as oracle + safety net.
- **One op per file / shared helpers.** New standalone op in its own file; the epilogue is a shared
  GLSL include + a shared C++ helper, not copy-paste per kernel.  (`[[one-op-per-file]]`)
- **Measurable win**, confirmed via `run_io --timing/--profile` on device `R5CWB2KWVJY`.

Test set: **CNN suite**, **yonosplat encoder**, **model.json** (frame-interp tail).

### Non-goals

- Fusing reductions/GEMM/conv *as chain members* (they aren't per-element) — but they ARE valid chain
  **heads/producers** (the chain fuses into their epilogue).
- Improving accuracy (the epilogue reproduces the unfused roundings exactly).
- Reworking the existing conv/matmul epilogue *activation* slot — the chain epilogue runs *after* it.

## 2. Background (the three facts that shape the design)

### 2a. The pipeline layer uses push descriptors — appendable bindings, no second set

`vk::ComputePipeline` (`src/backend/vulkan/vk_pipeline.cpp:36-107`) creates a single descriptor set with
`numBuffers` storage-buffer bindings `0..numBuffers-1`, and `dispatch()` (`vk_pipeline.cpp:128-141`)
binds a flat `std::vector<VkBuffer>` at consecutive bindings via `cmdPushDescriptorSet`. **Consequence:**
epilogue buffers can be *appended* to a kernel's existing buffer list — they land at bindings
`[ownCount .. ownCount+K]` automatically. A kernel opts in with `#define PW_EPI_BASE <ownCount>` before
including the shared epilogue GLSL; no second descriptor set, no pipeline-layer change. Specialization
constants are supported (`vk_pipeline.cpp:82-101`) but binding *declarations* are static, so the
epilogue is a separate build variant (a `-DPW_EPI` define, like the existing `-DUSE_FP16` fp16 variant),
selected per node in `prepare()`.

### 2b. Two GPU worlds; the epilogue runs in the producer's world

NC4HW4 (channel-packed, `vec4`/thread: conv, binary, add, relu, gridsample) vs flat (row-major:
flat_binary, clip, gather-family). The producer's output world is decided at **load** by
`gpuFlatNode()`/`insertLayoutConverts` (`passes.cpp:2292`). The epilogue reads its operands in the
producer's world, so its broadcast index math is world-specific: NC4HW4 uses same-shape or `[N,C,1,1]`
channel-broadcast (divide `vec4` index by `HW`, as `binary.comp:15-20`); flat uses per-axis strides (as
`flat_binary.comp:17-23`). Two epilogue helper variants (`pw_apply4` for NC4HW4, `pw_apply` for flat).
Because the epilogue operands become **extra inputs of the producer node**, the existing load passes
(`insertLayoutConverts`, `markFp32`) convert them to the producer's world/precision for free.

### 2c. Bit-exactness is mechanical (double-STORE)

Every kernel computes in fp32 and writes `STORE(x)` (fp16 with forced RTE, `store16.glsl`). Unfused:
producer writes `STORE(vx_act(v))`; the chain reads it back and each chain op writes `STORE(...)`. Fused
epilogue reproduces this exactly:
```
v = vx_act(producer_value, act, lo, hi);
v = float(STORE(v));           // reproduce the producer's original store rounding
for each step: v = <op>(v, ...); v = float(STORE(v));   // reproduce each chain op's store rounding
dst = STORE(v);
```
Result is bit-identical to the unfused graph in fp32 and fp16. With zero steps, `pw_apply` is the
identity, and `STORE(float(STORE(vx_act(v))))` collapses to `STORE(vx_act(v))` — the unfused kernel,
unchanged. The win is deleted intermediate buffers/dispatches/barriers; arithmetic is untouched.

### 2d. Precedent

`fuseActivations`/`fuseResidualAdd`/`fuseMatMulBias` (`passes.cpp:1374/1701/1828`) already fold a
one-step epilogue into Conv/Gemm via `node.fusedAct`/`fusedResidual`/`fusedBias` + a single-consumer
check. This design is the N-step generalization of exactly that pattern, shared across all kernels.

## 3. Architecture

Six pieces. Detect & attach at **convert** (baked into `.vxm`); route world/precision at **load**;
apply in the producer kernel at **runtime**.

### 3a. Node metadata (no new fields, no `model_io` change)

The chain epilogue lives in the producer `node.attr` (auto-serialized via `writeAttr`/`readAttr`):
- `attr["pw_steps"]` : `Ints`, 4/step `[kind, code, operandInputIdx, bcastMode]`
  (`kind` 0=BINARY 1=UNARY 2=ACT; `code` = BinaryType/UnaryType/ActType; `operandInputIdx` = index into
  `node.inputs`, or -1; `bcastMode` 0=same 1=channel 2=general-flat).
- `attr["pw_params"]` : `Floats`, 2/step `[p0,p1]`.

Epilogue operands are **appended to `node.inputs`** after the producer's own inputs (precedent:
`fuseResidualAdd` does `conv.inputs.push_back(residual)`), with `attr["pw_opbase"]` = the first
appended index. `gpuFlatNode`/each op read only their own fixed input indices, so appended operands are
ignored by world/shape logic. The producer's `outputs[0]` is reused as the chain tail's output id, so
consumers already point at it.

**Standalone `FusedPointwise`** (degenerate producer): `OpType::FusedPointwise`, `inputs[0]`=primary,
`inputs[1..]`=operands, same `pw_steps`/`pw_params`, plus `attr["pw_flat"]` (0/1) so
`gpuFlatNode(FusedPointwise)` routes it. Used when no producer can carry the epilogue.

### 3b. Pass `fusePointwiseChains(Graph&)` — convert stage (`passes.cpp`, in `runStandardPasses`)

Runs after the conv-epilogue fusions (so they claim their single-step epilogues first) and before the
const-fold loop. For each maximal single-consumer run of per-element-independent ops (§5):
1. Identify the run's **producer** = the node producing `run.head.inputs[0]`.
2. If the producer exists, feeds only this chain (single-consumer), is **epilogue-capable** (§3d list),
   is in the same world/precision as the chain, and has no epilogue yet → **attach**: set
   `pw_steps`/`pw_params` on the producer, append operands to `producer.inputs`, set `pw_opbase`, reuse
   the tail output, remove the chain nodes.
3. Otherwise → emit a **standalone `FusedPointwise`** node for the run (if length ≥ 2; length-1 runs are
   left alone).

The world of a run is `gpuFlatNode()` of its ops (deterministic at convert from shapes/constants); a run
never crosses a world boundary. `constFold`+`eliminateDeadNodes` (later in the pipeline) clean up.

### 3c. Shared GLSL epilogue `shaders/pw_epilogue.glsl`

Included by every epilogue-capable kernel. Guarded by `PW_EPI` (compiled into the `_epi` variant only).
Declares, at bindings starting `PW_EPI_BASE` (the kernel `#define`s it to its own buffer count): a
**plan SSBO** (`numSteps`, geometry, per-step `{kind,code,opSlot,bcast,p0,p1}` and, flat-only, per-step
operand strides) + `PW_EPI_MAXOP` operand SSBOs. Provides:
```glsl
#ifdef PW_EPI
float pw_apply (float v, int outIdx);  // flat: strided operand index, step loop, round each step
vec4  pw_apply4(vec4  v, int vecIdx);  // nc4:  same-shape or /HW operand, per-lane, round each step
#else
#define pw_apply(v, i)   (v)
#define pw_apply4(v, i)  (v)
#endif
```
Each step does `v = float(STORE(<op>(v, operand, code)))` (§2c). Reuses `vx_binary`/`vx_unary`/`vx_act`
from `common.glsl`, so the CPU oracle and GPU stay identical.

### 3d. Per-kernel change (uniform, ~3 lines) + build variant

An epilogue-capable kernel adds, guarded by `PW_EPI`:
```glsl
#define PW_EPI_BASE 4            // this kernel's own buffer count (its bindings 0..3)
#include "pw_epilogue.glsl"
...
dst[i] = STORE(pw_apply(float(STORE(vx_act(v, pc.act, pc.actLo, pc.actHi))), i));   // was STORE(vx_act(...))
```
`CMakeLists.txt` emits an extra `<name>_epi.spv` / `<name>_epi_fp16.spv` per epilogue-capable shader
(the `-DPW_EPI` combinatoric variant, alongside `-DUSE_FP16`).

**Epilogue-capable kernels** = every op that writes its output one-element-(or vec4)-per-thread, which
is essentially all of them: conv family (`conv`, `conv1x1*`, `dwconv`, `conv3x3_lds`, `conv_reg`,
`wino_*`), `matmul*`/`gemm`/`fc`, pooling (`avgpool*`, `maxpool`, GAP), `softmax`/`layernorm`/`reduce`
(epilogue at the post-reduction store), `gridsample`, `resize`, `add`/`binary`/`relu`/`unary`/`clip`,
the flat gather family, `fused_se`/`fused_dwpw`, `prelu`, `convtranspose`. Ops NOT worth an epilogue
(pure index remaps that only move bytes — `transpose`/`slice`/`concat`/`pad`/`gather`/`depth_to_space`)
can still be chain heads via the **standalone** `FusedPointwise` path if a pointwise tail follows.

### 3e. Shared C++ helper `bindEpilogue()` (`vk_op_common.h`) + per-op wiring

```cpp
// Returns "" or "_epi"; when "_epi", builds+caches the plan buffer and resolves operands.
struct Epilogue { std::string suffix; int extraBuffers = 0; /* plan + operands */ };
Epilogue prepareEpilogue(const Node& node, VkOpEnv& env, bool flatWorld, const Shape& outShape);
void     appendEpilogueBuffers(std::vector<VkBuffer>& bufs, /*cached*/);
```
Each producer op's `prepare()` becomes: `auto epi = prepareEpilogue(node, env, flat, out);
shader(base + epi.suffix, fp16); numBuffers = ownCount + epi.extraBuffers;`. Its `record()` calls
`appendEpilogueBuffers(bufs)` after its own buffers. That is the entire per-op change (the plan build +
operand upload + broadcast-stride computation live once in the helper). `PW_EPI_MAXOP` bounds operand
count; longer chains split (logged).

### 3f. CPU: one central epilogue hook

In the CPU executor loop, after any op's `run()`, if `node.attr` has `pw_steps`, call a shared
`applyPwEpilogue(node, ctx)` that runs the chain in-place on the op's output (fp32; the step logic is
the standalone `FusedPointwiseCpu` body, factored into a shared function). Zero per-CPU-op edits. This
is the oracle and the 0-fallback path.

### 3g. `OpType::FusedPointwise` + shape inference

Add `FusedPointwise` to `op_type.h` + `opTypeName`; `inferShapes` arm: output = primary input shape/
dtype; `gpuFlatNode` arm: return `attr["pw_flat"]`. (Producer-attached epilogues need no shape-infer
change — the producer's output is unchanged.)

## 4. Bit-exactness — see §2c. Golden = the fusion-off GPU output; expect `max|d| == 0` in fp32 and fp16.

## 5. Eligibility

**Chain members** (the pointwise tail): `Binary`, `Add`, `Unary` (all `UnaryType`), `Clip`, `Relu`,
`PRelu`. Same running `[N,C,H,W]`; single world; single storage class; **single-consumer** on every
non-tail intermediate; operands same-shape / channel-broadcast / world-supported broadcast / constant;
within the operand/step budget.

**Chain head / producer** (via epilogue): any epilogue-capable op (§3d) that (a) feeds only this chain,
(b) shares the chain's world & precision, (c) has no epilogue yet. Otherwise the chain becomes a
standalone `FusedPointwise`.

**Breakers** (split the chain): multi-consumer output; world/dtype boundary; reshape/transpose/slice/
concat/pad/gather in the *middle* of a pointwise run (they are heads, not members); `Where`/`Equal`/
`Greater`/`Cast` (excluded from the first cut — candidate extensions).

## 6. Testing & validation

1. **Host** (`tests/test_ops.cpp`, CPU oracle): standalone `FusedPointwise` step-chain values; and
   producer+epilogue end-to-end — build `Conv→Mul→Add→Clip`, run pass off vs on, assert bit-exact fp32
   and expected node collapse. `applyPwEpilogue` covered directly.
2. **Device bit-exactness** (gate): per producer family, compile a probe (e.g. `Conv→chain`,
   `MatMul→chain`, `GridSample→chain`), diff fusion off vs on with `--winograd off --tuning off` at
   `--precision high` (fp32) and `normal` (fp16); assert `cmp` byte-identical (`max|d|==0`).
3. **Perf**: `run_io --timing` (`submit+gpu` drop) + `--profile` (chain rows absorbed into producer
   rows; dispatch count down). Report per model.
4. **No regression**: CNN suite cosine unchanged + runtime ≤ baseline; yonosplat cosine ≥ 0.9999;
   model.json shapes/no-NaN. Guaranteed by `maxErr==0`; tests confirm the pass didn't mis-attach.

## 7. Rollout (staged by baseline profile; mechanism is uniform)

- **Phase 0**: `--profile` baseline → rank producer families by fusable-tail time; drive the order below.
- **Phase A (infra + first family)**: Node encoding, `pw_epilogue.glsl`, `prepareEpilogue`/CPU hook,
  the pass (attach-to-producer + standalone fallback), build variants, `OpType::FusedPointwise`.
  Validate on the **flat elementwise family** (`flat_binary`/`add`/`unary`/`clip`) — it already has a
  1-step act epilogue, so this exercises both the epilogue path and the standalone path with minimal
  new kernel code.
- **Phase B+**: roll the same 3-line epilogue + `prepareEpilogue` across families in profile order —
  conv family → matmul/gemm/fc → pooling → softmax/layernorm/reduce → gridsample/resize → NC4HW4
  binary/add/relu/unary. Each phase: add epilogue to that family, device-validate bit-exact + profile.
- **Final**: `PassOptions::fusePointwiseChains` default-on; `convert/compile.cpp --no-fuse-pointwise`
  A/B flag; recompile all test-set `.vxm`s; full-suite bit-exact + perf report.

## 8. Risks / magnitude

- **Scope.** This touches many kernels (each: 3 GLSL lines + a build variant + a `prepareEpilogue`
  call). The *mechanism* is uniform and shared, but the *rollout* is broad — hence the staged,
  profile-ordered phases and a bit-exact gate per family. Not all families must ship at once; each phase
  is independently valuable and validated.
- **Push-constant vs plan buffer.** The step-table lives in the **plan SSBO**, not push constants, so
  heavy kernels (conv/matmul, already near the PC limit) are unaffected. Operand count bounded by
  `PW_EPI_MAXOP`; longer chains split (logged, no silent cap).
- **Shader-variant blow-up.** Epilogue doubles variants for epilogue-capable shaders (`_epi` × fp16).
  Acceptable (SPIR-V is embedded, build-time only); confirm binary size / build time stay reasonable.
- **Per-kernel store-site correctness.** Each kernel's `outIdx` passed to `pw_apply` must be the logical
  output element (for operand broadcast). The bit-exact device gate per family catches any mismatch.
- **Where the win is.** Conv-dominated CNNs may show little (their pointwise is already conv-folded);
  the win concentrates where standalone pointwise chains and gridsample/resize/softmax tails live
  (yonosplat, model.json). Phase 0 sets honest expectations.
