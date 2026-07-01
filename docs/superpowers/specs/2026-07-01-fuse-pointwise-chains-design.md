# Design: generic `fusePointwiseChains` operator fusion

Status: **proposed** (awaiting review)
Date: 2026-07-01
Author: katolikov

## 1. Goal

Reduce inference runtime by merging *synchronizable* op chains — runs of per-element-independent
operators (Mul/Add/Sub/Div/Clip/activation/…, optionally headed by a gather like GridSample) — into a
single fused GPU kernel, eliminating the intermediate global-memory buffers, dispatches, and pipeline
barriers between them.

Hard requirements (standing project principles):

- **Generic, chain-shape driven.** Fuse *any* eligible chain from any model out of the box. No
  per-model naming, no hand-tuned patterns. (`[[generic-fix-not-per-model]]`)
- **Bit-exact.** fp32 `maxErr == 0` vs the unfused graph; fp16 `maxErr == 0` vs the unfused fp16 graph
  (i.e. the "fp16 floor" is reproduced exactly). Turning fusion on must not shift any model's numerics.
- **Default on.** Enabled by default in the standard pass pipeline. (`PassOptions` flag defaulting true;
  no env vars — `[[no-env-vars-use-config]]`.)
- **0 fallback.** A fused node runs on the GPU with no CPU fallback for the models in the test set; a
  CPU implementation exists as the correctness oracle and as a safety net.
- **One op per file.** New backend ops each get their own source file. (`[[one-op-per-file]]`)
- **Measurable win.** A `submit+gpu` time drop on the target models, confirmed per-op via
  `run_io --timing/--profile`.

Test set (must stay green / improve): **CNN suite** (mobilenetv3/mnasnet/densenet/shufflenet/
efficientnet/inception/yolo), **yonosplat encoder**, **model.json** (frame-interp tail).

### Non-goals

- Fusing reductions/GEMMs/reshape/transpose/concat/pad/softmax/layernorm — these break per-element
  independence and are out of scope.
- Improving accuracy. The fused kernel deliberately reproduces the unfused roundings exactly (see §4);
  it does not accumulate in fp32 across the chain.
- Changing the conv-epilogue fusions (`fuseActivations`/`fuseResidualAdd`/`fuseSwish`/`fuseMatMulBias`).
  Pointwise-chain fusion runs *after* them and only mops up what they did not claim.

## 2. Background (the two facts that shape the design)

### 2a. Two GPU "worlds"; a fused kernel lives entirely in one

Every pointwise op has two code paths, chosen per node by the layout pass flag `TensorDesc::gpuFlat`
(`opIsFlat()`, `src/backend/vulkan/ops/flat_ops.h:13`):

- **NC4HW4** (channel-packed, `vec4`-per-thread): `add.comp`, `binary.comp`, `relu.comp`, and
  `gridsample.comp`. Buffer length `packedElems = N·cBlocks(C)·4·H·W`. Binary broadcasting here is
  limited to same-shape or `[N,C,1,1]` **runtime** operands (`shaders/binary.comp:15-20`).
- **Flat** (row-major): `flat_binary`, `clip` (flat-only), `flat_broadcast`. Buffer length
  `numElements`. Supports arbitrary NumPy broadcast and constant operands (`flat_ops.h:232`).
  `unary` is layout-agnostic (`unary.cpp:21`).

The layout of each op is decided by `gpuFlatNode()` (`src/import/passes.cpp:2292`) — notably a `Binary`/
`Add` goes **flat** if either operand is a constant initializer, or if the broadcast is anything other
than NC4HW4 same-shape / `[N,C,1,1]`. GridSample is **always NC4HW4** (its output is a dense
`[N,C,Hout,Wout]`, one `vec4` per output pixel). So the frame-interp `GridSample→Add→Mul` chain is
NC4HW4; broad generic pointwise coverage (constants, arbitrary broadcast, Clip) is flat.

**Consequence:** we implement fusion for **both** worlds. A single fused chain must not span a world
boundary (that boundary is where the unfused graph has a `ConvertLayout`).

### 2b. Bit-exactness is mechanical

Every elementwise kernel reads `float(store_elem)`, computes in fp32, and writes `STORE(value)` where
`STORE == float16_t` in the fp16 variant, with **round-to-nearest-even forced** by `store16.glsl`
(`spirv_execution_mode RoundingModeRTE`). In the unfused chain, each op writes its result to a global
buffer as fp16 (RTE) and the next op reads it back. A fused kernel reproduces this **exactly** by
rounding every intermediate through `float(STORE(x))` between steps (a no-op in fp32; the identical RTE
narrow-then-widen in fp16). The result is bit-identical to the unfused graph in both precisions; the
only thing removed is the global-memory traffic. (See §4.)

### 2c. Existing precedent to mirror

- Fused OpTypes already exist: `OpType::FusedSE`, `OpType::FusedDwPw` (`include/vknn/op_type.h:55-56`),
  each with a Vulkan op (`ops/fused_se.cpp`, registered via `VKNN_REGISTER_VK_OP`) and a CPU op
  (`src/backend/cpu/ops/fused_se.cpp`, `VKNN_REGISTER_CPU_OP`).
- Fusion passes share one idiom (`fuseResidualAdd` `passes.cpp:1701`, `fuseActivations` `:1374`,
  `fuseMatMulBias` `:1828`, `fuseDwPw` `:2109`): build a `producer[tensor]→nodeIdx` map, match a
  pattern, enforce a **single-consumer** check on the intermediate, rewire consumers of the removed
  node's output to the fused node's output, then erase consumed nodes.
- The flat binary kernel already carries a fused-activation epilogue in its push constant
  (`flat_binary.comp:9-12` `act, actLo, actHi`) — the pattern we generalize to an *N-step* chain.

## 3. Architecture

Five pieces. Placement (per decision): **detect & fuse at convert time**, **route world at load**.

```
convert (compile.cpp → runStandardPasses)          load (session.plan)                 runtime
─────────────────────────────────────────         ───────────────────────────         ─────────
inferShapes                                        insertLayoutConverts                FusedPointwiseOp
… existing fusions …                       ──▶     (gpuFlatNode picks flat/NC4)  ──▶    (flat OR nc4 kernel,
fusePointwiseChains  ── FusedPointwise node ──▶    markFp32 (fp32 frontier)             one dispatch)
… fold loop, DCE, saveVxm …                        pool alloc / plan
```

### 3a. Pass: `fusePointwiseChains(Graph&)` — `src/import/passes.cpp`

Runs in `runStandardPasses` (`passes.cpp:2681`) **after `fuseSqueezeExcite`/`fuseDwPw` (step 8) and
before the const-fold loop (step 9)**, gated by `PassOptions::fusePointwiseChains = true`. Rationale:
run last among fusions so the conv-epilogue passes (strictly better — they remove the intermediate
entirely) claim their Add/Clip/Swish first; we mop up the rest. Shapes are available (`inferShapes` ran
at step 1). Runs before DCE, so orphaned producers are cleaned up for us.

Also invoked by `convert/compile.cpp:55` (it calls `runStandardPasses(g, opt)` and saves the `.vxm`),
so the fusion is **baked into the `.vxm`** — a compile-time optimization done once.

**Eligibility of a single op as a chain member** (all must hold):

1. Op type is per-element-independent: `Binary` (Mul/Sub/Div/Max/Min/Pow/Add), `Add`, `Unary` (the full
   `UnaryType` table), `Clip`, `Relu`, `PRelu`. Chain **head** may additionally be `GridSample`
   (produces one dense output element per thread from a gather of its *input* only).
2. Output shape equals the chain's running shape `[N,C,H,W]` (no shape change mid-chain). GridSample
   head establishes the shape.
3. Every additional (non-primary) operand is same-shape, `[N,C,1,1]` channel-broadcast, a general
   broadcast the target world supports, or a constant. (World-specific; see 3b.)
4. **Single-consumer:** the op's output feeds exactly one downstream node and is not a graph output
   (same check as `fuseResidualAdd` `passes.cpp:1755-1776`). Otherwise the intermediate must be
   materialized and cannot be fused away. The chain *tail* (last op, whose output survives) is exempt.

**Chain growth.** Greedy: start at an eligible head with a single consumer, extend while the next op is
eligible, stays in the same world (per `gpuFlatNode` computed on the original ops — a chain must not
cross a flat↔NC4HW4 boundary), stays in the same fp32/fp16 storage class (do not cross a future
`markFp32` frontier — approximated by not fusing across `ConvertDtype`, which does not exist yet at this
stage; the storage class is uniform within a chain of same-typed activations, so this is a no-op in
practice but documented), and does not exceed the push-constant budget (§3d — split into two fused
nodes if it would; log the split, `[[check-runtime-with-cosine]]` style, no silent cap).

A chain of length 1 is left unfused (no benefit).

**Rewrite** (mirrors `fuseResidualAdd`):

- Create one `FusedPointwise` node. `inputs = [primary, then each step's extra tensor operand in step
  order]` (constants stay as initializer tensor ids — the backend uploads them, exactly as
  `flat::Binary` does via `uploadInit`). `outputs = [chain tail output]`. `name` = head node name +
  `"#pwchain"`.
- Encode the step-table in `node.attr` (auto-serialized to `.vxm` via `writeAttr`/`readAttr`, no
  `model_io` change):
  - `"pw_steps"` : `Ints`, 4 per step — `[kind, code, operandInputIdx, bcastMode]`.
    `kind ∈ {HEAD_GRIDSAMPLE, BINARY, UNARY, ACT}`; `code` = `BinaryType`/`UnaryType`/`ActType`;
    `operandInputIdx` = index into `node.inputs` of this step's second operand (or `-1` for
    unary/act/head); `bcastMode` = 0 same-shape / 1 channel `[N,C,1,1]` / 2 general-flat.
  - `"pw_params"` : `Floats`, 2 per step — `[p0, p1]` (Clip lo/hi, LeakyRelu/Elu/HardSigmoid params;
    unused = 0).
  - GridSample head carries its own attributes (`mode`, `padding_mode`, `align_corners`) copied onto
    the fused node; its grid is `operandInputIdx` of step 0.
- Rewire: every consumer input and graph output pointing at the tail output already points at
  `outputs[0]` (we reuse the tail's output tensor id as the fused node's output), so no consumer
  rewrite is needed beyond that; erase all consumed nodes (`remove` set → rebuild `g.nodes`, exactly
  `passes.cpp:1807-1817`).

### 3b. Layout routing — extend `gpuFlatNode()` (`passes.cpp:2292`)

Add a `case OpType::FusedPointwise:` that returns the world the chain was built for. Because the world
is a deterministic function of shapes + constant-ness (target-agnostic), storing it once at fusion time
keeps the `.vxm` portable. Encode it as `"pw_flat"` (`Int`, 0/1) on the node; `gpuFlatNode` returns it.
The existing pass then sets the output tensor's `gpuFlat`, marks intermediate operand tensors, and
splices `ConvertLayout` around the fused node exactly as for any other op.

### 3c. `OpType::FusedPointwise` + shape inference

- Add `FusedPointwise` to `include/vknn/op_type.h` (after `FusedDwPw`) and to `opTypeName`
  (`src/core/op.cpp`). No ONNX-name mapping (it is synthesized, never imported).
- Add an `inferShapes` arm: output shape = the primary input's shape (GridSample head → grid-derived
  spatial dims, same rule as the standalone GridSample arm at `passes.cpp:293`-style). Dtype = primary
  input dtype.

### 3d. Backend ops

**Vulkan — `src/backend/vulkan/ops/fused_pointwise.cpp`** (`struct FusedPointwiseOp: VulkanOp`,
`VKNN_REGISTER_VK_OP(OpType::FusedPointwise, FusedPointwiseOp)`). `prepare()`:

- Read the step-table from `node.attr`. Decide world via `opIsFlat(node, env)`.
- Upload any constant operands (`uploadInit`/`operandBuf`, like `flat::Binary` and `unary.cpp`).
- Build one push-constant blob (§ layout below) and one `vk::ComputePipeline` on
  `shader("fused_pw_flat", env.useFp16)` **or** `shader("fused_pw_nc4", env.useFp16)`.

`record()` binds `[primary, operand1..K, output]` and dispatches `groups(total, 256)` (flat) or one
`vec4`/thread over `packedElems` (NC4HW4), a single dispatch.

**Push-constant layout** (device supports ≥256 B — the existing `flat::Binary` PC is 152 B):
- Header: `int total, rank, numSteps, numOperands;` then `int outDim[8];` (flat) or `int N,C,H,W,HW;`
  (NC4HW4).
- Per step (≤ `PW_MAX_STEPS`): `int kind, code, operandBinding, bcastMode; float p0, p1;` and, **flat
  only**, `int stride[rank]` for the operand (NC4HW4 needs only `bcastMode`, no stride array — same-
  shape or divide-by-`HW`). This is why the two worlds are two shaders (mirroring `flat_binary.comp`
  vs `binary.comp`).
- `PW_MAX_STEPS` / `PW_MAX_OPERANDS` chosen so the worst case fits ≤256 B (flat: ~6 steps at rank ≤4,
  or fewer at higher rank; NC4HW4: comfortably 8+). The pass splits longer chains. Operand buffers are
  a fixed `PW_MAX_OPERANDS` bindings; unused slots bind the output buffer as a harmless dummy.

**Shaders — `shaders/fused_pw_flat.comp`, `shaders/fused_pw_nc4.comp`** (each auto-emits an `_fp16`
variant via `#include "precision.glsl"`; `store16.glsl` RTE pulled in automatically). Body per thread:

```glsl
// flat variant (nc4 analogous with vec4 + channel-broadcast index)
uint gid = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x;
if (gid >= uint(pc.total)) return;
// decode gid → per-operand indices via pc.outDim[]/stride[] (as flat_binary.comp:17-23)
float acc = float(primary[idx0]);                     // load (fp16→fp32)
for (int s = 0; s < pc.numSteps; ++s) {
  Step st = pc.step[s];
  if      (st.kind == BINARY) acc = vx_binary(acc, float(operand[st.bind][idxS]), st.code);
  else if (st.kind == UNARY)  acc = vx_unary(acc, st.code, st.p0, st.p1);
  else if (st.kind == ACT)    acc = vx_act(acc, st.code, st.p0, st.p1);
  acc = float(STORE(acc));                            // ← reproduce the unfused intermediate rounding
}
d[gid] = STORE(acc);                                  // final store (RTE in fp16)
```

The `acc = float(STORE(acc))` line is the bit-exactness guarantee (§4). The GridSample head (step 0,
NC4HW4 shader only) is inlined by copying the sampling block from `gridsample.comp:33-58` before the
loop, writing into `acc` (a `vec4`).

**CPU — `src/backend/cpu/ops/fused_pointwise.cpp`** (`VKNN_REGISTER_CPU_OP`). The oracle and the
0-fallback safety net. Computes the same chain in fp32 with full NumPy broadcasting (reuse the CPU
binary broadcast logic, `src/backend/cpu/ops/binary.cpp:31-141`) and the CPU `vx_*` equivalents. CPU is
fp32-only, so no intermediate rounding is needed for the oracle; host tests compare CPU-fused vs
CPU-unfused (both fp32 → exact).

## 4. Bit-exactness strategy (decided: round intermediates)

Inside a fused fp16 chain, round every intermediate through the storage dtype: `acc = float(STORE(acc))`
after each step. In fp32 this is a no-op; in fp16 it is the identical RTE narrow-then-widen the unfused
graph performs when it writes/reads the intermediate buffer. Therefore:

- fp32: fused output `== ` unfused output, `maxErr == 0`.
- fp16: fused output `==` unfused fp16 output, `maxErr == 0` (the "fp16 floor" reproduced exactly).

The speedup is entirely from removing intermediate global-memory buffers, dispatches, and barriers — the
arithmetic and rounding are unchanged. This makes validation trivial (golden = the pre-fusion GPU
output; expect `max|d| == 0`) and carries **zero** numeric-regression risk for a default-on pass. We
explicitly do **not** keep fp32 accumulation across the chain (that would change fp16 output).

## 5. Eligibility summary (what is / isn't fusable)

Fusable chain members (per-element-independent): `Binary`, `Add`, `Unary` (all), `Clip`, `Relu`,
`PRelu`; chain **head** may be `GridSample`. Constraints: same running `[N,C,H,W]`; single world; single
storage class; single-consumer on every non-tail intermediate; operands same-shape / channel-broadcast /
world-supported broadcast / constant; within push-constant budget.

Chain **breakers** (never fused, force a chain split): `Reshape`, `Transpose`, `Slice`, `Concat`, `Pad`,
`Softmax`, `LayerNorm`, `Reduce`, `Gemm`, `MatMul`, `Conv`, `ConvTranspose`, pooling, `Gather`,
`ScatterND`, `Where`/`Equal`/`Greater`/`GreaterEqual` (multi-input compare/select — excluded from the
first cut; candidate extension), `Cast` (dtype change — candidate extension), any op with a
multi-consumer output, and any op that would cross a world / dtype boundary.

## 6. Testing & validation

1. **Host unit tests** (`tests/test_ops.cpp`, CPU oracle): add fused-chain cases — build the chain as
   separate CPU ops (Mul→Add→Clip; GridSample→Add→Mul; Unary runs; channel-broadcast Mul), run the
   `FusedPointwise` CPU op on the same graph, assert `expectNear(fused, unfused, 0)`. Extend the pass to
   confirm it produces the expected `FusedPointwise` node counts on small synthetic graphs.
2. **Device bit-exactness** (the gate): for each target model, dump outputs with fusion **off**
   (`--precision high` fp32 and `--precision normal` fp16), then **on**, identical
   `--winograd off --tuning off`. `vknn_benchmark` golden = the fusion-off output. Assert per output
   `max|d| == 0` (fp32) and `max|d| == 0` (fp16), `nan == 0`, `cosine == 1.0`.
3. **Perf** (the point): `run_io --timing` shows a `submit+gpu` drop; `--profile` shows the Add/Mul/…/
   GridSample rows collapsing into one `FusedPointwise` row and a lower dispatch count. Report the delta
   per model. Confirm on device `R5CWB2KWVJY` (Xclipse).
4. **No regression**: full CNN suite cosine unchanged and runtime ≤ baseline
   (`[[check-runtime-with-cosine]]`); yonosplat 6-output cosine ≥ 0.9999; model.json shapes/no-NaN.

Because bit-exactness is `maxErr == 0`, "no regression" is guaranteed by construction; the tests confirm
the pass didn't mis-fuse (e.g. across a boundary) rather than probing tolerance.

## 7. Rollout

- `PassOptions::fusePointwiseChains = true` (default on). `convert/compile.cpp` gains a
  `--no-fuse-pointwise` flag for A/B measurement (mirroring `--no-fuse-swish`).
- Stale `.vxm`s must be recompiled with `vknn_compile` (they predate the new node; loader is
  forward-compatible via the generic attribute path, but they won't contain fused nodes).

## 8. Open risks

- **Where the win actually is.** Most CNN pointwise is already folded into conv epilogues, so the
  standalone-chain win concentrates in flat geometry tails, yonosplat pointwise runs, and the
  frame-interp warp chain. **Implementation step 1 is a `--profile` baseline** to quantify the
  addressable dispatch count before building, so we target real chains and set honest expectations.
- **Push-constant budget** caps chain length per fused node; long chains split into several fused nodes
  (still correct, slightly less optimal). If profiling shows long chains dominate, a follow-up can move
  the step-table into a small uniform buffer to lift the cap.
- **GridSample head** only exists in the NC4HW4 shader; a flat-world GridSample chain (should not occur —
  GridSample is NC4HW4) is simply not fused.
