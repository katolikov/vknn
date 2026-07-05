# ADR-0010: One general pointwise-region fusion; ConvGemm lowering

## Status
Accepted (2026-07-05).

## Context
Fusion had grown into seven hand-pattern passes (`fuseActivations`, `fuseResidualAdd`, `fuseSwish`,
`fuseMatMulBias`, `fusePointwiseChains`, plus the experimental `fuseSqueezeExcite`/`fuseDwPw`), each
matching one shape of subgraph with its own metadata (`fusedAct`, `fusedResidual`, `fusedBias`,
`pw_steps`). Three structural problems drove this redesign:

- **Coverage stops at single-consumer chains.** `fusePointwiseChains` merged only a maximal
  single-consumer per-element chain; any fanout broke it. On a representative frame-interpolation
  graph, one 99-op elementwise mesh fragmented into 39 chains (30 standalone dispatches) that a
  single kernel could compute; DenseNet-class pre-activation BatchNorms never fused at all.
- **The narrow passes were not byte-exact.** `fusedAct` (for SiLU/HardSwish), `fusedResidual`, and
  `fusedBias` all operate on the producer's unrounded fp32 accumulator, so their fused output was
  not byte-identical to the unfused graph — and because those passes ran unconditionally, the
  fused==unfused gate never covered them.
- **Overlapping machinery.** Each pattern needed its own matcher, node metadata, and kernel plumbing.

## Decision
1. **One fusion pass.** `fusePointwiseChains` grows each maximal same-shape per-element region —
   `Binary/Add/Unary/Clip/Relu/PRelu/Where/Greater/GreaterEqual/Equal`, connected through def-use
   edges in either direction, fanout included — and emits it as a single fused unit: a producer
   store epilogue (`pwEpilogueCapable`, now including `Concat` and `ConvGemm`) or a standalone
   `FusedPointwise` node. Residual Adds, swish diamonds, MatMul bias-Adds, and activations are all
   cases of this pass; `fuseResidualAdd`, `fuseSwish`, and `fuseMatMulBias` are deleted, and
   `fuseActivations` survives only as the prerequisite of the experimental SE/DwPw matchers (O2+).
2. **A register-file VM.** `pw_steps` records are 8 ints (kind, code, srcA, srcB, srcC, dst, bcast,
   bcastSrc): sources name the accumulator, the unit's entry value, one of 4 registers, or a tensor
   operand; `select` (Where) and `load` join the step kinds; a unit holds up to 16 steps, 6 operands,
   and 4 extra output streams (`pw_outs`) that export fanned-out intermediates, each stored
   TO_STORE-rounded. Regions are kept convex; budget overruns split a region into its largest
   fitting prefix.
3. **Byte-exactness is the gate, per step.** The unit's entry value is the producer's
   already-rounded store and every step result passes TO_STORE, reproducing each fp16 store of the
   unfused graph bit for bit in the same order. The only inline exception is a lone Relu — or a Clip
   whose bounds round-trip fp16 exactly — folded onto a Conv/Gemm `fusedAct`, which is byte-safe
   because a monotone clamp with representable bounds commutes with RTE rounding. Consequence:
   models whose old builds used the fp32-accumulator epilogues shift at the fp16 ulp level, and
   `fused == unfused` (`--no-fuse-pointwise`) is now byte-identical for everything the pass does.
   The runtime keeps reading `fusedResidual`/`fusedBias` for `.vxm` files compiled by older builds.
4. **BatchNorm lowering.** A BatchNorm that `foldBatchNorm` cannot absorb lowers unconditionally to
   a per-channel Mul+Add (host-folded fp32 scale/shift), which the region fusion then merges —
   DenseNet-121 drops from 309 to 185 nodes with every BN+ReLU tail folded into a Concat store.
5. **ConvGemm lowering (fp16-floor, not byte).** `lowerConv` (default on, `--no-lower-conv` to opt
   out) rewrites each non-Winograd, non-1x1, group-1 KxK Conv into a `ConvGemm` node: weights
   repacked `[K][Cout]` at convert time (a pure permutation), one LDS-tiled implicit-GEMM kernel
   (64x64x16 tiles, fp32 accumulation in a fixed chunked order, explicit vknnRte16 stores). Like
   Winograd and `subpixelConvTranspose`, the accumulation order shifts: the gate for this rewrite is
   fp16-floor equivalence to plain Conv plus run-to-run byte determinism, not a byte compare.

## Consequences
- `-O1` is exactly: the general pointwise fusion (byte-exact) + the ConvGemm lowering (fp16-floor).
  `PassOptions::fuseSwish` and `--[no-]fuse-swish` are gone.
- The fused==unfused byte gate covers the entire default fusion surface on both devices, and the
  stand-in model suite (diamonds, register meshes with Where/Greater, PRelu with NC4 pad lanes,
  >8-step chains, pre-activation BN, flat MatMul units with exports, dw+residual blocks) pins each
  mechanism.
- Every op that walks a hosting node's inputs must bound the walk at `pwCoreInputs` (kernels, shape
  inference, the layout classifier, bindEstimate) — appended unit operands are not data inputs.
