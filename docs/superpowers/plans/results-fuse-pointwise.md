# Results — pointwise-chain fusion (standalone), branch `fuse-pointwise-chains`

Shipped: the generic `fusePointwiseChains` pass + standalone `FusedPointwise` GPU kernels (flat + NC4HW4)
+ CPU oracle, default-on, bit-exact. Producer-epilogue (MatMul) is preserved on
`fuse-pointwise-matmul-wip` pending a host-memory fix (see below).

## What landed (commits on `fuse-pointwise-chains`)

| Commit | What |
|---|---|
| `96f81f5` | `OpType::FusedPointwise` + constants + pass decl/flag |
| `7661e47` | CPU `FusedPointwise` op + shared `applyPwEpilogue` + `Attr::getfloats` + tests |
| `e93512a` | `fusePointwiseChains` pass (standalone) + fp32-intermediate guard + layout/shape arms |
| `b4662b0` | `pw_epilogue.glsl` (plan SSBO) + standalone flat + NC4HW4 kernels + `--no-fuse-pointwise` |
| `9eabefe` | central CPU epilogue hook |
| `f10b870` | **fix**: chain primary must be a runtime full-size input (device null-deref) |
| `1abd681` | **fix**: run pass after const-fold + shape resolution; guard empty/int64 (dynamic-shape crash) |

Host tests: **28/28** (incl. `Ops.FusedPw*`, `Passes.FusePointwise*`, `Ops.CpuEpilogueHookOnProducer`,
`Passes.FusePointwiseRuntimePrimary`).

## Device validation — Xclipse 940 (`R5CWB2KWVJY`), `--winograd off --tuning off`

Bit-exactness gate = fused vs `--no-fuse-pointwise`, byte-identical (`cmp`):

| Target | fp32 | fp16 | Notes |
|---|---|---|---|
| flat probe (Mul→Add→Clip, per-channel bcast) | **bit-exact** | **bit-exact** | flat kernel + fusedAct→ACT step |
| NC4HW4 probe (HardSwish→Sigmoid) | **bit-exact** | **bit-exact** | nc4 kernel |
| yolov8n (1 chain) | **bit-exact** | **bit-exact** | only CNN with a chain |
| CNN suite (effnet/mnv3/mnasnet/shufflenet/densenet) | **no-op** | **no-op** | 0 chains → `.vxm` byte-identical to nofuse |
| **yonosplat encoder** (249 chains) | — | **bit-exact 6/6 outputs** | no crash; **~1.4% faster** cold (17307 vs 17554 ms) |

Bit-exactness holds because each fused step rounds its intermediate through `float(STORE(x))`,
reproducing the unfused fp16 (RTE) / fp32 rounding exactly. CNN pointwise is already conv-epilogue-folded,
so fusion is a safe no-op there; the win concentrates on the pointwise-dominated models (yonosplat's 249
chains; model.json's 274 pointwise ops — which can't be run end-to-end).

## Memory

Fusion is memory-**neutral-or-positive** (frees chain intermediates). yonosplat standalone runs within the
same envelope as nofuse (reaches 4803 MB, finishes). No regression.

## Two real device bugs found + fixed (only the real model exposed them)

1. **const-primary null-deref** — `Mul(const, x)` made the constant the chain primary → `env.devBuf(const)`
   null on GPU. Fix: `pwHeadPrimary` selects the runtime full-size input; constants become operands.
2. **shape-subgraph fusion → rank-0** — the pass ran before the const-fold/shape-resolution loop, so it
   grabbed pointwise ops inside the dynamic-shape subgraph (RoPE `Neg`, `attn/Sqrt`), leaving unresolved
   (rank-0) shapes that segfaulted a downstream `Concat`. Fix: run the pass last + guard empty/int64.

## Deferred — MatMul producer-epilogue (`fuse-pointwise-matmul-wip`, `406cef0`)

Numerically correct (mm probe: fp32 maxErr 0; fp16 *more accurate* than unfused — cos 1.0 vs 0.99999976),
but **OOMs host memory** on yonosplat (aborts at ~3000 allocs / 3536 MB while standalone-only survives to
4000 / 4803 MB). Root cause = host-side memory, not device/count; needs Vulkan memory profiling +
pipeline/allocation-sharing across same-config matmul nodes before it can ship. Code + full diagnosis
preserved on the WIP branch and in memory `pw-fusion-correctness-guards`.
