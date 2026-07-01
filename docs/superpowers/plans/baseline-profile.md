# Phase 0 baseline profile — where the fusable pointwise time is

Device `R5CWB2KWVJY` (Xclipse), `vknn_benchmark --profile`, `precision normal` (fp16), fresh VXM2
compiled from ONNX. Per-op-type GPU breakdown (`profile_by_type_ms`). Absolute ms for yonosplat are a
**cold, untuned, per-op-barriered** run (inflated); use the **relative** split for rollout ordering.

## CNNs — conv-dominated, fusable-pointwise 1–8%

| Model | Conv | GAP | Binary | Unary | Add | fusable-pw |
|---|---|---|---|---|---|---|
| yolov8n (20.5 ms) | 82.8% | — | 0.2% | 0.3% | 0.8% | ~1.3% (+Concat/Softmax/Transpose head) |
| efficientnet_b0 (3.4 ms) | 81.8% | 7.0% | **6.5%** | 1.2% | — | ~7.7% (SE-scale Mul) |
| mobilenetv3 (2.3 ms) | 84.6% | 4.1% | 3.0% | 0.9% | — | ~3.9% (SE-scale Mul) |

CNN pointwise is mostly already folded into conv epilogues (`fuseActivations`/`fuseResidualAdd`/
`fuseSwish`). The one real target is the **SE-scale `Mul`** (channel-broadcast `Binary`, effnet 6.5% /
mnv3 3.0%) — captured by the elementwise family (standalone or as a `Unary`/`GAP` epilogue). Marginal.

## yonosplat encoder — MatMul-dominated (transformer)

| Type | % | Type | % |
|---|---|---|---|
| MatMul | 73.4% | Add | 1.9% (pw) |
| Conv | 5.0% | Transpose | 1.8% |
| **Binary** | **4.6% (pw)** | LayerNorm | 1.0% |
| ConvertLayout | 3.8% | **Unary** | 0.9% (pw) |
| Softmax | 3.7% | Clip | 0.0% (pw) |

Fusable-pointwise = **7.4%** (Binary+Add+Unary+Clip). The big lever is folding the pointwise **tail into
the MatMul epilogue** (bias/gelu/residual/scale after the 73% MatMul) and the **Softmax/LayerNorm
epilogues**. The standalone elementwise family already captures the 7.4% as merged chains.

## model.json (frame-interp) — pointwise-dominated (op inventory)

Op histogram (not run on device — the Conv backbone can't be reconnected from the dump, per memory):
**Mul 81, Add 53, Sub 49, Div 26, Reciprocal 21, PRelu 24, Relu 12, Clip 8, Pow 2, Sigmoid 1 → ~274
per-element ops**; plus GridSample 7, Resize 10, ConvTranspose 3, Conv 36, Where 10, Equal/Greater 12.
This model is where fusion pays off most: long pointwise chains (Mul/Add/Sub/Div/Reciprocal runs,
GridSample→Add→Mul warps). The elementwise family (standalone + epilogue) captures the bulk; GridSample/
Resize heads fold the warp chains into the sampler.

## Rollout order (grounded)

1. **Phase A — infra + elementwise family** (Binary/Add/Unary/Clip/Relu/PRelu; flat + NC4HW4; standalone
   `FusedPointwise` + epilogue on elementwise producers). Captures all of model.json's chains,
   yonosplat's 7.4%, CNN SE-Mul. Highest value, and the foundation for every later phase.
2. **Phase B — MatMul/Gemm/FC epilogue.** Yonosplat's biggest lever (fold tail into the 73% MatMul).
3. **Phase C — Softmax/LayerNorm/Reduce epilogue.** Yonosplat (Softmax 3.7%, LayerNorm 1%).
4. **Phase D — GridSample/Resize/ConvTranspose epilogue heads.** model.json warp chains.
5. **Phase E — Conv epilogue.** CNN (marginal; generalizes the existing conv act/residual folding to
   PRelu/Binary tails).
6. **Phase F — Pooling (avg/max/GAP) epilogue.** Minor.
7. **Final — default-on + full bit-exact + perf validation + recompile.**

Standalone `FusedPointwise` (Phase A) already delivers broad wins everywhere; Phases B–F fold chains
into heavy producers for incremental gains, ordered by measured impact.
