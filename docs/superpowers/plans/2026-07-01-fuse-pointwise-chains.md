# Pointwise-chain epilogue fusion — Implementation Plan (v2.1, release)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This plan is **self-contained** — every code block needed is inline.

**Goal:** Fuse a synchronizable pointwise chain (Mul/Add/Sub/Div/Clip/activation/Unary/PRelu run) into the epilogue of whatever op produces it — any operator can be the head — or into one standalone `FusedPointwise` kernel when there is no fusable producer. Bit-exact, default on, memory-neutral-or-better, release quality.

**Architecture:** Convert-stage pass (`fusePointwiseChains` in `runStandardPasses`, baked into `.vxm`) attaches a maximal single-consumer chain to its producer as a step-table in `node.attr` (operands appended to `node.inputs`), or emits a standalone `FusedPointwise`. **One shared mechanism**: a plan SSBO + `shaders/pw_epilogue.glsl` (`pw_apply`/`pw_apply4`), used by *both* the standalone kernel and every producer epilogue. Producer kernels opt in with three lines + a `-DPW_EPI` build variant; epilogue operand/plan buffers **append** to the kernel's push-descriptor buffer list (no second descriptor set). Each step rounds through `float(STORE(x))` → bit-identical to the unfused fp16/fp32 graph. One central CPU hook applies the chain after any op (oracle + fallback).

**Tech Stack:** C++17, Vulkan compute (GLSL→SPIR-V; auto `_fp16` + new `_epi` variants), GoogleTest host tests, device validation on `R5CWB2KWVJY` (Xclipse).

**Spec:** `docs/superpowers/specs/2026-07-01-fuse-pointwise-chains-design.md`.
**Baseline:** `docs/superpowers/plans/baseline-profile.md` (rollout order grounded).

---

## Critical review (v2.1) — corrections vs the first draft

A hard re-read of the first plan surfaced seven defects; all are fixed below.

1. **fp32-intermediate bit-exactness bug (would have shipped).** `markFp32` runs at **load** and promotes
   named tensors to fp32 storage for accuracy. The double-STORE reproduces the *fp16* rounding of an
   intermediate — but if the unfused graph kept that intermediate in **fp32**, rounding it to fp16 in the
   fused kernel is *not* bit-exact. **Fix:** the pass consults the same fp32 predicate `markFp32` uses
   (`mixedPrecisionFp32Tensors()` name preset + any compile-time `fp32Tensors`) and **splits the chain at
   any fp32-promoted tensor**, and never fuses across an fp16↔fp32 storage boundary. Documented: fusion
   assumes the compile-time/default precision policy; a custom runtime `fp32Tensors` requires recompile.
   (Guard also lives in memory `[[pw-fusion-correctness-guards]]`.)
2. **Unified on one plan-SSBO mechanism.** The first draft put per-step operand strides in the standalone
   kernel's push constant (up to `kPwMaxSteps × rank` ints → overflows the 256 B PC limit) but used an
   SSBO for the epilogue — two mechanisms. **Fix:** both use one plan SSBO. The standalone kernel is a
   trivial "copy primary" head + the *same* `pw_apply`; there is exactly one chain-execution code path.
3. **Packed-fp16 kernels need explicit widening.** Perf kernels (`conv*_fp16`, `matmul*_fp16`, `fc_fp16`,
   `binary_fp16`) hand-pack `f16vec4` and don't use the `STORE` macro uniformly. Their epilogue must
   `float()`-widen, run the chain in fp32, RTE-store. Phases 7/10 call this out; the STORE-templated
   elementwise kernels (Phase 6) are the easy ones and go first.
4. **Commutative operand order.** A chain value can be `inputs[0]` *or* `inputs[1]` of a downstream
   Binary/Add. **Fix:** for commutative ops (Add/Mul/Max/Min) fuse either order (operand = the other
   input); for non-commutative (Sub/Div/Pow) only when the chain value is `inputs[0]` (else skip). Encoded
   in `pwEncodeStep`.
5. **Pass is O(n), not O(n²).** Precompute the producer map + per-tensor consumer count once; the "next
   consumer" step reads the single consumer from a prebuilt `consumersOf` adjacency, not a full rescan.
6. **Memory budget is explicit** (see the dedicated section below). Fusion is memory-**positive** (frees
   chain intermediates); plan SSBOs are tiny and packed; `_epi` variants are emitted **only** for families
   actually rolled out; constant operands reuse the weight-cache (no duplication).
7. **Self-contained.** All code is inline (the first draft referenced a non-committed "v1 draft").

---

## Shared contract (every phase uses verbatim)

**OpType:** `OpType::FusedPointwise` (in `op_type.h` after `FusedDwPw`); `opTypeName` → `"FusedPointwise"`.

**Constants** (`op_type.h`, `namespace vknn`):
```cpp
constexpr int kPwMaxSteps    = 8;  // steps per fused unit; longer chains split
constexpr int kPwMaxOperands = 6;  // extra tensor operands per unit (primary excluded)
constexpr int kPwMaxRank     = 4;  // flat broadcast rank stored in the plan (rank>4 -> not flat-fused)
```

**Chain encoding** — on the producer node (epilogue) or the standalone node:
- `attr["pw_steps"]`  : `Ints`, 4/step `[kind, code, operandInputIdx, bcastMode]`
  (`kind` 0=BINARY 1=UNARY 2=ACT; `code`=BinaryType/UnaryType/ActType; `operandInputIdx`=index into
  `node.inputs` or -1; `bcastMode` 0=same 1=channel`[N,C,1,1]` 2=general-flat).
- `attr["pw_params"]` : `Floats`, 2/step `[p0,p1]`.
- `attr["pw_opbase"]` : `Int`, first appended-operand index in `node.inputs` (epilogue case).
- `attr["pw_flat"]`   : `Int`, 0=NC4HW4 / 1=flat (standalone node; routes `gpuFlatNode`).
- Operands **appended** to `node.inputs` (precedent `fuseResidualAdd`, `passes.cpp:1779`).

**Plan SSBO layout** (built once in `prepare()`; shared by standalone + epilogue; `struct PwPlanCPU`
mirrored in C++ and the GLSL `PwPlan` block — keep byte-identical):
```
int numSteps, rank, worldFlat, pad;
int outDim[kPwMaxRank];                 // flat: output dims; nc4: [HW,0,0,0]
int step[kPwMaxSteps*4];                // kind,code,opSlot,bcast per step
int stride[kPwMaxSteps*kPwMaxRank];     // flat only: per-step operand strides
float p0[kPwMaxSteps]; float p1[kPwMaxSteps];
```

**fp32 guard predicate** (Phase 3): a chain must not include or cross a tensor that
`markFp32`'s default preset would promote. Factor the preset's name-match into a reusable
`bool pwTensorIsFp32(const Graph&, TensorId)` and split chains there.

**Attr setter idiom** (`tests/test_ops.cpp:17`): `Attr a; a.kind=Attr::Ints; a.ints=v; node.attr.map[k]=a;`

---

## Memory budget (release constraint — must not regress)

Fusion **lowers** peak memory: attaching a chain to its producer (or a standalone node) frees the chain's
intermediate activation buffers (the pool reclaims them). This matters — the 950M model's fp32 `.vxm`
already OOMs the device. Rules enforced by the plan:

- **Plan SSBOs are tiny** (`sizeof(PwPlanCPU)` ≈ a few hundred bytes) and built once per node in
  `prepare()`; if a graph has hundreds of fused nodes, suballocate them from one pooled buffer.
- **No constant-operand duplication:** upload constant operands via `uploadCached`/`operandBuf` keyed by
  tensor id, so a constant shared by several chains uploads once.
- **`_epi` shader variants only for rolled-out families** (they add embedded SPIR-V → binary + startup
  RAM). Do not blanket-emit `_epi` for all ~30 kernels.
- **Reused tail output tensor** = no new allocation for the fused output.
- **Verification (Phase 12):** peak device memory (from the run log / `--profile` alloc summary) must be
  ≤ the unfused baseline on yonosplat and model.json.

---

## Phase 0: Baseline profile — DONE (`docs/superpowers/plans/baseline-profile.md`)

- [x] Order fixed: elementwise family → matmul → softmax/layernorm → gridsample/resize → conv → pooling.

---

## Phase 1: IR plumbing

**Files:** `include/vknn/op_type.h`, `src/core/op.cpp`, `src/import/passes.h`

- [ ] **Step 1: enum + constants** — after `FusedDwPw,` in `op_type.h`:
```cpp
        FusedPointwise, // fused per-element chain (standalone); also the epilogue carried by producers
```
and after the enum, in `namespace vknn`: the three `constexpr int kPw*` from the contract.

- [ ] **Step 2: opTypeName** — `src/core/op.cpp`: `case OpType::FusedPointwise: return "FusedPointwise";`

- [ ] **Step 3: decl + flag** — `passes.h`: `void fusePointwiseChains(Graph &g);` and
  `bool fusePointwiseChains = true;` in `PassOptions`.

- [ ] **Step 4: build host** — `./build.sh --host`. Expected: clean compile.

- [ ] **Step 5: commit**
```bash
git add include/vknn/op_type.h src/core/op.cpp src/import/passes.h
git commit -m "ir: add OpType::FusedPointwise + fusePointwiseChains decl/flag/limits"
```

---

## Phase 2: CPU op + shared `applyPwEpilogue` + `Attr::Floats` + tests

**Files:** create `src/backend/cpu/ops/fused_pointwise.cpp`; `include/vknn/attr.h`,
`include/vknn/attributes.h`, `src/core/model_io.cpp` (add `Floats` if absent); `tests/test_ops.cpp`.

- [ ] **Step 1: `Attr::Floats` support** — verify `Attr` has `std::vector<float> floats;` and
  `Attributes::getfloats`; if not, add them (mirror `ints`/`getints`). Ensure `writeAttr`/`readAttr`
  (`model_io.cpp`) serialize `Attr::Floats` (mirror the `Ints` case) so `.vxm` round-trips the chain.

- [ ] **Step 2: failing test** — add the `runFusedPw` helper (builds a standalone `FusedPointwise` node,
  runs on CPU) and two value checks:
```cpp
static OpOut runFusedPw(const std::vector<int64_t>& xshape, const std::vector<float>& xdata,
                        const std::vector<Init>& operands, const std::vector<int64_t>& steps,
                        const std::vector<float>& params) {
    Graph g; TensorDesc xi; xi.name="x"; xi.shape=xshape; xi.isInput=true;
    TensorId x=g.addTensor(xi); g.inputs.push_back(x); std::vector<TensorId> ids{x};
    for (size_t k=0;k<operands.size();++k){ TensorDesc d; d.name="o"+std::to_string(k); d.shape=operands[k].shape; d.isInitializer=true;
        TensorId id=g.addTensor(d); HostBuffer hb; hb.resizeElems(operands[k].data.size(),DType::Float32);
        for(size_t i=0;i<operands[k].data.size();++i) hb.f32()[i]=operands[k].data[i]; g.initializers[id]=hb; ids.push_back(id); }
    TensorDesc yo; yo.name="y"; yo.isOutput=true; TensorId y=g.addTensor(yo);
    Node n; n.type=OpType::FusedPointwise; n.name="pw"; n.inputs=ids; n.outputs={y};
    { Attr a; a.kind=Attr::Ints;   a.ints=steps;  n.attr.map["pw_steps"]=a; }
    { Attr a; a.kind=Attr::Floats; a.floats=params; n.attr.map["pw_params"]=a; }
    { Attr a; a.kind=Attr::Int;    a.i=1;         n.attr.map["pw_flat"]=a; }
    g.nodes.push_back(n); g.outputs={y};
    Config cfg; cfg.backend=BackendKind::Cpu; auto sess=Session::create(std::move(g),cfg);
    EXPECT_TRUE(sess); if(!sess) return {};
    IOTensor in; in.name="x"; in.shape=xshape; in.dtype=DType::Float32; in.data.resize(xdata.size()*4);
    for(size_t i=0;i<xdata.size();++i) reinterpret_cast<float*>(in.data.data())[i]=xdata[i];
    std::vector<IOTensor> outs; EXPECT_EQ(sess->run({in},outs),Status::Ok); EXPECT_FALSE(outs.empty());
    if(outs.empty()) return {}; const float*o=outs[0].f32();
    return {std::vector<float>(o,o+numElements(outs[0].shape)), outs[0].shape};
}
TEST(Ops, FusedPwMulAddClip) {
    std::vector<int64_t> steps{0,(int)BinaryType::Mul,1,0, 0,(int)BinaryType::Add,2,0, 2,(int)ActType::Clip,-1,0};
    std::vector<float> params{0,0, 0,0, 0.f,10.f};
    auto got=runFusedPw({1,1,2,2},{1,2,3,4}, {{{1,1,2,2},{2,2,2,2}},{{1,1,2,2},{1,1,1,1}}}, steps, params);
    expectNear(got.data,{3,5,7,9});
}
TEST(Ops, FusedPwUnaryChannelMul) {
    std::vector<int64_t> steps{1,(int)UnaryType::Sigmoid,-1,0, 0,(int)BinaryType::Mul,1,1};
    std::vector<float> params{0,0, 0,0};
    auto got=runFusedPw({1,3,1,2},{0,0,0,0,0,0}, {{{1,3,1,1},{10,20,30}}}, steps, params);
    expectNear(got.data,{5,5,10,10,15,15});
}
```

- [ ] **Step 3: run → FAIL** — `cmake --build build-host -j && ./build-host/vknn_tests --gtest_filter='Ops.FusedPw*'`.

- [ ] **Step 4: CPU op + shared applier** — create `src/backend/cpu/ops/fused_pointwise.cpp`:
```cpp
// CPU FusedPointwise + shared applyPwEpilogue: run a per-element step chain in fp32. Oracle + fallback.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <algorithm>
#include <cmath>
namespace vknn {
    namespace {
        float pwBinary(float a,float b,int op){switch((BinaryType)op){case BinaryType::Mul:return a*b;case BinaryType::Sub:return a-b;
            case BinaryType::Div:return a/b;case BinaryType::Max:return std::max(a,b);case BinaryType::Min:return std::min(a,b);
            case BinaryType::Pow:{if(a<0.f&&b==std::round(b)){float r=std::pow(-a,b);return (std::fmod(std::round(b),2.f)!=0.f)?-r:r;}return std::pow(a,b);}
            default:return a+b;}}
        float pwUnary(float x,int op,float a,float b){switch((UnaryType)op){case UnaryType::Sigmoid:return 1.f/(1.f+std::exp(-x));
            case UnaryType::Tanh:return std::tanh(x);case UnaryType::HardSwish:return x*std::clamp(x+3.f,0.f,6.f)/6.f;
            case UnaryType::HardSigmoid:return std::clamp(a*x+b,0.f,1.f);case UnaryType::LeakyRelu:return x>0.f?x:a*x;
            case UnaryType::Elu:return x>0.f?x:a*(std::exp(x)-1.f);case UnaryType::Abs:return std::fabs(x);case UnaryType::Neg:return -x;
            case UnaryType::Exp:return std::exp(x);case UnaryType::Log:return std::log(x);case UnaryType::Sqrt:return std::sqrt(x);
            case UnaryType::Floor:return std::floor(x);case UnaryType::Ceil:return std::ceil(x);case UnaryType::Relu:return std::max(x,0.f);
            case UnaryType::SiLU:return x/(1.f+std::exp(-x));case UnaryType::Erf:return std::erf(x);case UnaryType::Cos:return std::cos(x);
            case UnaryType::Sin:return std::sin(x);case UnaryType::Reciprocal:return 1.f/x;
            case UnaryType::Softplus:return std::max(x,0.f)+std::log(1.f+std::exp(-std::fabs(x)));default:return x;}}
        float pwAct(float x,int act,float lo,float hi){switch((ActType)act){case ActType::Relu:return std::max(x,0.f);
            case ActType::Relu6:return std::clamp(x,0.f,6.f);case ActType::Clip:return std::clamp(x,lo,hi);
            case ActType::HardSwish:return x*std::clamp(x+3.f,0.f,6.f)/6.f;case ActType::SiLU:return x/(1.f+std::exp(-x));default:return x;}}
    }
    // Apply pw_steps/pw_params in-place on node.outputs[0] (already holds the head result). fp32 oracle.
    void applyPwEpilogue(const Node& node, ExecContext& ctx) {
        RtTensor& Y=ctx.t(node.outputs[0]); const Shape& out=Y.shape; int64_t n=numElements(out);
        float* y=Y.host.f32(); size_t rank=out.size();
        const auto& st=node.attr.getints("pw_steps"); const auto& pr=node.attr.getfloats("pw_params");
        int nSteps=(int)(st.size()/4);
        auto bstr=[&](const Shape& s){ std::vector<int64_t> ob(rank,0); int64_t sB=1; size_t off=rank-s.size();
            for(int i=(int)rank-1;i>=0;--i){ int64_t d=(i<(int)off)?1:s[i-off]; ob[i]=(d==1)?0:sB; sB*=d; } return ob; };
        for(int64_t lin=0;lin<n;++lin){ float acc=y[lin];
            for(int s=0;s<nSteps;++s){ int kind=(int)st[s*4],code=(int)st[s*4+1],oi=(int)st[s*4+2];
                float p0=pr[s*2],p1=pr[s*2+1];
                if(kind==0){ const RtTensor& O=ctx.t(node.inputs[oi]); auto ob=bstr(O.shape);
                    int64_t io=0; for(size_t d=0;d<rank;++d){ int64_t strd=1; for(size_t e=d+1;e<rank;++e) strd*=out[e]; io+=((lin/strd)%out[d])*ob[d]; }
                    acc=pwBinary(acc,O.host.f32()[io],code);
                } else if(kind==1) acc=pwUnary(acc,code,p0,p1); else if(kind==2) acc=pwAct(acc,code,p0,p1);
            } y[lin]=acc; }
    }
    namespace { struct FusedPointwiseCpu: CpuOp {
        void run(const Node& node, ExecContext& ctx) override {
            const RtTensor& X=ctx.t(node.inputs[0]); RtTensor& Y=ctx.t(node.outputs[0]);
            float* y=cpu::allocOut(Y,X.shape); const float* x=X.host.f32();
            for(int64_t i=0;i<numElements(X.shape);++i) y[i]=x[i];
            applyPwEpilogue(node,ctx);
        }
    }; }
    VKNN_REGISTER_CPU_OP(OpType::FusedPointwise, FusedPointwiseCpu);
}
```
Declare `void applyPwEpilogue(const Node&, ExecContext&);` in `cpu_backend.h` (used by the Phase-5 hook).

- [ ] **Step 5: run → PASS** — `./build-host/vknn_tests --gtest_filter='Ops.FusedPw*'`.

- [ ] **Step 6: commit**
```bash
git add src/backend/cpu/ops/fused_pointwise.cpp src/backend/cpu/cpu_backend.h tests/test_ops.cpp include/vknn/attr.h include/vknn/attributes.h src/core/model_io.cpp
git commit -m "cpu: FusedPointwise op + shared applyPwEpilogue + Attr::Floats round-trip + tests"
```

---

## Phase 3: Pass — standalone fusion + fp32 guard (host-validated)

**Files:** `src/import/passes.cpp`, `tests/test_ops.cpp`.

- [ ] **Step 1: failing e2e test** — `Mul→Add→Relu`, run pass off vs on, assert bit-exact + one node:
```cpp
#include "import/passes.h"
static Graph makeChainGraph(){ Graph g; TensorDesc xi; xi.name="x"; xi.shape={1,1,2,2}; xi.isInput=true;
    TensorId x=g.addTensor(xi); g.inputs={x};
    auto k=[&](const char*nm,std::vector<int64_t>sh,std::vector<float>d){ TensorDesc t;t.name=nm;t.shape=sh;t.isInitializer=true;
        TensorId id=g.addTensor(t); HostBuffer hb; hb.resizeElems(d.size(),DType::Float32);
        for(size_t i=0;i<d.size();++i) hb.f32()[i]=d[i]; g.initializers[id]=hb; return id; };
    TensorId s=k("s",{1,1,2,2},{2,2,2,2}), b=k("b",{1,1,2,2},{-5,-5,-5,-5});
    TensorId t0=g.addTensor({.name="t0"}),t1=g.addTensor({.name="t1"}),y=g.addTensor({.name="y"}); g.desc(y).isOutput=true;
    Node m;m.type=OpType::Binary;m.subOp=(int)BinaryType::Mul;m.name="mul";m.inputs={x,s};m.outputs={t0};
    Node a;a.type=OpType::Add;a.name="add";a.inputs={t0,b};a.outputs={t1};
    Node r;r.type=OpType::Relu;r.name="relu";r.inputs={t1};r.outputs={y};
    g.nodes={m,a,r}; g.outputs={y}; return g; }
static std::vector<float> runGraphCpu(Graph g,const std::vector<float>&xd){ Config cfg; cfg.backend=BackendKind::Cpu;
    auto sess=Session::create(std::move(g),cfg); EXPECT_TRUE(sess); if(!sess) return {};
    IOTensor in; in.name="x"; in.shape={1,1,2,2}; in.dtype=DType::Float32; in.data.resize(xd.size()*4);
    for(size_t i=0;i<xd.size();++i) reinterpret_cast<float*>(in.data.data())[i]=xd[i];
    std::vector<IOTensor> outs; EXPECT_EQ(sess->run({in},outs),Status::Ok); const float*o=outs[0].f32();
    return {o,o+numElements(outs[0].shape)}; }
TEST(Passes, FusePointwiseBitExact){ std::vector<float> xd{1,2,3,4};
    auto unfused=runGraphCpu(makeChainGraph(),xd);
    Graph fg=makeChainGraph(); inferShapes(fg,1); fusePointwiseChains(fg);
    int fused=0; for(auto&n:fg.nodes) if(n.type==OpType::FusedPointwise) fused++;
    EXPECT_EQ(fused,1); EXPECT_LE(fg.nodes.size(),1u);
    auto got=runGraphCpu(std::move(fg),xd); ASSERT_EQ(got.size(),unfused.size());
    for(size_t i=0;i<got.size();++i) EXPECT_FLOAT_EQ(got[i],unfused[i]); }
```

- [ ] **Step 2: run → FAIL** — `./build-host/vknn_tests --gtest_filter='Passes.FusePointwise*'`.

- [ ] **Step 3: implement the pass (standalone mode + fp32 guard)** — in `passes.cpp` before
  `runStandardPasses` (so `gpuFlatNode`:2292 and `pwTensorIsFp32` are in scope):
```cpp
    static bool pwEligible(const Node& n){ switch(n.type){
        case OpType::Binary: case OpType::Add: case OpType::Unary: case OpType::Clip: case OpType::Relu: return true;
        default: return false; } }
    // Encode op `n` (chain value flows in as `acc`). run=current chain shape. Sets operandTensor for BINARY.
    static bool pwEncodeStep(const Graph& g, const Node& n, const Shape& run, TensorId chainVal,
                             int& kind,int& code,int& bcast,float& p0,float& p1, TensorId& operand){
        p0=p1=0; operand=kNoTensor;
        auto bcastOf=[&](const Shape& s)->int{ if(s==run) return 0;
            if(s.size()==4&&run.size()==4&&s[0]==run[0]&&s[1]==run[1]&&s[2]==1&&s[3]==1) return 1; return 2; };
        auto pickBinary=[&](int codeIn,bool commutative)->bool{
            TensorId a=n.inputs[0],b=n.inputs[1];
            if(a==chainVal){ operand=b; code=codeIn; kind=0; bcast=bcastOf(g.desc(b).shape); return true; }
            if(b==chainVal && commutative){ operand=a; code=codeIn; kind=0; bcast=bcastOf(g.desc(a).shape); return true; }
            return false; }; // non-commutative with chainVal as inputs[1] -> not fusable here
        switch(n.type){
            case OpType::Add:    return pickBinary((int)BinaryType::Add,true);
            case OpType::Binary: { BinaryType bt=(BinaryType)n.subOp;
                bool comm = (bt==BinaryType::Mul||bt==BinaryType::Max||bt==BinaryType::Min);
                return pickBinary((int)bt, comm); }
            case OpType::Unary:  kind=1; code=n.subOp; bcast=0; p0=n.actLo; p1=n.actHi; return true;
            case OpType::Relu:   kind=2; code=(int)ActType::Relu; bcast=0; return true;
            case OpType::Clip:   kind=2; code=(int)ActType::Clip; bcast=0; p0=-3.4e38f; p1=3.4e38f;
                if(n.attr.has("min")) p0=n.attr.getf("min",p0); if(n.attr.has("max")) p1=n.attr.getf("max",p1);
                if(n.inputs.size()>1&&n.inputs[1]!=kNoTensor&&g.isInitializer(n.inputs[1])) p0=initFloats(g,n.inputs[1])[0];
                if(n.inputs.size()>2&&n.inputs[2]!=kNoTensor&&g.isInitializer(n.inputs[2])) p1=initFloats(g,n.inputs[2])[0];
                return true;
            default: return false; } }

    void fusePointwiseChains(Graph& g){
        std::vector<int> producer(g.tensors.size(),-1), consumers(g.tensors.size(),0);
        for(size_t i=0;i<g.nodes.size();++i) for(TensorId o:g.nodes[i].outputs) if(o!=kNoTensor) producer[o]=(int)i;
        for(auto& n:g.nodes) for(TensorId in:n.inputs) if(in!=kNoTensor&&in<(TensorId)consumers.size()) consumers[in]++;
        for(TensorId go:g.outputs) if(go!=kNoTensor) consumers[go]++;
        // single next consumer of tensor t (or -1); prebuilt to keep the pass O(n).
        std::vector<int> nextOf(g.tensors.size(),-1);
        for(size_t j=0;j<g.nodes.size();++j) for(TensorId in:g.nodes[j].inputs) if(in!=kNoTensor&&in<(TensorId)nextOf.size()) nextOf[in]=(int)j;
        auto single=[&](TensorId t){ return t!=kNoTensor && consumers[t]==1 && !pwTensorIsFp32(g,t); };
        std::set<int> removed; int fused=0;
        for(size_t i=0;i<g.nodes.size();++i){
            if(removed.count((int)i)||!pwEligible(g.nodes[i])) continue;
            TensorId prim=g.nodes[i].inputs[0];
            int pp=(prim<(TensorId)producer.size())?producer[prim]:-1;
            if(pp>=0&&pwEligible(g.nodes[pp])&&single(prim)&&!removed.count(pp)) continue; // not the head
            if(pwTensorIsFp32(g,prim)) continue;
            bool wantFlat=gpuFlatNode(g,g.nodes[i]); Shape run=g.desc(g.nodes[i].outputs[0]).shape;
            std::vector<int64_t> steps; std::vector<float> params; std::vector<TensorId> inputs{prim}; std::vector<int> chain;
            int cur=(int)i; TensorId chainVal=prim;
            while(true){ Node& nd=g.nodes[cur];
                if(!pwEligible(nd)||removed.count(cur)) break;
                if(gpuFlatNode(g,nd)!=wantFlat) break;
                if(g.desc(nd.outputs[0]).shape!=run) break;
                if(pwTensorIsFp32(g,nd.outputs[0])) break;                 // fp32-intermediate guard
                if((int)steps.size()/4>=kPwMaxSteps) break;
                int kind,code,bcast; float p0,p1; TensorId operand;
                if(!pwEncodeStep(g,nd,run,nd.inputs[0]==chainVal?chainVal:nd.inputs[0],kind,code,bcast,p0,p1,operand)) break;
                // the value entering nd must be the running chainVal
                if(nd.inputs[0]!=chainVal && !(kind==0 && (operand!=kNoTensor))) break;
                int oi=-1; if(operand!=kNoTensor){ if((int)inputs.size()-1>=kPwMaxOperands) break;
                    if(pwTensorIsFp32(g,operand)) break; inputs.push_back(operand); oi=(int)inputs.size()-1; }
                steps.insert(steps.end(),{(int64_t)kind,(int64_t)code,(int64_t)oi,(int64_t)bcast});
                params.insert(params.end(),{p0,p1}); chain.push_back(cur);
                TensorId outT=nd.outputs[0]; if(!single(outT)) break;
                int nxt=nextOf[outT]; if(nxt<0||g.nodes[nxt].inputs.empty()) break;
                chainVal=outT; cur=nxt; }
            if(chain.size()<2) continue;
            int tail=chain.back(); TensorId tailOut=g.nodes[tail].outputs[0];
            Node fn; fn.type=OpType::FusedPointwise; fn.name=g.nodes[chain.front()].name+"#pwchain";
            fn.inputs=inputs; fn.outputs={tailOut};
            { Attr a;a.kind=Attr::Ints;a.ints=steps;fn.attr.map["pw_steps"]=a; }
            { Attr a;a.kind=Attr::Floats;a.floats=params;fn.attr.map["pw_params"]=a; }
            { Attr a;a.kind=Attr::Int;a.i=wantFlat?1:0;fn.attr.map["pw_flat"]=a; }
            g.nodes[chain.front()]=fn; for(size_t kk=1;kk<chain.size();++kk) removed.insert(chain[kk]); fused++;
        }
        if(fused){ std::vector<Node> kept; for(size_t i=0;i<g.nodes.size();++i) if(!removed.count((int)i)) kept.push_back(g.nodes[i]);
            g.nodes=std::move(kept); VKNN_INFO<<"fusePointwiseChains: fused "<<fused<<" chain(s)"; }
    }
```
Add `pwTensorIsFp32` factored from `markFp32`'s preset (name match against `mixedPrecisionFp32Tensors()`;
return false if the preset is empty/unknown at convert). If `markFp32` logic isn't easily reachable from
the pass, expose a small helper in the same TU.

- [ ] **Step 4: `inferShapes` + `gpuFlatNode` arms**
```cpp
// inferShapes:
            case OpType::FusedPointwise:
                g.desc(n.outputs[0]).shape=g.desc(n.inputs[0]).shape;
                g.desc(n.outputs[0]).dtype=g.desc(n.inputs[0]).dtype; break;
// gpuFlatNode:
            case OpType::FusedPointwise: return n.attr.geti("pw_flat",0)!=0;
```

- [ ] **Step 5: wire into `runStandardPasses`** — after `fuseDwPw` (passes.cpp:2699), before the
  const-fold loop: `if (opt.fusePointwiseChains) fusePointwiseChains(g);`

- [ ] **Step 6: run tests** — `./build-host/vknn_tests --gtest_filter='Passes.*:Ops.FusedPw*'` then full
  `./build-host/vknn_tests`. Expected: all PASS.

- [ ] **Step 7: commit**
```bash
git add src/import/passes.cpp tests/test_ops.cpp
git commit -m "import: fusePointwiseChains (standalone) + fp32-intermediate guard + layout/shape arms + tests"
```

---

## Phase 4: `pw_epilogue.glsl` + standalone Vulkan kernels — device bit-exact

**Files:** create `shaders/pw_epilogue.glsl`, `shaders/fused_pw_flat.comp`, `shaders/fused_pw_nc4.comp`;
create `src/backend/vulkan/ops/fused_pointwise.cpp`; check `vk_backend.cpp` support arm.

- [ ] **Step 1: `shaders/pw_epilogue.glsl`** — the ONE chain executor (plan SSBO at `PW_EPI_BASE`, operand
  SSBOs at `PW_EPI_BASE+1..+kPwMaxOperands` as explicit bindings + a switch — no descriptor-indexing
  feature needed). `pw_apply` (flat) / `pw_apply4` (nc4), each doing `acc=float(STORE(acc))` per step; the
  standalone kernels define `PW_EPI` unconditionally, producers via the `_epi` build variant:
```glsl
#ifndef VX_PW_EPILOGUE_GLSL
#define VX_PW_EPILOGUE_GLSL
#include "common.glsl"
#ifndef PW_EPI_MAXSTEPS
#define PW_EPI_MAXSTEPS 8
#endif
#ifndef PW_EPI_MAXOPS
#define PW_EPI_MAXOPS 6
#endif
#ifndef PW_EPI_MAXRANK
#define PW_EPI_MAXRANK 4
#endif
layout(std430, binding = PW_EPI_BASE) readonly buffer PwPlan {
  int numSteps, rank, worldFlat, pad;
  int outDim[PW_EPI_MAXRANK];
  int step[PW_EPI_MAXSTEPS*4];                 // kind,code,opSlot,bcast
  int stride[PW_EPI_MAXSTEPS*PW_EPI_MAXRANK];  // flat operand strides
  float p0[PW_EPI_MAXSTEPS]; float p1[PW_EPI_MAXSTEPS];
} plan;
layout(std430, binding = PW_EPI_BASE+1) readonly buffer PwOp1 { STORE d[]; } pwop1;
layout(std430, binding = PW_EPI_BASE+2) readonly buffer PwOp2 { STORE d[]; } pwop2;
layout(std430, binding = PW_EPI_BASE+3) readonly buffer PwOp3 { STORE d[]; } pwop3;
layout(std430, binding = PW_EPI_BASE+4) readonly buffer PwOp4 { STORE d[]; } pwop4;
layout(std430, binding = PW_EPI_BASE+5) readonly buffer PwOp5 { STORE d[]; } pwop5;
layout(std430, binding = PW_EPI_BASE+6) readonly buffer PwOp6 { STORE d[]; } pwop6;
float pwLoad(int slot,int idx){ if(slot==1)return float(pwop1.d[idx]); if(slot==2)return float(pwop2.d[idx]);
  if(slot==3)return float(pwop3.d[idx]); if(slot==4)return float(pwop4.d[idx]); if(slot==5)return float(pwop5.d[idx]);
  return float(pwop6.d[idx]); }
vec4 pwLoad4(int slot,int idx){ if(slot==1)return vec4(pwop1.d[idx*4],pwop1.d[idx*4+1],pwop1.d[idx*4+2],pwop1.d[idx*4+3]);
  if(slot==2)return vec4(pwop2.d[idx*4],pwop2.d[idx*4+1],pwop2.d[idx*4+2],pwop2.d[idx*4+3]);
  if(slot==3)return vec4(pwop3.d[idx*4],pwop3.d[idx*4+1],pwop3.d[idx*4+2],pwop3.d[idx*4+3]);
  if(slot==4)return vec4(pwop4.d[idx*4],pwop4.d[idx*4+1],pwop4.d[idx*4+2],pwop4.d[idx*4+3]);
  if(slot==5)return vec4(pwop5.d[idx*4],pwop5.d[idx*4+1],pwop5.d[idx*4+2],pwop5.d[idx*4+3]);
  return vec4(pwop6.d[idx*4],pwop6.d[idx*4+1],pwop6.d[idx*4+2],pwop6.d[idx*4+3]); }
float pw_apply(float acc, int outIdx){
  for(int s=0;s<plan.numSteps;++s){ int kind=plan.step[s*4],code=plan.step[s*4+1],slot=plan.step[s*4+2];
    if(kind==0){ int rem=outIdx,oi=0; for(int k=plan.rank-1;k>=0;--k){ int c=rem%plan.outDim[k]; rem/=plan.outDim[k]; oi+=c*plan.stride[s*PW_EPI_MAXRANK+k]; }
      acc=vx_binary(acc, pwLoad(slot,oi), code); }
    else if(kind==1) acc=vx_unary(acc,code,plan.p0[s],plan.p1[s]);
    else             acc=vx_act(acc,code,plan.p0[s],plan.p1[s]);
    acc=float(STORE(acc)); }
  return acc; }
vec4 pw_apply4(vec4 acc, int vecIdx){ int HW=plan.outDim[0];
  for(int s=0;s<plan.numSteps;++s){ int kind=plan.step[s*4],code=plan.step[s*4+1],slot=plan.step[s*4+2],bc=plan.step[s*4+3];
    if(kind==0){ int oi=(bc==1)?vecIdx/HW:vecIdx; vec4 b=pwLoad4(slot,oi);
      acc=vec4(vx_binary(acc.x,b.x,code),vx_binary(acc.y,b.y,code),vx_binary(acc.z,b.z,code),vx_binary(acc.w,b.w,code)); }
    else if(kind==1) acc=vec4(vx_unary(acc.x,code,plan.p0[s],plan.p1[s]),vx_unary(acc.y,code,plan.p0[s],plan.p1[s]),vx_unary(acc.z,code,plan.p0[s],plan.p1[s]),vx_unary(acc.w,code,plan.p0[s],plan.p1[s]));
    else             acc=vec4(vx_act(acc.x,code,plan.p0[s],plan.p1[s]),vx_act(acc.y,code,plan.p0[s],plan.p1[s]),vx_act(acc.z,code,plan.p0[s],plan.p1[s]),vx_act(acc.w,code,plan.p0[s],plan.p1[s]));
    acc=vec4(float(STORE(acc.x)),float(STORE(acc.y)),float(STORE(acc.z)),float(STORE(acc.w))); }
  return acc; }
#endif
```

- [ ] **Step 2: standalone kernels** — `shaders/fused_pw_flat.comp`:
```glsl
#version 450
#include "precision.glsl"
#define PW_EPI_BASE 2
#include "pw_epilogue.glsl"
layout(local_size_x=256) in;
layout(std430,binding=0) readonly buffer Prim { STORE prim[]; };
layout(std430,binding=1) writeonly buffer Dst { STORE dst[]; };
layout(push_constant) uniform PC { int total; } pc;
void main(){ uint gid=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*gl_NumWorkGroups.x*gl_WorkGroupSize.x;
  if(gid>=uint(pc.total)) return; dst[gid]=STORE(pw_apply(float(prim[gid]), int(gid))); }
```
`shaders/fused_pw_nc4.comp` (vec4/thread; `total` = number of vec4s):
```glsl
#version 450
#include "precision.glsl"
#define PW_EPI_BASE 2
#include "pw_epilogue.glsl"
layout(local_size_x=256) in;
layout(std430,binding=0) readonly buffer Prim { STORE prim[]; };
layout(std430,binding=1) writeonly buffer Dst { STORE dst[]; };
layout(push_constant) uniform PC { int total; } pc;
void main(){ uint gid=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*gl_NumWorkGroups.x*gl_WorkGroupSize.x;
  if(gid>=uint(pc.total)) return; vec4 v=vec4(prim[gid*4],prim[gid*4+1],prim[gid*4+2],prim[gid*4+3]);
  vec4 o=pw_apply4(v,int(gid)); dst[gid*4]=STORE(o.x); dst[gid*4+1]=STORE(o.y); dst[gid*4+2]=STORE(o.z); dst[gid*4+3]=STORE(o.w); }
```
Note `precision.glsl` must be included before `pw_epilogue.glsl` so `STORE` is defined; the plan+operand
bindings start at `PW_EPI_BASE=2` (after prim=0, dst=1).

- [ ] **Step 3: standalone Vulkan op** — `src/backend/vulkan/ops/fused_pointwise.cpp`. `prepare()` builds
  the `PwPlanCPU` (byte-identical to the GLSL block), uploads it to a device buffer, resolves/uploads
  operands via `operandBuf`, picks `shader("fused_pw_flat"/"fused_pw_nc4", env.useFp16)` with
  `numBuffers = 2 + 1 + kPwMaxOperands` (prim, dst, plan, 6 operands); `record()` binds
  `[prim, dst, plan, op1..op6]` (unused operand slots → dst as a harmless dummy) and dispatches
  `groups(total,256)`. `VKNN_REGISTER_VK_OP(OpType::FusedPointwise, FusedPointwiseOp)`. (Model the plan
  build on `flat::Binary::prepare` for strides and `fused_se.cpp` for the op skeleton; the exact
  `ComputePipeline` ctor + `dispatch(cmd, {handles}, &pc, size, groups)` are in `flat_ops.h:276-282`.)

- [ ] **Step 4: build Android** — `./build.sh --android`. Confirm `sizeof(PwPlanCPU) == ` the GLSL block
  size (std430); adjust padding if not.

- [ ] **Step 5: device bit-exact** — compile a flat probe (`Mul→Add→Clip`, rank ≤ 4) + an NC4HW4 probe
  (`Add→Mul→Relu`, `[1,C,H,W]`, `C%4==0`) with `vknn_compile`; diff fusion OFF vs ON at `--precision high`
  (fp32) AND `normal` (fp16), `--winograd off --tuning off`; `cmp` byte-identical.

- [ ] **Step 6: commit**
```bash
git add shaders/pw_epilogue.glsl shaders/fused_pw_flat.comp shaders/fused_pw_nc4.comp src/backend/vulkan/ops/fused_pointwise.cpp
git commit -m "vulkan: pw_epilogue.glsl (plan SSBO) + standalone FusedPointwise kernels, bit-exact"
```

> **Milestone:** standalone chain fusion works GPU end-to-end, bit-exact — captures the bulk of
> model.json's 274 pointwise ops, and frees their intermediate buffers (memory win).

---

## Phase 5: Central CPU hook + producer-attach in the pass

**Files:** `src/backend/cpu/cpu_backend.cpp`, `src/import/passes.cpp`, `tests/test_ops.cpp`.

- [ ] **Step 1: failing test** — `Conv→Mul→Add→Clip`: pass off vs on → bit-exact fp32; the Conv node now
  carries `pw_steps` and the 3 pointwise nodes are gone.

- [ ] **Step 2: CPU hook** — in the CPU executor loop, after `op->run(node,ctx)`:
```cpp
if (node.type != OpType::FusedPointwise && node.attr.has("pw_steps")) applyPwEpilogue(node, ctx);
```

- [ ] **Step 3: producer-attach** — extend the pass: before emitting a standalone node, if the head's
  `inputs[0]` producer is single-consumer, `pwEpilogueCapable(type)`, same world/precision, has no
  `pw_steps`, and neither its output nor the operands are fp32-promoted → set `pw_steps`/`pw_params`/
  `pw_opbase` on the producer, append operands to `producer.inputs`, reuse the tail output, remove the
  chain nodes. Else standalone. Start `pwEpilogueCapable` = {Binary, Add, Unary, Clip, Relu} (grows per
  phase).

- [ ] **Step 4: run full host tests** — `./build-host/vknn_tests`. Expected: PASS.

- [ ] **Step 5: commit**
```bash
git add src/backend/cpu/cpu_backend.cpp src/import/passes.cpp tests/test_ops.cpp
git commit -m "fusion: central CPU epilogue hook + producer-attach (elementwise producers)"
```

---

## Phase 6: GPU epilogue infra + elementwise family

**Files:** `CMakeLists.txt` (`_epi` variants for the elementwise shaders); `vk_op_common.h`
(`prepareEpilogue`/`appendEpilogueBuffers`); elementwise shaders + ops (`flat_binary`/`add`/`relu`/
`unary`/`clip`/`binary`).

- [ ] **Step 1: build variants** — in the `CMakeLists.txt` shader loop, for a **curated list** of
  epilogue-capable shaders (this phase: the elementwise set only — memory guard), also emit
  `<name>_epi.spv` (`-DPW_EPI`) and `<name>_epi_fp16.spv` (`-DPW_EPI -DUSE_FP16`).

- [ ] **Step 2: `prepareEpilogue`/`appendEpilogueBuffers`** — in `vk_op_common.h`:
```cpp
struct EpiBind { std::string suffix; uint32_t extra=0; };
// Build+cache the plan buffer + resolve operands (operandBuf, keyed for no dup). "" / "_epi".
EpiBind prepareEpilogue(const Node& node, VkOpEnv& env, bool flatWorld, const Shape& outShape,
                        /*out*/ std::shared_ptr<vk::Buffer>& planBuf, std::vector<std::shared_ptr<vk::Buffer>>& holds);
void appendEpilogueBuffers(std::vector<VkBuffer>& bufs, const Node& node, VkOpEnv& env,
                           const std::shared_ptr<vk::Buffer>& planBuf, std::vector<std::shared_ptr<vk::Buffer>>& holds,
                           vk::Buffer* dummy);
```
(Build the same `PwPlanCPU` as the standalone op; append `[plan, op1..op6]` padding unused with `dummy`.)

- [ ] **Step 3: wire the elementwise kernels** — each: `#define PW_EPI_BASE <own buf count>` +
  `#include "pw_epilogue.glsl"` (after `precision.glsl`), and wrap the store:
  `d[i] = STORE(pw_apply(float(STORE(vx_act(v,pc.act,pc.actLo,pc.actHi))), int(gid)));` (flat) or the
  `pw_apply4` vec4 form (nc4). In each op's `prepare()`: `auto e=prepareEpilogue(...); shader(base+e.suffix,
  fp16); numBuffers=own+e.extra;`; in `record()`: `appendEpilogueBuffers(...)`.

- [ ] **Step 4: device bit-exact** — probes `Sigmoid→Mul(channel)`, `Add→Relu→Clip`, and the effnet SE
  `Mul`; diff OFF vs ON fp32 + fp16 → byte-identical; `--profile` shows pointwise rows folded into the
  producer row.

- [ ] **Step 5: commit**
```bash
git add CMakeLists.txt src/backend/vulkan/ops/vk_op_common.h shaders/flat_binary.comp shaders/add.comp shaders/relu.comp shaders/unary.comp shaders/clip.comp shaders/binary.comp src/backend/vulkan/ops/binary.cpp src/backend/vulkan/ops/add.cpp src/backend/vulkan/ops/relu.cpp src/backend/vulkan/ops/unary.cpp src/backend/vulkan/ops/clip.cpp
git commit -m "vulkan: prepareEpilogue + elementwise-family epilogue via pw_epilogue.glsl (bit-exact)"
```

---

## Phases 7–11: Roll the epilogue across producer families (profile order)

Same mechanical change as Phase 6, per family: add to `pwEpilogueCapable`; add the 3 epilogue lines +
`_epi` build variants; the `prepareEpilogue`/`appendEpilogueBuffers` two-liner in the op; device bit-exact
gate (OFF vs ON, fp32 + fp16, `cmp` byte-identical) + `--profile`; commit. `outIdx` passed to `pw_apply`
must be the logical output element index (linear for flat, vec4/channel-block for nc4).

- [ ] **Phase 7 — MatMul/Gemm/FC** (yonosplat 73% lever). Shaders `matmul*`, `matmul_tiled*`, `gemm`,
  `fc*`; ops `matmul.cpp`/`gemm.cpp`/`fc.cpp`. **Packed-fp16 care (review pt 3):** these hand-pack
  `f16vec4` — widen to fp32 at the store, run `pw_apply`, RTE-store. Probe `MatMul→Add(bias)→Gelu→Add`.
- [ ] **Phase 8 — Softmax/LayerNorm/Reduce** (post-reduction store). `softmax*`/`flat_softmax`/
  `flat_layernorm`/`flat_reduce`; ops `softmax.cpp`/`layernorm.cpp`/`reduce.cpp`. Probe `LayerNorm→Mul→Add`.
- [ ] **Phase 9 — GridSample/Resize/ConvTranspose** (model.json warps). `gridsample*`/`resize*`/
  `convtranspose`; ops likewise. Probe `GridSample→Add→Mul`.
- [ ] **Phase 10 — Conv family** (CNN; **packed-fp16 care**). `conv*`/`dwconv*`/`conv1x1*`/`conv3x3_lds*`/
  `conv_reg*`/`wino_*` output; ops `conv.cpp`/`fused_dwpw.cpp`. Plan lives in the SSBO → conv PC
  unaffected. Probe `Conv→PRelu`, `Conv→Div`.
- [ ] **Phase 11 — Pooling** (minor). `avgpool*`/`maxpool*`; probe `AvgPool→Mul→Add`.

---

## Phase 12: Ship — validation, flag, recompile, perf + memory report

**Files:** `convert/compile.cpp`; `docs/superpowers/plans/results-fuse-pointwise.md`.

- [ ] **Step 1: A/B flag** — `--no-fuse-pointwise` → `opt.fusePointwiseChains=false` (mirror
  `--no-fuse-swish`).
- [ ] **Step 2: recompile** the CNN suite, yonosplat, model.json to fused + `_nofuse` `.vxm`s.
- [ ] **Step 3: full-suite bit-exact gate** — per model/output `cmp fused nofuse` byte-identical at fp32
  and fp16.
- [ ] **Step 4: CNN no-regression** — `vknn_benchmark` goldens: `cosine==1.0`/PASS, `submit+gpu ≤`
  baseline.
- [ ] **Step 5: perf + MEMORY** — `run_io --timing --profile` OFF vs ON on yonosplat + model.json: record
  `submit+gpu` delta, dispatch-count drop, **and peak device memory (must be ≤ baseline)**.
- [ ] **Step 6: full host suite** — `./build-host/vknn_tests` green.
- [ ] **Step 7: commit + finish** — then `superpowers:finishing-a-development-branch`.

---

## Self-review

- **Spec coverage:** all §3 pieces mapped (encoding→P1/2, pass→P3/5, pw_epilogue.glsl→P4, per-kernel→P6/7–11,
  prepareEpilogue→P6, CPU hook→P5, OpType/infer/gpuFlat→P1/3, bit-exact→P4/6). §7 rollout→P6–11. §8 risks
  →Critical review + Memory budget.
- **Corrections applied:** fp32 guard (P3), unified plan-SSBO (P4), packed-fp16 (P7/10), commutative
  operands (P3 `pwEncodeStep`), O(n) pass (P3 `nextOf`), memory budget (dedicated section), self-contained
  (all code inline).
- **Type consistency:** `pw_steps`/`pw_params`/`pw_opbase`/`pw_flat`, `[kind,code,operandInputIdx,bcastMode]`,
  `PwPlanCPU`↔`PwPlan`, `kPwMaxSteps/Operands/Rank`, `applyPwEpilogue`, `prepareEpilogue`/
  `appendEpilogueBuffers`, `pwEpilogueCapable`, `pwTensorIsFp32` — identical across CPU, pass, shaders, ops.
```
