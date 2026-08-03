#ifndef VX_PW_EPILOGUE_GLSL
#define VX_PW_EPILOGUE_GLSL
#include "common.glsl"
// Storage element type of the operand buffers. Templated kernels get it from precision.glsl;
// hand-written fp16 kernels #define STORE float16_t before including this; pure-fp32 kernels
// (vec4/float buffers, no precision.glsl) fall back to float here.
#ifndef STORE
#define STORE float
#endif
// Storage-rounding conversion. Templated kernels get it from precision.glsl/store16.glsl;
// pure-fp32 kernels fall back to the identity here.
#ifndef TO_STORE
#define TO_STORE(x) STORE(x)
#endif
// Per-step rounding. Pinned to the integer-exact vknnRte16 whenever fp16 storage is in play
// (VX_STORE16_GLSL): a unit's step must round identically to the standalone elementwise kernel it
// replaces, regardless of the HOST kernel's rounding mode — drivers whose RoundingModeRTE breaks
// ties away from even would otherwise split fused and unfused bytes on exact-tie values.
#ifdef VX_STORE16_GLSL
#define PW_ROUND(x) float(vknnRte16(x))
#else
#define PW_ROUND(x) float(TO_STORE(x))
#endif
#ifndef PW_EPI_MAXSTEPS
#define PW_EPI_MAXSTEPS 16 // == kPwMaxSteps (include/vknn/op_type.h)
#endif
#ifndef PW_EPI_MAXRANK
#define PW_EPI_MAXRANK 4 // == kPwMaxRank (include/vknn/op_type.h)
#endif
// Value references inside a step (mirrors kPwRef* in include/vknn/op_type.h): -1 accumulator,
// -2 entry value, -3 none, -4-r register r, -8-i operand slot i.
// Step kinds and the step-record width, mirroring kPwKind* / the 8-int step record in
// include/vknn/op_type.h (kind, code, srcA, srcB, srcC, dst, bcast, bcastSrc).
#define PW_STEP_FIELDS 8
#define PW_KIND_BINARY 0
#define PW_KIND_UNARY  1
#define PW_KIND_ACT    2
#define PW_KIND_SELECT 3
#define PW_KIND_LOAD   4
#define PW_REF_ACC   (-1)
#define PW_REF_ENTRY (-2)
#define PW_REF_NONE  (-3)
#define PW_REF_REG0  (-4)
#define PW_REF_OP0   (-8)
// Operand broadcast classes, mirroring the kPwBcast* constants in include/vknn/op_type.h.
#define PW_BCAST_SAME      0
#define PW_BCAST_CHANNEL   1
#define PW_BCAST_GENERAL   2
#define PW_BCAST_SCALAR    3
#define PW_BCAST_SPATIAL   4
#define PW_BCAST_ROW       5
#define PW_BCAST_COL       6
#define PW_BCAST_ROW_SPLAT 7
#define PW_BCAST_COL_SPLAT 8
#define PW_BCAST_PACKED    9
// One bit per broadcast class, governing the BLOCKED-world resolvers (pwLoadBc4 / pwNc4Idx) alone.
// They carry an arm per class, run per element, and inline into every kernel that carries an
// epilogue, so an arm no plan can reach is pure cost in all of them; a kernel therefore compiles
// only the classes its plans may carry. The flat resolver (pwFlatIdx) needs no such mask: every
// class it sees resolves through its one strided walk.
//
// The split is what the resolver has to compute, not which shapes are expressible. A DIRECT class
// addresses its operand from the store index alone; a GEOMETRIC class first recovers the store's
// (n, channel-block, h, w) from that index, which is the division chain that costs. PW_BCAST_MASK
// defaults to both groups, the set a kernel serving arbitrary plans needs.
#define PW_BCAST_BIT(cls) (1 << (cls))
#define PW_BCAST_MASK_DIRECT (PW_BCAST_BIT(PW_BCAST_SAME) | PW_BCAST_BIT(PW_BCAST_CHANNEL) \
                            | PW_BCAST_BIT(PW_BCAST_GENERAL) | PW_BCAST_BIT(PW_BCAST_SCALAR))
#define PW_BCAST_MASK_GEOMETRIC (PW_BCAST_BIT(PW_BCAST_SPATIAL) | PW_BCAST_BIT(PW_BCAST_ROW) \
                               | PW_BCAST_BIT(PW_BCAST_COL) | PW_BCAST_BIT(PW_BCAST_ROW_SPLAT) \
                               | PW_BCAST_BIT(PW_BCAST_COL_SPLAT) | PW_BCAST_BIT(PW_BCAST_PACKED))
#define PW_BCAST_MASK_ALL (PW_BCAST_MASK_DIRECT | PW_BCAST_MASK_GEOMETRIC)
#ifndef PW_BCAST_MASK
#define PW_BCAST_MASK PW_BCAST_MASK_ALL
#endif
#define PW_BCAST_HAS(cls) ((PW_BCAST_MASK & PW_BCAST_BIT(cls)) != 0)
// The four row/column classes share one guarded block: they alone recover both W and H.
#define PW_BCAST_MASK_ROWCOL (PW_BCAST_BIT(PW_BCAST_ROW) | PW_BCAST_BIT(PW_BCAST_COL) \
                            | PW_BCAST_BIT(PW_BCAST_ROW_SPLAT) | PW_BCAST_BIT(PW_BCAST_COL_SPLAT))
#define PW_BCAST_HAS_ROWCOL ((PW_BCAST_MASK & PW_BCAST_MASK_ROWCOL) != 0)
// Stride-slot order of a PW_BCAST_PACKED step's packed vec4-space strides
// (plan.stride[s*PW_EPI_MAXRANK + slot]), mirroring kPwPackedStride* in include/vknn/op_type.h.
// A broadcast axis carries stride 0; a zero CB stride marks a channel-1 operand (splat lane 0).
#define PW_PACKED_STRIDE_N  0
#define PW_PACKED_STRIDE_CB 1
#define PW_PACKED_STRIDE_H  2
#define PW_PACKED_STRIDE_W  3
// NC4HW4 channel-block width (kNC4Block in include/vknn/nchw.h): four channels per packed vec4.
#define NC4_LANES 4
// flags bit 0 records the fp32-chained discipline for the plan (set from the pw_relax attr). The
// discipline is selected at COMPILE time: kernels built with -DPW_RELAX contain only the
// fp32-chained appliers, kernels built without it only the strict per-step-rounded ones — a
// runtime branch would carry both bodies in every kernel and cost occupancy.
#define PW_FLAG_CHAIN32 1
// Operand slots every kernel carrying this epilogue declares: kPwMaxOperands, the same budget a
// standalone unit gets. Declaring fewer would not merely cost a fusion -- it changes ANSWERS, since
// a unit split for want of a slot rounds its intermediate through fp16 storage where the whole unit
// kept it in an fp32 register.
#define PW_OPERAND_SLOTS 9
// First binding after the operand block; the extra output streams follow it.
#define PW_EPI_OUT_BASE (PW_EPI_BASE + 1 + PW_OPERAND_SLOTS)
layout(std430, binding = PW_EPI_BASE) readonly buffer PwPlan {
  int numSteps, rank, worldFlat, numOuts, flags;
  int outDim[PW_EPI_MAXRANK];
  int step[PW_EPI_MAXSTEPS*PW_STEP_FIELDS];                // kind(0 bin,1 unary,2 act,3 select,4 load), code, srcA, srcB, srcC, dst, bcast, bcastSrc
  int stride[PW_EPI_MAXSTEPS*PW_EPI_MAXRANK]; // flat broadcast strides of the bcastSrc-marked operand
  float p0[PW_EPI_MAXSTEPS]; float p1[PW_EPI_MAXSTEPS];
  int outStep[4];                             // step whose value stores to extra output o (PW_REF_ENTRY = entry, PW_REF_NONE = unused)
} plan;
layout(std430, binding = PW_EPI_BASE+1) readonly buffer PwOp1 { STORE d[]; } pwop1;
layout(std430, binding = PW_EPI_BASE+2) readonly buffer PwOp2 { STORE d[]; } pwop2;
layout(std430, binding = PW_EPI_BASE+3) readonly buffer PwOp3 { STORE d[]; } pwop3;
layout(std430, binding = PW_EPI_BASE+4) readonly buffer PwOp4 { STORE d[]; } pwop4;
layout(std430, binding = PW_EPI_BASE+5) readonly buffer PwOp5 { STORE d[]; } pwop5;
layout(std430, binding = PW_EPI_BASE+6) readonly buffer PwOp6 { STORE d[]; } pwop6;
#if PW_OPERAND_SLOTS > 6
layout(std430, binding = PW_EPI_BASE+7) readonly buffer PwOp7 { STORE d[]; } pwop7;
layout(std430, binding = PW_EPI_BASE+8) readonly buffer PwOp8 { STORE d[]; } pwop8;
layout(std430, binding = PW_EPI_BASE+9) readonly buffer PwOp9 { STORE d[]; } pwop9;
#endif
layout(std430, binding = PW_EPI_OUT_BASE)   writeonly buffer PwOut1 { STORE d[]; } pwout1;
layout(std430, binding = PW_EPI_OUT_BASE+1) writeonly buffer PwOut2 { STORE d[]; } pwout2;
layout(std430, binding = PW_EPI_OUT_BASE+2) writeonly buffer PwOut3 { STORE d[]; } pwout3;
layout(std430, binding = PW_EPI_OUT_BASE+3) writeonly buffer PwOut4 { STORE d[]; } pwout4;
float pwLoad(int slot,int idx){
  if(slot==0)return float(pwop1.d[idx]);
  if(slot==1)return float(pwop2.d[idx]);
  if(slot==2)return float(pwop3.d[idx]);
  if(slot==3)return float(pwop4.d[idx]);
  if(slot==4)return float(pwop5.d[idx]);
  if(slot==5)return float(pwop6.d[idx]);
#if PW_OPERAND_SLOTS > 6
  if(slot==6)return float(pwop7.d[idx]);
  if(slot==7)return float(pwop8.d[idx]);
  return float(pwop9.d[idx]);
#else
  return float(pwop6.d[idx]);
#endif
}
vec4 pwLoad4(int slot,int idx){
  if(slot==0)return vec4(pwop1.d[idx*4],pwop1.d[idx*4+1],pwop1.d[idx*4+2],pwop1.d[idx*4+3]);
  if(slot==1)return vec4(pwop2.d[idx*4],pwop2.d[idx*4+1],pwop2.d[idx*4+2],pwop2.d[idx*4+3]);
  if(slot==2)return vec4(pwop3.d[idx*4],pwop3.d[idx*4+1],pwop3.d[idx*4+2],pwop3.d[idx*4+3]);
  if(slot==3)return vec4(pwop4.d[idx*4],pwop4.d[idx*4+1],pwop4.d[idx*4+2],pwop4.d[idx*4+3]);
  if(slot==4)return vec4(pwop5.d[idx*4],pwop5.d[idx*4+1],pwop5.d[idx*4+2],pwop5.d[idx*4+3]);
  if(slot==5)return vec4(pwop6.d[idx*4],pwop6.d[idx*4+1],pwop6.d[idx*4+2],pwop6.d[idx*4+3]);
#if PW_OPERAND_SLOTS > 6
  if(slot==6)return vec4(pwop7.d[idx*4],pwop7.d[idx*4+1],pwop7.d[idx*4+2],pwop7.d[idx*4+3]);
  if(slot==7)return vec4(pwop8.d[idx*4],pwop8.d[idx*4+1],pwop8.d[idx*4+2],pwop8.d[idx*4+3]);
  return vec4(pwop9.d[idx*4],pwop9.d[idx*4+1],pwop9.d[idx*4+2],pwop9.d[idx*4+3]);
#else
  return vec4(pwop6.d[idx*4],pwop6.d[idx*4+1],pwop6.d[idx*4+2],pwop6.d[idx*4+3]);
#endif
}
void pwStoreOut(int o,int idx,float v){ if(o==0)pwout1.d[idx]=STORE(v); else if(o==1)pwout2.d[idx]=STORE(v);
  else if(o==2)pwout3.d[idx]=STORE(v); else pwout4.d[idx]=STORE(v); }
void pwStoreOut4(int o,int idx,vec4 v){
  if(o==0){pwout1.d[idx*4]=STORE(v.x);pwout1.d[idx*4+1]=STORE(v.y);pwout1.d[idx*4+2]=STORE(v.z);pwout1.d[idx*4+3]=STORE(v.w);}
  else if(o==1){pwout2.d[idx*4]=STORE(v.x);pwout2.d[idx*4+1]=STORE(v.y);pwout2.d[idx*4+2]=STORE(v.z);pwout2.d[idx*4+3]=STORE(v.w);}
  else if(o==2){pwout3.d[idx*4]=STORE(v.x);pwout3.d[idx*4+1]=STORE(v.y);pwout3.d[idx*4+2]=STORE(v.z);pwout3.d[idx*4+3]=STORE(v.w);}
  else {pwout4.d[idx*4]=STORE(v.x);pwout4.d[idx*4+1]=STORE(v.y);pwout4.d[idx*4+2]=STORE(v.z);pwout4.d[idx*4+3]=STORE(v.w);} }
// Flat operand element index for step s: a same-shape operand (bc==0) reads at outIdx; only a real
// broadcast needs the per-axis strided decomposition (integer div/mod per element), so that loop is
// skipped for the common full-size operand.
int pwFlatIdx(int s,int bc,int outIdx){
  if(bc==PW_BCAST_SAME) return outIdx;
  // A per-pixel operand is the trailing H*W plane of a single-batch run, so it indexes by the
  // spatial remainder — the same value the strided walk below yields, without its div/mod chain.
  if(bc==PW_BCAST_SPATIAL) return outIdx % (plan.outDim[plan.rank-2]*plan.outDim[plan.rank-1]);
  int rem=outIdx; int oi=0; for(int k=plan.rank-1;k>=0;--k){ int c=rem%plan.outDim[k]; rem/=plan.outDim[k]; oi+=c*plan.stride[s*PW_EPI_MAXRANK+k]; } return oi; }
vec4 pwBin4(vec4 a,vec4 b,int code){ return vec4(vx_binary(a.x,b.x,code),vx_binary(a.y,b.y,code),vx_binary(a.z,b.z,code),vx_binary(a.w,b.w,code)); }
vec4 pwRound4(vec4 v){ return vec4(PW_ROUND(v.x),PW_ROUND(v.y),PW_ROUND(v.z),PW_ROUND(v.w)); }
vec4 pwToStore4(vec4 v){ return vec4(float(TO_STORE(v.x)),float(TO_STORE(v.y)),float(TO_STORE(v.z)),float(TO_STORE(v.w))); }
// The appliers share one execution model: acc starts at the entry value (the host kernel passes its
// RAW fp32 accumulator); dst >= 0 additionally keeps a copy in one of 4 registers for later steps;
// after each step the exported values (plan.outStep) store to the extra output streams. The two
// rounding disciplines:
//   strict (flags bit 0 clear): the entry is first rounded through TO_STORE (the byte the standalone
//     producer would store) and every compute step's result rounds through TO_STORE, so each fused
//     step reproduces, bit for bit, the fp16 store the unfused graph would make (a load passes the
//     already-rounded storage value through unrounded).
//   fp32-chained (flags bit 0 set): the entry still rounds to the producer's store byte (keeping
//     every inter-unit tensor on the unfused graph's trajectory), but the steps chain unrounded in
//     fp32 registers and the unit rounds ONCE per stored stream through the integer-exact PW_ROUND.
//     Fewer roundings than the unfused graph on every multi-step chain, and byte-identical to it
//     for single-step units and chains ending in a monotone activation (rounding commutes there).
#ifndef PW_RELAX
float pw_apply_st(float entryRaw, int outIdx){
  float entry=float(TO_STORE(entryRaw));
  float acc=entry; float r0=0.,r1=0.,r2=0.,r3=0.;
  for(int o=0;o<plan.numOuts;++o){ if(plan.outStep[o]==PW_REF_ENTRY) pwStoreOut(o,outIdx,entry); }
  for(int s=0;s<plan.numSteps;++s){
    int kind=plan.step[s*PW_STEP_FIELDS],code=plan.step[s*PW_STEP_FIELDS+1],a=plan.step[s*PW_STEP_FIELDS+2],b=plan.step[s*PW_STEP_FIELDS+3],c=plan.step[s*PW_STEP_FIELDS+4];
    int dst=plan.step[s*PW_STEP_FIELDS+5],bc=plan.step[s*PW_STEP_FIELDS+6],bsrc=plan.step[s*PW_STEP_FIELDS+7];
    float va,vb=0.,vc=0.;
    // resolve each source: operand refs load from their slot (the bcastSrc-marked one strided)
    if(a<=PW_REF_OP0)      va=pwLoad(PW_REF_OP0-a,pwFlatIdx(s,bsrc==1?bc:0,outIdx));
    else if(a==PW_REF_ACC) va=acc; else if(a==PW_REF_ENTRY) va=entry;
    else if(a==PW_REF_REG0) va=r0; else if(a==PW_REF_REG0-1) va=r1; else if(a==PW_REF_REG0-2) va=r2; else va=r3;
    if(b<=PW_REF_OP0)      vb=pwLoad(PW_REF_OP0-b,pwFlatIdx(s,bsrc==2?bc:0,outIdx));
    else if(b==PW_REF_ACC) vb=acc; else if(b==PW_REF_ENTRY) vb=entry;
    else if(b==PW_REF_REG0) vb=r0; else if(b==PW_REF_REG0-1) vb=r1; else if(b==PW_REF_REG0-2) vb=r2; else if(b==PW_REF_REG0-3) vb=r3;
    if(c<=PW_REF_OP0)      vc=pwLoad(PW_REF_OP0-c,pwFlatIdx(s,bsrc==3?bc:0,outIdx));
    else if(c==PW_REF_ACC) vc=acc; else if(c==PW_REF_ENTRY) vc=entry;
    else if(c==PW_REF_REG0) vc=r0; else if(c==PW_REF_REG0-1) vc=r1; else if(c==PW_REF_REG0-2) vc=r2; else if(c==PW_REF_REG0-3) vc=r3;
    if(kind==PW_KIND_BINARY)      acc=PW_ROUND(vx_binary(va,vb,code));
    else if(kind==PW_KIND_UNARY)  acc=PW_ROUND(vx_unary(va,code,plan.p0[s],plan.p1[s]));
    else if(kind==PW_KIND_ACT)    acc=PW_ROUND(vx_act(va,code,plan.p0[s],plan.p1[s]));
    else if(kind==PW_KIND_SELECT) acc=PW_ROUND(va!=0.?vb:vc);
    else                          acc=va;
    if(dst==0)r0=acc; else if(dst==1)r1=acc; else if(dst==2)r2=acc; else if(dst==3)r3=acc;
    for(int o=0;o<plan.numOuts;++o){ if(plan.outStep[o]==s) pwStoreOut(o,outIdx,acc); }
  }
  return acc; }
float pw_apply(float entry, int outIdx){ return pw_apply_st(entry,outIdx); }
#else
float pw_apply_rx(float entryRaw, int outIdx){
  float entry=float(TO_STORE(entryRaw));
  float acc=entry; float r0=0.,r1=0.,r2=0.,r3=0.;
  for(int o=0;o<plan.numOuts;++o){ if(plan.outStep[o]==PW_REF_ENTRY) pwStoreOut(o,outIdx,entry); }
  for(int s=0;s<plan.numSteps;++s){
    int kind=plan.step[s*PW_STEP_FIELDS],code=plan.step[s*PW_STEP_FIELDS+1],a=plan.step[s*PW_STEP_FIELDS+2],b=plan.step[s*PW_STEP_FIELDS+3],c=plan.step[s*PW_STEP_FIELDS+4];
    int dst=plan.step[s*PW_STEP_FIELDS+5],bc=plan.step[s*PW_STEP_FIELDS+6],bsrc=plan.step[s*PW_STEP_FIELDS+7];
    float va,vb=0.,vc=0.;
    if(a<=PW_REF_OP0)      va=pwLoad(PW_REF_OP0-a,pwFlatIdx(s,bsrc==1?bc:0,outIdx));
    else if(a==PW_REF_ACC) va=acc; else if(a==PW_REF_ENTRY) va=entry;
    else if(a==PW_REF_REG0) va=r0; else if(a==PW_REF_REG0-1) va=r1; else if(a==PW_REF_REG0-2) va=r2; else va=r3;
    if(b<=PW_REF_OP0)      vb=pwLoad(PW_REF_OP0-b,pwFlatIdx(s,bsrc==2?bc:0,outIdx));
    else if(b==PW_REF_ACC) vb=acc; else if(b==PW_REF_ENTRY) vb=entry;
    else if(b==PW_REF_REG0) vb=r0; else if(b==PW_REF_REG0-1) vb=r1; else if(b==PW_REF_REG0-2) vb=r2; else if(b==PW_REF_REG0-3) vb=r3;
    if(c<=PW_REF_OP0)      vc=pwLoad(PW_REF_OP0-c,pwFlatIdx(s,bsrc==3?bc:0,outIdx));
    else if(c==PW_REF_ACC) vc=acc; else if(c==PW_REF_ENTRY) vc=entry;
    else if(c==PW_REF_REG0) vc=r0; else if(c==PW_REF_REG0-1) vc=r1; else if(c==PW_REF_REG0-2) vc=r2; else if(c==PW_REF_REG0-3) vc=r3;
    if(kind==PW_KIND_BINARY)      acc=vx_binary(va,vb,code);
    else if(kind==PW_KIND_UNARY)  acc=vx_unary(va,code,plan.p0[s],plan.p1[s]);
    else if(kind==PW_KIND_ACT)    acc=vx_act(va,code,plan.p0[s],plan.p1[s]);
    else if(kind==PW_KIND_SELECT) acc=va!=0.?vb:vc;
    else                          acc=va;
    if(dst==0)r0=acc; else if(dst==1)r1=acc; else if(dst==2)r2=acc; else if(dst==3)r3=acc;
    for(int o=0;o<plan.numOuts;++o){ if(plan.outStep[o]==s) pwStoreOut(o,outIdx,PW_ROUND(acc)); }
  }
  return PW_ROUND(acc); }
float pw_apply(float entry, int outIdx){ return pw_apply_rx(entry,outIdx); }
#endif
// Scalar NC4HW4 store: packedIdx = ((n*Cb+cb)*HW + hw)*4 + lane. Operands are NC4HW4-packed
// (runtime activations natively; constants packed at upload), so bc==0 (same shape) loads at
// packedIdx, bc==1 (per-channel [N,C,1,1]) at the store's channel-block lane, bc==3 at element 0.
// bc==4 (per-pixel [1,1,H,W]) packs one value per spatial position at channel lane 0, so vecIdx%HW
// recovers the pixel and the *4 lands on that lane. Single batch only (pwBcastClass enforces it):
// vecIdx%HW carries no batch index.
// bc==5 (per-row [N,C,H,1]) and bc==6 (per-column [N,C,1,W]) are packed with their own spatial
// extent (H or W), so the operand's block index is the store's channel-block index (vecIdx/HW,
// which carries n) times that extent plus the row hw/W or column hw%W — four distinct lane values,
// any batch. bc==7 ([1,1,H,1]) and bc==8 ([1,1,1,W]) hold one value per row/column at channel
// lane 0 like bc==4, and share its single-batch restriction. W rides plan.outDim[1], H
// plan.outDim[2] and the run's channel-block count plan.outDim[3]; the NC4 plan uses no other slot.
// bc==9 (generic 1-or-full mask, PW_BCAST_PACKED) covers every remaining broadcast pattern: the
// store's (n, cb, h, w) recover from vecIdx via outDim, and the operand's own NC4HW4 block index
// is their dot product with the step's packed vec4-space strides (plan.stride, zero on each
// broadcast axis). A zero cb stride means the operand's channel axis is 1 — its value lives at
// channel lane 0 of its block and splats across the four lanes; a non-broadcast channel axis
// aligns lane for lane. Any batch: n is recovered explicitly, unlike the vecIdx%HW classes.
// One vec4 operand fetch for the NC4HW4 kernel, by broadcast class: scalar, per-pixel and the
// row/column *Splat classes hold ONE value for the four channel lanes and splat it, while
// same-shape, per-channel and the row/column channel-carrying classes hold four distinct lane
// values and load them as a vec4. `s` is the step index: only the packed class reads its
// per-step strides.
int pwPackedBlock(int s,int vecIdx,int HW){
  int W=plan.outDim[1]; int runCb=plan.outDim[3];
  int hw=vecIdx%HW; int blockIdx=vecIdx/HW;
  int cb=blockIdx%runCb; int n=blockIdx/runCb;
  int h=hw/W; int w=hw-h*W;
  return n*plan.stride[s*PW_EPI_MAXRANK+PW_PACKED_STRIDE_N]+cb*plan.stride[s*PW_EPI_MAXRANK+PW_PACKED_STRIDE_CB]
        +h*plan.stride[s*PW_EPI_MAXRANK+PW_PACKED_STRIDE_H]+w*plan.stride[s*PW_EPI_MAXRANK+PW_PACKED_STRIDE_W]; }
vec4 pwLoadBc4(int slot,int s,int m,int vecIdx,int HW){
#if PW_BCAST_HAS(PW_BCAST_SCALAR)
  if(m==PW_BCAST_SCALAR)  return vec4(pwLoad(slot,0));
#endif
#if PW_BCAST_HAS(PW_BCAST_SPATIAL)
  if(m==PW_BCAST_SPATIAL) return vec4(pwLoad(slot,(vecIdx%HW)*4));
#endif
#if PW_BCAST_HAS(PW_BCAST_PACKED)
  if(m==PW_BCAST_PACKED){
    int ob=pwPackedBlock(s,vecIdx,HW);
    if(plan.stride[s*PW_EPI_MAXRANK+PW_PACKED_STRIDE_CB]==0) return vec4(pwLoad(slot,ob*NC4_LANES)); // channel axis 1: splat lane 0
    return pwLoad4(slot,ob);
  }
#endif
#if PW_BCAST_HAS_ROWCOL
  if(m>=PW_BCAST_ROW){
    int W=plan.outDim[1]; int H=plan.outDim[2];
    int blockIdx=vecIdx/HW; int hw=vecIdx%HW;
#if PW_BCAST_HAS(PW_BCAST_ROW)
    if(m==PW_BCAST_ROW) return pwLoad4(slot,blockIdx*H+hw/W);
#endif
#if PW_BCAST_HAS(PW_BCAST_COL)
    if(m==PW_BCAST_COL) return pwLoad4(slot,blockIdx*W+hw%W);
#endif
#if PW_BCAST_HAS(PW_BCAST_ROW_SPLAT)
    if(m==PW_BCAST_ROW_SPLAT) return vec4(pwLoad(slot,(hw/W)*4));
#endif
    return vec4(pwLoad(slot,(hw%W)*4)); // PW_BCAST_COL_SPLAT
  }
#endif
  return pwLoad4(slot,(m==PW_BCAST_CHANNEL)?vecIdx/HW:vecIdx); }
int pwNc4Idx(int s,int bc,int packedIdx,int lane,int vecIdx){
  int HW=plan.outDim[0];
#if PW_BCAST_HAS(PW_BCAST_SCALAR)
  if(bc==PW_BCAST_SCALAR)  return 0;
#endif
#if PW_BCAST_HAS(PW_BCAST_CHANNEL)
  if(bc==PW_BCAST_CHANNEL) return (vecIdx/HW)*4+lane;
#endif
#if PW_BCAST_HAS(PW_BCAST_SPATIAL)
  if(bc==PW_BCAST_SPATIAL) return (vecIdx%HW)*4;
#endif
#if PW_BCAST_HAS(PW_BCAST_PACKED)
  if(bc==PW_BCAST_PACKED){
    int ob=pwPackedBlock(s,vecIdx,HW);
    return plan.stride[s*PW_EPI_MAXRANK+PW_PACKED_STRIDE_CB]==0 ? ob*NC4_LANES : ob*NC4_LANES+lane; // channel axis 1: lane 0 value
  }
#endif
#if PW_BCAST_HAS_ROWCOL
  if(bc>=PW_BCAST_ROW){
    int W=plan.outDim[1]; int H=plan.outDim[2];
    int blockIdx=vecIdx/HW; int hw=vecIdx%HW;
#if PW_BCAST_HAS(PW_BCAST_ROW)
    if(bc==PW_BCAST_ROW) return (blockIdx*H+hw/W)*4+lane;
#endif
#if PW_BCAST_HAS(PW_BCAST_COL)
    if(bc==PW_BCAST_COL) return (blockIdx*W+hw%W)*4+lane;
#endif
#if PW_BCAST_HAS(PW_BCAST_ROW_SPLAT)
    if(bc==PW_BCAST_ROW_SPLAT) return (hw/W)*4;
#endif
    return (hw%W)*4; // PW_BCAST_COL_SPLAT
  }
#endif
  return packedIdx; }
#ifndef PW_RELAX
float pw_apply_nc4_st(float entryRaw, int packedIdx){ int lane=packedIdx&3, vecIdx=packedIdx>>2;
  float entry=float(TO_STORE(entryRaw));
  float acc=entry; float r0=0.,r1=0.,r2=0.,r3=0.;
  for(int o=0;o<plan.numOuts;++o){ if(plan.outStep[o]==PW_REF_ENTRY) pwStoreOut(o,packedIdx,entry); }
  for(int s=0;s<plan.numSteps;++s){
    int kind=plan.step[s*PW_STEP_FIELDS],code=plan.step[s*PW_STEP_FIELDS+1],a=plan.step[s*PW_STEP_FIELDS+2],b=plan.step[s*PW_STEP_FIELDS+3],c=plan.step[s*PW_STEP_FIELDS+4];
    int dst=plan.step[s*PW_STEP_FIELDS+5],bc=plan.step[s*PW_STEP_FIELDS+6],bsrc=plan.step[s*PW_STEP_FIELDS+7];
    float va,vb=0.,vc=0.;
    if(a<=PW_REF_OP0)      va=pwLoad(PW_REF_OP0-a,pwNc4Idx(s,bsrc==1?bc:0,packedIdx,lane,vecIdx));
    else if(a==PW_REF_ACC) va=acc; else if(a==PW_REF_ENTRY) va=entry;
    else if(a==PW_REF_REG0) va=r0; else if(a==PW_REF_REG0-1) va=r1; else if(a==PW_REF_REG0-2) va=r2; else va=r3;
    if(b<=PW_REF_OP0)      vb=pwLoad(PW_REF_OP0-b,pwNc4Idx(s,bsrc==2?bc:0,packedIdx,lane,vecIdx));
    else if(b==PW_REF_ACC) vb=acc; else if(b==PW_REF_ENTRY) vb=entry;
    else if(b==PW_REF_REG0) vb=r0; else if(b==PW_REF_REG0-1) vb=r1; else if(b==PW_REF_REG0-2) vb=r2; else if(b==PW_REF_REG0-3) vb=r3;
    if(c<=PW_REF_OP0)      vc=pwLoad(PW_REF_OP0-c,pwNc4Idx(s,bsrc==3?bc:0,packedIdx,lane,vecIdx));
    else if(c==PW_REF_ACC) vc=acc; else if(c==PW_REF_ENTRY) vc=entry;
    else if(c==PW_REF_REG0) vc=r0; else if(c==PW_REF_REG0-1) vc=r1; else if(c==PW_REF_REG0-2) vc=r2; else if(c==PW_REF_REG0-3) vc=r3;
    if(kind==PW_KIND_BINARY)      acc=PW_ROUND(vx_binary(va,vb,code));
    else if(kind==PW_KIND_UNARY)  acc=PW_ROUND(vx_unary(va,code,plan.p0[s],plan.p1[s]));
    else if(kind==PW_KIND_ACT)    acc=PW_ROUND(vx_act(va,code,plan.p0[s],plan.p1[s]));
    else if(kind==PW_KIND_SELECT) acc=PW_ROUND(va!=0.?vb:vc);
    else                          acc=va;
    if(dst==0)r0=acc; else if(dst==1)r1=acc; else if(dst==2)r2=acc; else if(dst==3)r3=acc;
    for(int o=0;o<plan.numOuts;++o){ if(plan.outStep[o]==s) pwStoreOut(o,packedIdx,acc); }
  }
  return acc; }
float pw_apply_nc4(float entry, int packedIdx){ return pw_apply_nc4_st(entry,packedIdx); }
#else
float pw_apply_nc4_rx(float entryRaw, int packedIdx){ int lane=packedIdx&3, vecIdx=packedIdx>>2;
  float entry=float(TO_STORE(entryRaw));
  float acc=entry; float r0=0.,r1=0.,r2=0.,r3=0.;
  for(int o=0;o<plan.numOuts;++o){ if(plan.outStep[o]==PW_REF_ENTRY) pwStoreOut(o,packedIdx,entry); }
  for(int s=0;s<plan.numSteps;++s){
    int kind=plan.step[s*PW_STEP_FIELDS],code=plan.step[s*PW_STEP_FIELDS+1],a=plan.step[s*PW_STEP_FIELDS+2],b=plan.step[s*PW_STEP_FIELDS+3],c=plan.step[s*PW_STEP_FIELDS+4];
    int dst=plan.step[s*PW_STEP_FIELDS+5],bc=plan.step[s*PW_STEP_FIELDS+6],bsrc=plan.step[s*PW_STEP_FIELDS+7];
    float va,vb=0.,vc=0.;
    if(a<=PW_REF_OP0)      va=pwLoad(PW_REF_OP0-a,pwNc4Idx(s,bsrc==1?bc:0,packedIdx,lane,vecIdx));
    else if(a==PW_REF_ACC) va=acc; else if(a==PW_REF_ENTRY) va=entry;
    else if(a==PW_REF_REG0) va=r0; else if(a==PW_REF_REG0-1) va=r1; else if(a==PW_REF_REG0-2) va=r2; else va=r3;
    if(b<=PW_REF_OP0)      vb=pwLoad(PW_REF_OP0-b,pwNc4Idx(s,bsrc==2?bc:0,packedIdx,lane,vecIdx));
    else if(b==PW_REF_ACC) vb=acc; else if(b==PW_REF_ENTRY) vb=entry;
    else if(b==PW_REF_REG0) vb=r0; else if(b==PW_REF_REG0-1) vb=r1; else if(b==PW_REF_REG0-2) vb=r2; else if(b==PW_REF_REG0-3) vb=r3;
    if(c<=PW_REF_OP0)      vc=pwLoad(PW_REF_OP0-c,pwNc4Idx(s,bsrc==3?bc:0,packedIdx,lane,vecIdx));
    else if(c==PW_REF_ACC) vc=acc; else if(c==PW_REF_ENTRY) vc=entry;
    else if(c==PW_REF_REG0) vc=r0; else if(c==PW_REF_REG0-1) vc=r1; else if(c==PW_REF_REG0-2) vc=r2; else if(c==PW_REF_REG0-3) vc=r3;
    if(kind==PW_KIND_BINARY)      acc=vx_binary(va,vb,code);
    else if(kind==PW_KIND_UNARY)  acc=vx_unary(va,code,plan.p0[s],plan.p1[s]);
    else if(kind==PW_KIND_ACT)    acc=vx_act(va,code,plan.p0[s],plan.p1[s]);
    else if(kind==PW_KIND_SELECT) acc=va!=0.?vb:vc;
    else                          acc=va;
    if(dst==0)r0=acc; else if(dst==1)r1=acc; else if(dst==2)r2=acc; else if(dst==3)r3=acc;
    for(int o=0;o<plan.numOuts;++o){ if(plan.outStep[o]==s) pwStoreOut(o,packedIdx,PW_ROUND(acc)); }
  }
  return PW_ROUND(acc); }
float pw_apply_nc4(float entry, int packedIdx){ return pw_apply_nc4_rx(entry,packedIdx); }
#endif
#ifndef PW_RELAX
vec4 pw_apply4_st(vec4 entryRaw, int vecIdx){ int HW=plan.outDim[0];
  vec4 entry=pwToStore4(entryRaw);
  vec4 acc=entry; vec4 r0=vec4(0.),r1=vec4(0.),r2=vec4(0.),r3=vec4(0.);
  for(int o=0;o<plan.numOuts;++o){ if(plan.outStep[o]==PW_REF_ENTRY) pwStoreOut4(o,vecIdx,entry); }
  for(int s=0;s<plan.numSteps;++s){
    int kind=plan.step[s*PW_STEP_FIELDS],code=plan.step[s*PW_STEP_FIELDS+1],a=plan.step[s*PW_STEP_FIELDS+2],b=plan.step[s*PW_STEP_FIELDS+3],c=plan.step[s*PW_STEP_FIELDS+4];
    int dst=plan.step[s*PW_STEP_FIELDS+5],bc=plan.step[s*PW_STEP_FIELDS+6],bsrc=plan.step[s*PW_STEP_FIELDS+7];
    vec4 va,vb=vec4(0.),vc=vec4(0.);
    if(a<=PW_REF_OP0){ int m=bsrc==1?bc:0; va=pwLoadBc4(PW_REF_OP0-a,s,m,vecIdx,HW); }
    else if(a==PW_REF_ACC) va=acc; else if(a==PW_REF_ENTRY) va=entry;
    else if(a==PW_REF_REG0) va=r0; else if(a==PW_REF_REG0-1) va=r1; else if(a==PW_REF_REG0-2) va=r2; else va=r3;
    if(b<=PW_REF_OP0){ int m=bsrc==2?bc:0; vb=pwLoadBc4(PW_REF_OP0-b,s,m,vecIdx,HW); }
    else if(b==PW_REF_ACC) vb=acc; else if(b==PW_REF_ENTRY) vb=entry;
    else if(b==PW_REF_REG0) vb=r0; else if(b==PW_REF_REG0-1) vb=r1; else if(b==PW_REF_REG0-2) vb=r2; else if(b==PW_REF_REG0-3) vb=r3;
    if(c<=PW_REF_OP0){ int m=bsrc==3?bc:0; vc=pwLoadBc4(PW_REF_OP0-c,s,m,vecIdx,HW); }
    else if(c==PW_REF_ACC) vc=acc; else if(c==PW_REF_ENTRY) vc=entry;
    else if(c==PW_REF_REG0) vc=r0; else if(c==PW_REF_REG0-1) vc=r1; else if(c==PW_REF_REG0-2) vc=r2; else if(c==PW_REF_REG0-3) vc=r3;
    if(kind==PW_KIND_BINARY)      acc=pwRound4(pwBin4(va,vb,code));
    else if(kind==PW_KIND_UNARY)  acc=pwRound4(vec4(vx_unary(va.x,code,plan.p0[s],plan.p1[s]),vx_unary(va.y,code,plan.p0[s],plan.p1[s]),vx_unary(va.z,code,plan.p0[s],plan.p1[s]),vx_unary(va.w,code,plan.p0[s],plan.p1[s])));
    else if(kind==PW_KIND_ACT)    acc=pwRound4(vec4(vx_act(va.x,code,plan.p0[s],plan.p1[s]),vx_act(va.y,code,plan.p0[s],plan.p1[s]),vx_act(va.z,code,plan.p0[s],plan.p1[s]),vx_act(va.w,code,plan.p0[s],plan.p1[s])));
    else if(kind==PW_KIND_SELECT) acc=pwRound4(vec4(va.x!=0.?vb.x:vc.x,va.y!=0.?vb.y:vc.y,va.z!=0.?vb.z:vc.z,va.w!=0.?vb.w:vc.w));
    else                          acc=va;
    if(dst==0)r0=acc; else if(dst==1)r1=acc; else if(dst==2)r2=acc; else if(dst==3)r3=acc;
    for(int o=0;o<plan.numOuts;++o){ if(plan.outStep[o]==s) pwStoreOut4(o,vecIdx,acc); }
  }
  return acc; }
vec4 pw_apply4(vec4 entry, int vecIdx){ return pw_apply4_st(entry,vecIdx); }
#else
vec4 pw_apply4_rx(vec4 entryRaw, int vecIdx){ int HW=plan.outDim[0];
  vec4 entry=pwToStore4(entryRaw);
  vec4 acc=entry; vec4 r0=vec4(0.),r1=vec4(0.),r2=vec4(0.),r3=vec4(0.);
  for(int o=0;o<plan.numOuts;++o){ if(plan.outStep[o]==PW_REF_ENTRY) pwStoreOut4(o,vecIdx,entry); }
  for(int s=0;s<plan.numSteps;++s){
    int kind=plan.step[s*PW_STEP_FIELDS],code=plan.step[s*PW_STEP_FIELDS+1],a=plan.step[s*PW_STEP_FIELDS+2],b=plan.step[s*PW_STEP_FIELDS+3],c=plan.step[s*PW_STEP_FIELDS+4];
    int dst=plan.step[s*PW_STEP_FIELDS+5],bc=plan.step[s*PW_STEP_FIELDS+6],bsrc=plan.step[s*PW_STEP_FIELDS+7];
    vec4 va,vb=vec4(0.),vc=vec4(0.);
    if(a<=PW_REF_OP0){ int m=bsrc==1?bc:0; va=pwLoadBc4(PW_REF_OP0-a,s,m,vecIdx,HW); }
    else if(a==PW_REF_ACC) va=acc; else if(a==PW_REF_ENTRY) va=entry;
    else if(a==PW_REF_REG0) va=r0; else if(a==PW_REF_REG0-1) va=r1; else if(a==PW_REF_REG0-2) va=r2; else va=r3;
    if(b<=PW_REF_OP0){ int m=bsrc==2?bc:0; vb=pwLoadBc4(PW_REF_OP0-b,s,m,vecIdx,HW); }
    else if(b==PW_REF_ACC) vb=acc; else if(b==PW_REF_ENTRY) vb=entry;
    else if(b==PW_REF_REG0) vb=r0; else if(b==PW_REF_REG0-1) vb=r1; else if(b==PW_REF_REG0-2) vb=r2; else if(b==PW_REF_REG0-3) vb=r3;
    if(c<=PW_REF_OP0){ int m=bsrc==3?bc:0; vc=pwLoadBc4(PW_REF_OP0-c,s,m,vecIdx,HW); }
    else if(c==PW_REF_ACC) vc=acc; else if(c==PW_REF_ENTRY) vc=entry;
    else if(c==PW_REF_REG0) vc=r0; else if(c==PW_REF_REG0-1) vc=r1; else if(c==PW_REF_REG0-2) vc=r2; else if(c==PW_REF_REG0-3) vc=r3;
    if(kind==PW_KIND_BINARY)      acc=pwBin4(va,vb,code);
    else if(kind==PW_KIND_UNARY)  acc=vec4(vx_unary(va.x,code,plan.p0[s],plan.p1[s]),vx_unary(va.y,code,plan.p0[s],plan.p1[s]),vx_unary(va.z,code,plan.p0[s],plan.p1[s]),vx_unary(va.w,code,plan.p0[s],plan.p1[s]));
    else if(kind==PW_KIND_ACT)    acc=vec4(vx_act(va.x,code,plan.p0[s],plan.p1[s]),vx_act(va.y,code,plan.p0[s],plan.p1[s]),vx_act(va.z,code,plan.p0[s],plan.p1[s]),vx_act(va.w,code,plan.p0[s],plan.p1[s]));
    else if(kind==PW_KIND_SELECT) acc=vec4(va.x!=0.?vb.x:vc.x,va.y!=0.?vb.y:vc.y,va.z!=0.?vb.z:vc.z,va.w!=0.?vb.w:vc.w);
    else                          acc=va;
    if(dst==0)r0=acc; else if(dst==1)r1=acc; else if(dst==2)r2=acc; else if(dst==3)r3=acc;
    for(int o=0;o<plan.numOuts;++o){ if(plan.outStep[o]==s) pwStoreOut4(o,vecIdx,pwRound4(acc)); }
  }
  return pwRound4(acc); }
vec4 pw_apply4(vec4 entry, int vecIdx){ return pw_apply4_rx(entry,vecIdx); }
#endif
#endif
