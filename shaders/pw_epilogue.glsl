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
#define PW_EPI_MAXSTEPS 16
#endif
#ifndef PW_EPI_MAXRANK
#define PW_EPI_MAXRANK 4
#endif
// Value references inside a step (mirrors kPwRef* in include/vknn/op_type.h): -1 accumulator,
// -2 entry value, -3 none, -4-r register r, -8-i operand slot i.
#define PW_REF_ACC   (-1)
#define PW_REF_ENTRY (-2)
#define PW_REF_NONE  (-3)
#define PW_REF_REG0  (-4)
#define PW_REF_OP0   (-8)
// flags bit 0 records the fp32-chained discipline for the plan (set from the pw_relax attr). The
// discipline is selected at COMPILE time: kernels built with -DPW_RELAX contain only the
// fp32-chained appliers, kernels built without it only the strict per-step-rounded ones — a
// runtime branch would carry both bodies in every kernel and cost occupancy.
#define PW_FLAG_CHAIN32 1
layout(std430, binding = PW_EPI_BASE) readonly buffer PwPlan {
  int numSteps, rank, worldFlat, numOuts, flags;
  int outDim[PW_EPI_MAXRANK];
  int step[PW_EPI_MAXSTEPS*8];                // kind(0 bin,1 unary,2 act,3 select,4 load), code, srcA, srcB, srcC, dst, bcast, bcastSrc
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
layout(std430, binding = PW_EPI_BASE+7)  writeonly buffer PwOut1 { STORE d[]; } pwout1;
layout(std430, binding = PW_EPI_BASE+8)  writeonly buffer PwOut2 { STORE d[]; } pwout2;
layout(std430, binding = PW_EPI_BASE+9)  writeonly buffer PwOut3 { STORE d[]; } pwout3;
layout(std430, binding = PW_EPI_BASE+10) writeonly buffer PwOut4 { STORE d[]; } pwout4;
float pwLoad(int slot,int idx){ if(slot==0)return float(pwop1.d[idx]); if(slot==1)return float(pwop2.d[idx]);
  if(slot==2)return float(pwop3.d[idx]); if(slot==3)return float(pwop4.d[idx]); if(slot==4)return float(pwop5.d[idx]);
  return float(pwop6.d[idx]); }
vec4 pwLoad4(int slot,int idx){ if(slot==0)return vec4(pwop1.d[idx*4],pwop1.d[idx*4+1],pwop1.d[idx*4+2],pwop1.d[idx*4+3]);
  if(slot==1)return vec4(pwop2.d[idx*4],pwop2.d[idx*4+1],pwop2.d[idx*4+2],pwop2.d[idx*4+3]);
  if(slot==2)return vec4(pwop3.d[idx*4],pwop3.d[idx*4+1],pwop3.d[idx*4+2],pwop3.d[idx*4+3]);
  if(slot==3)return vec4(pwop4.d[idx*4],pwop4.d[idx*4+1],pwop4.d[idx*4+2],pwop4.d[idx*4+3]);
  if(slot==4)return vec4(pwop5.d[idx*4],pwop5.d[idx*4+1],pwop5.d[idx*4+2],pwop5.d[idx*4+3]);
  return vec4(pwop6.d[idx*4],pwop6.d[idx*4+1],pwop6.d[idx*4+2],pwop6.d[idx*4+3]); }
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
int pwFlatIdx(int s,int bc,int outIdx){ if(bc==0) return outIdx;
  // A per-pixel operand is the trailing H*W plane of a single-batch run, so it indexes by the
  // spatial remainder — the same value the strided walk below yields, without its div/mod chain.
  if(bc==4) return outIdx % (plan.outDim[plan.rank-2]*plan.outDim[plan.rank-1]);
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
    int kind=plan.step[s*8],code=plan.step[s*8+1],a=plan.step[s*8+2],b=plan.step[s*8+3],c=plan.step[s*8+4];
    int dst=plan.step[s*8+5],bc=plan.step[s*8+6],bsrc=plan.step[s*8+7];
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
    if(kind==0)      acc=PW_ROUND(vx_binary(va,vb,code));
    else if(kind==1) acc=PW_ROUND(vx_unary(va,code,plan.p0[s],plan.p1[s]));
    else if(kind==2) acc=PW_ROUND(vx_act(va,code,plan.p0[s],plan.p1[s]));
    else if(kind==3) acc=PW_ROUND(va!=0.?vb:vc);
    else             acc=va;
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
    int kind=plan.step[s*8],code=plan.step[s*8+1],a=plan.step[s*8+2],b=plan.step[s*8+3],c=plan.step[s*8+4];
    int dst=plan.step[s*8+5],bc=plan.step[s*8+6],bsrc=plan.step[s*8+7];
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
    if(kind==0)      acc=vx_binary(va,vb,code);
    else if(kind==1) acc=vx_unary(va,code,plan.p0[s],plan.p1[s]);
    else if(kind==2) acc=vx_act(va,code,plan.p0[s],plan.p1[s]);
    else if(kind==3) acc=va!=0.?vb:vc;
    else             acc=va;
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
// One vec4 operand fetch for the NC4HW4 kernel, by broadcast class: a scalar and a per-pixel
// operand both hold ONE value for the four channel lanes and splat it, while same-shape and
// per-channel operands hold four distinct lane values and load them as a vec4.
vec4 pwLoadBc4(int slot,int m,int vecIdx,int HW){
  return (m==3)?vec4(pwLoad(slot,0)):(m==4)?vec4(pwLoad(slot,(vecIdx%HW)*4)):pwLoad4(slot,(m==1)?vecIdx/HW:vecIdx); }
int pwNc4Idx(int bc,int packedIdx,int lane,int vecIdx){ int HW=plan.outDim[0];
  return (bc==3)?0:(bc==1)?(vecIdx/HW)*4+lane:(bc==4)?(vecIdx%HW)*4:packedIdx; }
#ifndef PW_RELAX
float pw_apply_nc4_st(float entryRaw, int packedIdx){ int lane=packedIdx&3, vecIdx=packedIdx>>2;
  float entry=float(TO_STORE(entryRaw));
  float acc=entry; float r0=0.,r1=0.,r2=0.,r3=0.;
  for(int o=0;o<plan.numOuts;++o){ if(plan.outStep[o]==PW_REF_ENTRY) pwStoreOut(o,packedIdx,entry); }
  for(int s=0;s<plan.numSteps;++s){
    int kind=plan.step[s*8],code=plan.step[s*8+1],a=plan.step[s*8+2],b=plan.step[s*8+3],c=plan.step[s*8+4];
    int dst=plan.step[s*8+5],bc=plan.step[s*8+6],bsrc=plan.step[s*8+7];
    float va,vb=0.,vc=0.;
    if(a<=PW_REF_OP0)      va=pwLoad(PW_REF_OP0-a,pwNc4Idx(bsrc==1?bc:0,packedIdx,lane,vecIdx));
    else if(a==PW_REF_ACC) va=acc; else if(a==PW_REF_ENTRY) va=entry;
    else if(a==PW_REF_REG0) va=r0; else if(a==PW_REF_REG0-1) va=r1; else if(a==PW_REF_REG0-2) va=r2; else va=r3;
    if(b<=PW_REF_OP0)      vb=pwLoad(PW_REF_OP0-b,pwNc4Idx(bsrc==2?bc:0,packedIdx,lane,vecIdx));
    else if(b==PW_REF_ACC) vb=acc; else if(b==PW_REF_ENTRY) vb=entry;
    else if(b==PW_REF_REG0) vb=r0; else if(b==PW_REF_REG0-1) vb=r1; else if(b==PW_REF_REG0-2) vb=r2; else if(b==PW_REF_REG0-3) vb=r3;
    if(c<=PW_REF_OP0)      vc=pwLoad(PW_REF_OP0-c,pwNc4Idx(bsrc==3?bc:0,packedIdx,lane,vecIdx));
    else if(c==PW_REF_ACC) vc=acc; else if(c==PW_REF_ENTRY) vc=entry;
    else if(c==PW_REF_REG0) vc=r0; else if(c==PW_REF_REG0-1) vc=r1; else if(c==PW_REF_REG0-2) vc=r2; else if(c==PW_REF_REG0-3) vc=r3;
    if(kind==0)      acc=PW_ROUND(vx_binary(va,vb,code));
    else if(kind==1) acc=PW_ROUND(vx_unary(va,code,plan.p0[s],plan.p1[s]));
    else if(kind==2) acc=PW_ROUND(vx_act(va,code,plan.p0[s],plan.p1[s]));
    else if(kind==3) acc=PW_ROUND(va!=0.?vb:vc);
    else             acc=va;
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
    int kind=plan.step[s*8],code=plan.step[s*8+1],a=plan.step[s*8+2],b=plan.step[s*8+3],c=plan.step[s*8+4];
    int dst=plan.step[s*8+5],bc=plan.step[s*8+6],bsrc=plan.step[s*8+7];
    float va,vb=0.,vc=0.;
    if(a<=PW_REF_OP0)      va=pwLoad(PW_REF_OP0-a,pwNc4Idx(bsrc==1?bc:0,packedIdx,lane,vecIdx));
    else if(a==PW_REF_ACC) va=acc; else if(a==PW_REF_ENTRY) va=entry;
    else if(a==PW_REF_REG0) va=r0; else if(a==PW_REF_REG0-1) va=r1; else if(a==PW_REF_REG0-2) va=r2; else va=r3;
    if(b<=PW_REF_OP0)      vb=pwLoad(PW_REF_OP0-b,pwNc4Idx(bsrc==2?bc:0,packedIdx,lane,vecIdx));
    else if(b==PW_REF_ACC) vb=acc; else if(b==PW_REF_ENTRY) vb=entry;
    else if(b==PW_REF_REG0) vb=r0; else if(b==PW_REF_REG0-1) vb=r1; else if(b==PW_REF_REG0-2) vb=r2; else if(b==PW_REF_REG0-3) vb=r3;
    if(c<=PW_REF_OP0)      vc=pwLoad(PW_REF_OP0-c,pwNc4Idx(bsrc==3?bc:0,packedIdx,lane,vecIdx));
    else if(c==PW_REF_ACC) vc=acc; else if(c==PW_REF_ENTRY) vc=entry;
    else if(c==PW_REF_REG0) vc=r0; else if(c==PW_REF_REG0-1) vc=r1; else if(c==PW_REF_REG0-2) vc=r2; else if(c==PW_REF_REG0-3) vc=r3;
    if(kind==0)      acc=vx_binary(va,vb,code);
    else if(kind==1) acc=vx_unary(va,code,plan.p0[s],plan.p1[s]);
    else if(kind==2) acc=vx_act(va,code,plan.p0[s],plan.p1[s]);
    else if(kind==3) acc=va!=0.?vb:vc;
    else             acc=va;
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
    int kind=plan.step[s*8],code=plan.step[s*8+1],a=plan.step[s*8+2],b=plan.step[s*8+3],c=plan.step[s*8+4];
    int dst=plan.step[s*8+5],bc=plan.step[s*8+6],bsrc=plan.step[s*8+7];
    vec4 va,vb=vec4(0.),vc=vec4(0.);
    if(a<=PW_REF_OP0){ int m=bsrc==1?bc:0; va=pwLoadBc4(PW_REF_OP0-a,m,vecIdx,HW); }
    else if(a==PW_REF_ACC) va=acc; else if(a==PW_REF_ENTRY) va=entry;
    else if(a==PW_REF_REG0) va=r0; else if(a==PW_REF_REG0-1) va=r1; else if(a==PW_REF_REG0-2) va=r2; else va=r3;
    if(b<=PW_REF_OP0){ int m=bsrc==2?bc:0; vb=pwLoadBc4(PW_REF_OP0-b,m,vecIdx,HW); }
    else if(b==PW_REF_ACC) vb=acc; else if(b==PW_REF_ENTRY) vb=entry;
    else if(b==PW_REF_REG0) vb=r0; else if(b==PW_REF_REG0-1) vb=r1; else if(b==PW_REF_REG0-2) vb=r2; else if(b==PW_REF_REG0-3) vb=r3;
    if(c<=PW_REF_OP0){ int m=bsrc==3?bc:0; vc=pwLoadBc4(PW_REF_OP0-c,m,vecIdx,HW); }
    else if(c==PW_REF_ACC) vc=acc; else if(c==PW_REF_ENTRY) vc=entry;
    else if(c==PW_REF_REG0) vc=r0; else if(c==PW_REF_REG0-1) vc=r1; else if(c==PW_REF_REG0-2) vc=r2; else if(c==PW_REF_REG0-3) vc=r3;
    if(kind==0)      acc=pwRound4(pwBin4(va,vb,code));
    else if(kind==1) acc=pwRound4(vec4(vx_unary(va.x,code,plan.p0[s],plan.p1[s]),vx_unary(va.y,code,plan.p0[s],plan.p1[s]),vx_unary(va.z,code,plan.p0[s],plan.p1[s]),vx_unary(va.w,code,plan.p0[s],plan.p1[s])));
    else if(kind==2) acc=pwRound4(vec4(vx_act(va.x,code,plan.p0[s],plan.p1[s]),vx_act(va.y,code,plan.p0[s],plan.p1[s]),vx_act(va.z,code,plan.p0[s],plan.p1[s]),vx_act(va.w,code,plan.p0[s],plan.p1[s])));
    else if(kind==3) acc=pwRound4(vec4(va.x!=0.?vb.x:vc.x,va.y!=0.?vb.y:vc.y,va.z!=0.?vb.z:vc.z,va.w!=0.?vb.w:vc.w));
    else             acc=va;
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
    int kind=plan.step[s*8],code=plan.step[s*8+1],a=plan.step[s*8+2],b=plan.step[s*8+3],c=plan.step[s*8+4];
    int dst=plan.step[s*8+5],bc=plan.step[s*8+6],bsrc=plan.step[s*8+7];
    vec4 va,vb=vec4(0.),vc=vec4(0.);
    if(a<=PW_REF_OP0){ int m=bsrc==1?bc:0; va=pwLoadBc4(PW_REF_OP0-a,m,vecIdx,HW); }
    else if(a==PW_REF_ACC) va=acc; else if(a==PW_REF_ENTRY) va=entry;
    else if(a==PW_REF_REG0) va=r0; else if(a==PW_REF_REG0-1) va=r1; else if(a==PW_REF_REG0-2) va=r2; else va=r3;
    if(b<=PW_REF_OP0){ int m=bsrc==2?bc:0; vb=pwLoadBc4(PW_REF_OP0-b,m,vecIdx,HW); }
    else if(b==PW_REF_ACC) vb=acc; else if(b==PW_REF_ENTRY) vb=entry;
    else if(b==PW_REF_REG0) vb=r0; else if(b==PW_REF_REG0-1) vb=r1; else if(b==PW_REF_REG0-2) vb=r2; else if(b==PW_REF_REG0-3) vb=r3;
    if(c<=PW_REF_OP0){ int m=bsrc==3?bc:0; vc=pwLoadBc4(PW_REF_OP0-c,m,vecIdx,HW); }
    else if(c==PW_REF_ACC) vc=acc; else if(c==PW_REF_ENTRY) vc=entry;
    else if(c==PW_REF_REG0) vc=r0; else if(c==PW_REF_REG0-1) vc=r1; else if(c==PW_REF_REG0-2) vc=r2; else if(c==PW_REF_REG0-3) vc=r3;
    if(kind==0)      acc=pwBin4(va,vb,code);
    else if(kind==1) acc=vec4(vx_unary(va.x,code,plan.p0[s],plan.p1[s]),vx_unary(va.y,code,plan.p0[s],plan.p1[s]),vx_unary(va.z,code,plan.p0[s],plan.p1[s]),vx_unary(va.w,code,plan.p0[s],plan.p1[s]));
    else if(kind==2) acc=vec4(vx_act(va.x,code,plan.p0[s],plan.p1[s]),vx_act(va.y,code,plan.p0[s],plan.p1[s]),vx_act(va.z,code,plan.p0[s],plan.p1[s]),vx_act(va.w,code,plan.p0[s],plan.p1[s]));
    else if(kind==3) acc=vec4(va.x!=0.?vb.x:vc.x,va.y!=0.?vb.y:vc.y,va.z!=0.?vb.z:vc.z,va.w!=0.?vb.w:vc.w);
    else             acc=va;
    if(dst==0)r0=acc; else if(dst==1)r1=acc; else if(dst==2)r2=acc; else if(dst==3)r3=acc;
    for(int o=0;o<plan.numOuts;++o){ if(plan.outStep[o]==s) pwStoreOut4(o,vecIdx,pwRound4(acc)); }
  }
  return pwRound4(acc); }
vec4 pw_apply4(vec4 entry, int vecIdx){ return pw_apply4_rx(entry,vecIdx); }
#endif
#endif
