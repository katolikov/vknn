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
layout(std430, binding = PW_EPI_BASE) readonly buffer PwPlan {
  int numSteps, rank, worldFlat, numOuts;
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
  int rem=outIdx; int oi=0; for(int k=plan.rank-1;k>=0;--k){ int c=rem%plan.outDim[k]; rem/=plan.outDim[k]; oi+=c*plan.stride[s*PW_EPI_MAXRANK+k]; } return oi; }
// The three appliers share one execution model: acc starts at the entry value; every step's result
// lands in acc, rounded through TO_STORE for compute kinds so each fused step reproduces, bit for
// bit, the fp16 store the unfused graph would make (a load passes the already-rounded storage value
// through unrounded); dst >= 0 additionally keeps a copy in one of 4 registers for later steps; and
// after each step the exported values (plan.outStep) store to the extra output streams.
float pw_apply(float entry, int outIdx){
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
    if(kind==0)      acc=float(TO_STORE(vx_binary(va,vb,code)));
    else if(kind==1) acc=float(TO_STORE(vx_unary(va,code,plan.p0[s],plan.p1[s])));
    else if(kind==2) acc=float(TO_STORE(vx_act(va,code,plan.p0[s],plan.p1[s])));
    else if(kind==3) acc=float(TO_STORE(va!=0.?vb:vc));
    else             acc=va;
    if(dst==0)r0=acc; else if(dst==1)r1=acc; else if(dst==2)r2=acc; else if(dst==3)r3=acc;
    for(int o=0;o<plan.numOuts;++o){ if(plan.outStep[o]==s) pwStoreOut(o,outIdx,acc); }
  }
  return acc; }
// Scalar NC4HW4 store: packedIdx = ((n*Cb+cb)*HW + hw)*4 + lane. Operands are NC4HW4-packed
// (runtime activations natively; constants packed at upload), so bc==0 (same shape) loads at
// packedIdx, bc==1 (per-channel [N,C,1,1]) at the store's channel-block lane, bc==3 at element 0.
int pwNc4Idx(int bc,int packedIdx,int lane,int vecIdx){ int HW=plan.outDim[0];
  return (bc==3)?0:(bc==1)?(vecIdx/HW)*4+lane:packedIdx; }
float pw_apply_nc4(float entry, int packedIdx){ int lane=packedIdx&3, vecIdx=packedIdx>>2;
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
    if(kind==0)      acc=float(TO_STORE(vx_binary(va,vb,code)));
    else if(kind==1) acc=float(TO_STORE(vx_unary(va,code,plan.p0[s],plan.p1[s])));
    else if(kind==2) acc=float(TO_STORE(vx_act(va,code,plan.p0[s],plan.p1[s])));
    else if(kind==3) acc=float(TO_STORE(va!=0.?vb:vc));
    else             acc=va;
    if(dst==0)r0=acc; else if(dst==1)r1=acc; else if(dst==2)r2=acc; else if(dst==3)r3=acc;
    for(int o=0;o<plan.numOuts;++o){ if(plan.outStep[o]==s) pwStoreOut(o,packedIdx,acc); }
  }
  return acc; }
vec4 pwBin4(vec4 a,vec4 b,int code){ return vec4(vx_binary(a.x,b.x,code),vx_binary(a.y,b.y,code),vx_binary(a.z,b.z,code),vx_binary(a.w,b.w,code)); }
vec4 pwRound4(vec4 v){ return vec4(float(TO_STORE(v.x)),float(TO_STORE(v.y)),float(TO_STORE(v.z)),float(TO_STORE(v.w))); }
vec4 pw_apply4(vec4 entry, int vecIdx){ int HW=plan.outDim[0];
  vec4 acc=entry; vec4 r0=vec4(0.),r1=vec4(0.),r2=vec4(0.),r3=vec4(0.);
  for(int o=0;o<plan.numOuts;++o){ if(plan.outStep[o]==PW_REF_ENTRY) pwStoreOut4(o,vecIdx,entry); }
  for(int s=0;s<plan.numSteps;++s){
    int kind=plan.step[s*8],code=plan.step[s*8+1],a=plan.step[s*8+2],b=plan.step[s*8+3],c=plan.step[s*8+4];
    int dst=plan.step[s*8+5],bc=plan.step[s*8+6],bsrc=plan.step[s*8+7];
    vec4 va,vb=vec4(0.),vc=vec4(0.);
    if(a<=PW_REF_OP0){ int m=bsrc==1?bc:0; va=(m==3)?vec4(pwLoad(PW_REF_OP0-a,0)):pwLoad4(PW_REF_OP0-a,(m==1)?vecIdx/HW:vecIdx); }
    else if(a==PW_REF_ACC) va=acc; else if(a==PW_REF_ENTRY) va=entry;
    else if(a==PW_REF_REG0) va=r0; else if(a==PW_REF_REG0-1) va=r1; else if(a==PW_REF_REG0-2) va=r2; else va=r3;
    if(b<=PW_REF_OP0){ int m=bsrc==2?bc:0; vb=(m==3)?vec4(pwLoad(PW_REF_OP0-b,0)):pwLoad4(PW_REF_OP0-b,(m==1)?vecIdx/HW:vecIdx); }
    else if(b==PW_REF_ACC) vb=acc; else if(b==PW_REF_ENTRY) vb=entry;
    else if(b==PW_REF_REG0) vb=r0; else if(b==PW_REF_REG0-1) vb=r1; else if(b==PW_REF_REG0-2) vb=r2; else if(b==PW_REF_REG0-3) vb=r3;
    if(c<=PW_REF_OP0){ int m=bsrc==3?bc:0; vc=(m==3)?vec4(pwLoad(PW_REF_OP0-c,0)):pwLoad4(PW_REF_OP0-c,(m==1)?vecIdx/HW:vecIdx); }
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
#endif
