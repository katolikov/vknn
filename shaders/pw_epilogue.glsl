#ifndef VX_PW_EPILOGUE_GLSL
#define VX_PW_EPILOGUE_GLSL
#include "common.glsl"
#ifndef PW_EPI_MAXSTEPS
#define PW_EPI_MAXSTEPS 8
#endif
#ifndef PW_EPI_MAXRANK
#define PW_EPI_MAXRANK 4
#endif
layout(std430, binding = PW_EPI_BASE) readonly buffer PwPlan {
  int numSteps, rank, worldFlat, pad;
  int outDim[PW_EPI_MAXRANK];
  int step[PW_EPI_MAXSTEPS*4];               // kind,code,opSlot,bcast per step
  int stride[PW_EPI_MAXSTEPS*PW_EPI_MAXRANK]; // flat operand strides
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
