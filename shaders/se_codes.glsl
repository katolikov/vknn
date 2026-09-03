// The fused Squeeze-Excite kernel's activation and gate switch. The including shader defines the
// SE_ACT_* / SE_GATE_* codes first (they mirror kSeAct* / kSeGate* in src/core/squeeze_excite.h,
// and live in the .comp so tools/check_shader_contracts.py can pin them); this file supplies the
// functions that branch on them.
#ifndef SE_CODES_GLSL
#define SE_CODES_GLSL

// FC1 activation: Relu / SiLU / HardSwish.
float seAct(int act, float x) {
  if (act == SE_ACT_SILU) return x / (1.0 + exp(-x));
  if (act == SE_ACT_HARDSWISH) return x * clamp(x + 3.0, 0.0, 6.0) / 6.0;
  return max(x, 0.0);
}

// Gate: HardSigmoid(alpha, beta) / Sigmoid.
float seGate(int gate, float x, float alpha, float beta) {
  if (gate == SE_GATE_SIGMOID) return 1.0 / (1.0 + exp(-x));
  return clamp(alpha * x + beta, 0.0, 1.0);
}

#endif
