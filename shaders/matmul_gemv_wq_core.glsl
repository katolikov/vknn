// Body of the split-K mat-vec over a packed quantized weight: out[r,n] = sum_k A[r,k]*W[k,n] for
// M == 1 rows, where W is stored as sub-16-bit groups with fp16 scales plus a few fp16 outlier
// columns (core/quant_weights.h). The weight stream is the whole traffic of a decode-step
// projection, so narrow packing moves a fraction of the bytes the fp16 kernel does; the FMA count
// is unchanged and the kernel stays DRAM-bound, which is exactly where the packing pays.
//
// A wrapper includes matmul_gemv_wq_head.glsl, defines its decode surface, then includes this file:
//   WQ_WORD_T       the packed type of a lane's 8-output block (uint = 4-bit, uvec2 = 8-bit)
//   WQ_LOAD_W(i)    load that block; `i` is a uint WORD index into bp[] (strides stay word-valued
//                   through pc.rowWords for every format)
//   WQ_DEC_LO(w)    vec4 of dequantized-integer weight values for the block's outputs n0..n0+3
//   WQ_DEC_HI(w)    the same for n0+4..n0+7
//   WQ_BLOCK_COL(n) the word column of the block holding output n (n / 8 * words-per-block)
//   WQ_MAIN_INIT    optional statements at the top of main() in uniform control flow (a codebook
//                   format stages its table here); empty when the wrapper leaves it undefined
//   WQ_BIAS         define to add the fused rank-1 [N] bias (binding 6) once, in the reducing lane
//
// Weight addressing (k-major, n-minor): block = WQ_BLOCK_COL(n) of k row, sign-extended integer
// times the fp16 group scale sc[(k/group)*N + n]. Outlier columns carry q = 0 in the grid and are
// added from the dense fp16 rows ov[j*N + n], split over the ky lanes, so they contribute exactly
// once. A subgroup's lanes cover ADJACENT n, so their simultaneous scale reads for one group are
// consecutive fp16 halfwords of one cache line — coalesced, never strided (the scale bank is
// [nGroups * N] with n minor for exactly this read).
//
// Two specializations, both selected by conditions that are fixed per dispatch (the accumulation
// order per output element is therefore fixed run-to-run):
//  - an INTERIOR lane (n0 + 8 <= N — every lane when N is a multiple of 8, which every LLM
//    projection is) reads its 8 scales / outlier values and writes its 8 outputs unguarded; only a
//    tail lane of a non-multiple-of-8 N pays the per-element bound selects;
//  - group == WQ_U * WQ_KS (the 128 default) makes a lane's per-group k slice exactly one unrolled
//    body, so the group loop IS the unrolled loop — no inner-loop bound or tail branch per group —
//    and the NEXT group's scales issue before the current FMA chain (the only extra latency in the
//    steady state). Any other group size takes the generic per-group loops.
#ifndef WQ_MAIN_INIT
#define WQ_MAIN_INIT
#endif

// 4 adjacent group scales, unguarded — interior lanes only (n0 + 8 <= N). idx = g*N + n0 (+4).
vec4 scaleQuad(int idx) {
  return vec4(float(sc[idx]), float(sc[idx + 1]), float(sc[idx + 2]), float(sc[idx + 3]));
}

// One WQ_U-deep k slice: all 16 loads (8 activations + 8 packed blocks) issue before any FMA
// consumes them, keeping several weight requests in flight per wave. A packed block is 8 outputs in
// a handful of bytes, so this kernel has a fraction of the fp16 GEMV's natural requests per output
// and must recover the memory-level parallelism through a deep unroll (the kernel is latency-, not
// bandwidth-bound). aIdx/wIdx address a[.]/bp[.] at the slice's first k step; wStep (hoisted in
// main) is the packed-word distance between a lane's consecutive k steps.
#define WQ_LOAD8(aIdx, wIdx) \
    float a0 = float(a[(aIdx)]); \
    float a1 = float(a[(aIdx) + WQ_KS]); \
    float a2 = float(a[(aIdx) + 2 * WQ_KS]); \
    float a3 = float(a[(aIdx) + 3 * WQ_KS]); \
    float a4 = float(a[(aIdx) + 4 * WQ_KS]); \
    float a5 = float(a[(aIdx) + 5 * WQ_KS]); \
    float a6 = float(a[(aIdx) + 6 * WQ_KS]); \
    float a7 = float(a[(aIdx) + 7 * WQ_KS]); \
    WQ_WORD_T w0 = WQ_LOAD_W((wIdx)); \
    WQ_WORD_T w1 = WQ_LOAD_W((wIdx) + wStep); \
    WQ_WORD_T w2 = WQ_LOAD_W((wIdx) + 2 * wStep); \
    WQ_WORD_T w3 = WQ_LOAD_W((wIdx) + 3 * wStep); \
    WQ_WORD_T w4 = WQ_LOAD_W((wIdx) + 4 * wStep); \
    WQ_WORD_T w5 = WQ_LOAD_W((wIdx) + 5 * wStep); \
    WQ_WORD_T w6 = WQ_LOAD_W((wIdx) + 6 * wStep); \
    WQ_WORD_T w7 = WQ_LOAD_W((wIdx) + 7 * wStep);
#define WQ_FMA8(sLo, sHi) \
    acc0 += a0 * WQ_DEC_LO(w0) * (sLo); acc1 += a0 * WQ_DEC_HI(w0) * (sHi); \
    acc0 += a1 * WQ_DEC_LO(w1) * (sLo); acc1 += a1 * WQ_DEC_HI(w1) * (sHi); \
    acc0 += a2 * WQ_DEC_LO(w2) * (sLo); acc1 += a2 * WQ_DEC_HI(w2) * (sHi); \
    acc0 += a3 * WQ_DEC_LO(w3) * (sLo); acc1 += a3 * WQ_DEC_HI(w3) * (sHi); \
    acc0 += a4 * WQ_DEC_LO(w4) * (sLo); acc1 += a4 * WQ_DEC_HI(w4) * (sHi); \
    acc0 += a5 * WQ_DEC_LO(w5) * (sLo); acc1 += a5 * WQ_DEC_HI(w5) * (sHi); \
    acc0 += a6 * WQ_DEC_LO(w6) * (sLo); acc1 += a6 * WQ_DEC_HI(w6) * (sHi); \
    acc0 += a7 * WQ_DEC_LO(w7) * (sLo); acc1 += a7 * WQ_DEC_HI(w7) * (sHi);
// One k step on the group scales held in s0/s1.
#define WQ_STEP1(aIdx, wIdx) { \
    float ak = float(a[(aIdx)]); \
    WQ_WORD_T wv = WQ_LOAD_W((wIdx)); \
    acc0 += ak * WQ_DEC_LO(wv) * s0; \
    acc1 += ak * WQ_DEC_HI(wv) * s1; \
  }

void main() {
  WQ_MAIN_INIT
  int row = int(gl_WorkGroupID.y);                              // output row (flattened batch, M == 1)
  int n0  = (int(gl_WorkGroupID.x) * WQ_NX + int(gl_LocalInvocationID.x)) * 8;
  int ky  = int(gl_LocalInvocationID.y);
  bool live  = n0 < pc.N;
  bool full8 = n0 + 8 <= pc.N; // this lane's whole 8-output block is interior
  int aBase = row * pc.K;
  int wcol  = WQ_BLOCK_COL(n0); // this lane's packed word column
  int wStep = WQ_KS * pc.rowWords;

  // Guarded per-column scale fetch: the tail block past N reads 0, so its padding values stay 0.
  #define SCALE4(g4, off) vec4( \
      n0 + off + 0 < pc.N ? float(sc[(g4) * pc.N + n0 + off + 0]) : 0.0, \
      n0 + off + 1 < pc.N ? float(sc[(g4) * pc.N + n0 + off + 1]) : 0.0, \
      n0 + off + 2 < pc.N ? float(sc[(g4) * pc.N + n0 + off + 2]) : 0.0, \
      n0 + off + 3 < pc.N ? float(sc[(g4) * pc.N + n0 + off + 3]) : 0.0)

  vec4 acc0 = vec4(0.0), acc1 = vec4(0.0);
  if (live) {
    if (full8 && pc.group == WQ_U * WQ_KS) {
      // Whole-group path: each full group is exactly one WQ_U-deep slice per lane.
      int nFull = pc.K / pc.group; // groups whose slice is a complete unrolled body
      vec4 s0 = scaleQuad(n0), s1 = scaleQuad(n0 + 4);
      int aIdx = aBase + ky;
      int wIdx = ky * pc.rowWords + wcol;
      int wGroupStep = pc.group * pc.rowWords;
      for (int g = 0; g < nFull; ++g) {
        WQ_LOAD8(aIdx, wIdx)
        // The next group's scales issue here, before the FMA chain consumes this group's loads
        // (software pipelining); min() keeps the last prefetch in bounds and leaves the trailing
        // partial group's scales in s0/s1 for the loop below.
        int scNext = min(g + 1, pc.nGroups - 1) * pc.N + n0;
        vec4 sn0 = scaleQuad(scNext), sn1 = scaleQuad(scNext + 4);
        WQ_FMA8(s0, s1)
        s0 = sn0; s1 = sn1;
        aIdx += pc.group;
        wIdx += wGroupStep;
      }
      // Trailing partial group (K % group != 0): strided single steps on the prefetched scales.
      for (int k = nFull * pc.group + ky; k < pc.K; k += WQ_KS)
        WQ_STEP1(aBase + k, k * pc.rowWords + wcol)
    } else {
      // Generic path: any group size >= 1, and the tail lane of a non-multiple-of-8 N.
      for (int g = 0; g < pc.nGroups; ++g) {
        vec4 s0, s1;
        if (full8) {
          int scIdx = g * pc.N + n0;
          s0 = scaleQuad(scIdx); s1 = scaleQuad(scIdx + 4);
        } else {
          s0 = SCALE4(g, 0); s1 = SCALE4(g, 4);
        }
        int kEnd = min(pc.K, (g + 1) * pc.group);
        int k = g * pc.group + ky;
        for (; k + (WQ_U - 1) * WQ_KS < kEnd; k += WQ_U * WQ_KS) {
          WQ_LOAD8(aBase + k, k * pc.rowWords + wcol)
          WQ_FMA8(s0, s1)
        }
        for (; k < kEnd; k += WQ_KS)
          WQ_STEP1(aBase + k, k * pc.rowWords + wcol)
      }
    }
    // Outlier columns, split over the ky lanes like the k grid (serializing them in the reducing
    // lane would cost as much as the whole packed loop): lane ky adds the j == ky (mod WQ_KS) rows
    // into its partial, so the reduce below folds them exactly once, in a fixed order.
    if (full8) {
      for (int j = int(ky); j < pc.nOut; j += WQ_KS) {
        float av = float(a[aBase + oi[j]]);
        int ovIdx = j * pc.N + n0;
        acc0 += av * vec4(float(ov[ovIdx]),     float(ov[ovIdx + 1]),
                          float(ov[ovIdx + 2]), float(ov[ovIdx + 3]));
        acc1 += av * vec4(float(ov[ovIdx + 4]), float(ov[ovIdx + 5]),
                          float(ov[ovIdx + 6]), float(ov[ovIdx + 7]));
      }
    } else {
      for (int j = int(ky); j < pc.nOut; j += WQ_KS) {
        float av = float(a[aBase + oi[j]]);
        for (int t = 0; t < 4; ++t) {
          if (n0 + t < pc.N)     acc0[t] += av * float(ov[j * pc.N + n0 + t]);
          if (n0 + 4 + t < pc.N) acc1[t] += av * float(ov[j * pc.N + n0 + 4 + t]);
        }
      }
    }
  }
  // Every invocation stores (dead lanes store 0) so barrier() stays in uniform control flow.
  part0[ky][gl_LocalInvocationID.x] = acc0;
  part1[ky][gl_LocalInvocationID.x] = acc1;
  barrier();
  if (ky != 0 || !live) return;

  vec4 s0 = vec4(0.0), s1 = vec4(0.0);
  for (int i = 0; i < WQ_KS; ++i) {
    s0 += part0[i][gl_LocalInvocationID.x];
    s1 += part1[i][gl_LocalInvocationID.x];
  }
  int dBase = row * pc.N;
  if (full8) {
#ifdef WQ_BIAS
    s0 += vec4(float(bias[n0]),     float(bias[n0 + 1]),
               float(bias[n0 + 2]), float(bias[n0 + 3]));
    s1 += vec4(float(bias[n0 + 4]), float(bias[n0 + 5]),
               float(bias[n0 + 6]), float(bias[n0 + 7]));
#endif
    for (int t = 0; t < 4; ++t) {
      int n = n0 + t;
#ifdef PW_EPI
      d[dBase + n] = TO_STORE(pw_apply(s0[t], dBase + n));
#else
      d[dBase + n] = TO_STORE(s0[t]);
#endif
      n = n0 + 4 + t;
#ifdef PW_EPI
      d[dBase + n] = TO_STORE(pw_apply(s1[t], dBase + n));
#else
      d[dBase + n] = TO_STORE(s1[t]);
#endif
    }
  } else {
#ifdef WQ_BIAS
    for (int t = 0; t < 4; ++t) {
      if (n0 + t < pc.N)     s0[t] += float(bias[n0 + t]);
      if (n0 + 4 + t < pc.N) s1[t] += float(bias[n0 + 4 + t]);
    }
#endif
    for (int t = 0; t < 4; ++t) {
      int n = n0 + t;
      if (n < pc.N) {
#ifdef PW_EPI
        d[dBase + n] = TO_STORE(pw_apply(s0[t], dBase + n));
#else
        d[dBase + n] = TO_STORE(s0[t]);
#endif
      }
      n = n0 + 4 + t;
      if (n < pc.N) {
#ifdef PW_EPI
        d[dBase + n] = TO_STORE(pw_apply(s1[t], dBase + n));
#else
        d[dBase + n] = TO_STORE(s1[t]);
#endif
      }
    }
  }
}
