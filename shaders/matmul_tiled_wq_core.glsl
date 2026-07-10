// Body of the {128,128,16} register-blocked GEMM over a packed quantized weight: the
// matmul_tiled_fast.comp micro-kernel with the B panel dequantized as it stages into LDS. The
// weight is a 2-D constant (core/quant_weights.h: k-major packed words, fp16 group scales, fp16
// outlier columns), so there is no bBase geometry: the batch offset applies to A and D only and
// no geometry SSBO is bound. Outlier columns are processed as extra K tiles after the packed loop —
// their A values gather through the same cooperative LDS staging, and their packed values are 0,
// so each contributes exactly once. Accumulation is fp32, ascending k within the packed grid, then
// the outlier tiles in oidx order.
//
// A wrapper includes matmul_tiled_wq_head.glsl, defines its decode surface, then includes this file:
//   WQ_WORD_T       the packed type of an 8-column block (uint = 4-bit, uvec2 = 8-bit)
//   WQ_ZERO         the WQ_WORD_T zero literal (edge threads stage from a zero block)
//   WQ_LOAD_W(i)    load a block; `i` is a uint WORD index into bp[] (strides stay word-valued
//                   through pc.rowWords for every format)
//   WQ_DEC(w, s)    the dequantized-integer weight value of block slot s (0..7) as float
//   WQ_BLOCK_COL(n) the word column of the block holding output column n
//   WQ_MAIN_INIT    optional statements at the top of main() in uniform control flow; empty when
//                   the wrapper leaves it undefined
//   WQ_BIAS         define to add the fused rank-1 [N] bias (binding 6) once per output store
//
// Each thread loads its block once and dequantizes all 8 adjacent-n slots against the 8 consecutive
// scales of its k-row (one scale-row base per thread, tracked incrementally across k tiles — no
// per-element word/scale refetch, no division in the k loop). The staged values are the same
// value * scale products narrowed to STORE as per-element staging produces.
#ifndef WQ_MAIN_INIT
#define WQ_MAIN_INIT
#endif

void main() {
  WQ_MAIN_INIT
  int M = pc.M, N = pc.N, K = pc.K;
  int batch = int(gl_WorkGroupID.z); // dense batch: B is shared, A/D step by whole matrices
  int aBase = batch * M * K;
  int dBase = batch * M * N;

  int tileM = int(gl_WorkGroupID.y) * TM;
  int tileN = int(gl_WorkGroupID.x) * TN;
  int tx  = int(gl_LocalInvocationID.x);
  int ty  = int(gl_LocalInvocationID.y);
  int tid = ty * TILE + tx;

  float acc[RM][RN];
  for (int i = 0; i < RM; ++i)
    for (int j = 0; j < RN; ++j)
      acc[i][j] = 0.0;

  // B-staging coordinates, invariant across k tiles: this thread owns packed block wCol of tile
  // k-row kRow.
  int kRow  = tid / BPR;    // k within the tile
  int wCol  = tid % BPR;    // block within the k-row
  int n0    = wCol * 8;     // first tile-local n of the block
  int gn0   = tileN + n0;   // first global n of the block
  int bpCol = WQ_BLOCK_COL(gn0); // word column in the packed payload
  // Scale-row base for gk = k0 + kRow, carried incrementally: each k tile advances gk % group by
  // TK, which crosses at most one group boundary when group >= TK (group is always >= 16 == TK;
  // the while keeps smaller groups correct too). The k loop then runs division-free.
  int sRow = (kRow / pc.group) * N; // (gk / group) * N
  int sRem = kRow % pc.group;       // gk % group

  for (int k0 = 0; k0 < K; k0 += TK) {
    for (int t = 0; t < (TM * TK) / (TILE * TILE); ++t) {
      int idx = tid + t * (TILE * TILE); // 0 .. TM*TK-1, laid out as m*TK + k
      int m = idx / TK, k = idx % TK;
      int gm = tileM + m, gk = k0 + k;
      As[k][m] = (gm < M && gk < K) ? a[aBase + gm * K + gk] : STORE(0.0);
    }
    // B panel: one block load + 8 consecutive scale loads dequantize the thread's whole block
    // (per-element staging refetches the same block and scale row 8x).
    {
      int gk = k0 + kRow;
      if (gk < K && gn0 + 7 < N) {
        WQ_WORD_T word = WQ_LOAD_W(gk * pc.rowWords + bpCol);
        int  scaleBase = sRow + gn0;
        Bs[kRow][n0    ] = STORE(WQ_DEC(word, 0) * float(sc[scaleBase    ]));
        Bs[kRow][n0 + 1] = STORE(WQ_DEC(word, 1) * float(sc[scaleBase + 1]));
        Bs[kRow][n0 + 2] = STORE(WQ_DEC(word, 2) * float(sc[scaleBase + 2]));
        Bs[kRow][n0 + 3] = STORE(WQ_DEC(word, 3) * float(sc[scaleBase + 3]));
        Bs[kRow][n0 + 4] = STORE(WQ_DEC(word, 4) * float(sc[scaleBase + 4]));
        Bs[kRow][n0 + 5] = STORE(WQ_DEC(word, 5) * float(sc[scaleBase + 5]));
        Bs[kRow][n0 + 6] = STORE(WQ_DEC(word, 6) * float(sc[scaleBase + 6]));
        Bs[kRow][n0 + 7] = STORE(WQ_DEC(word, 7) * float(sc[scaleBase + 7]));
      } else {
        // K edge (whole row zero) or the last partial block of the N edge (per-slot guard).
        bool rowLive = gk < K && gn0 < N;
        WQ_WORD_T word = WQ_ZERO;
        if (rowLive) word = WQ_LOAD_W(gk * pc.rowWords + bpCol);
        for (int e = 0; e < 8; ++e) {
          int   gn = gn0 + e;
          STORE v  = STORE(0.0);
          if (rowLive && gn < N) v = STORE(WQ_DEC(word, e) * float(sc[sRow + gn]));
          Bs[kRow][n0 + e] = v;
        }
      }
      sRem += TK;
      while (sRem >= pc.group) { sRem -= pc.group; sRow += N; }
    }
    barrier();
    for (int kk = 0; kk < TK; ++kk) {
      float av[RM], bv[RN];
      for (int i = 0; i < RM; ++i)
        av[i] = float(As[kk][ty * RM + i]);
      for (int j = 0; j < RN; ++j)
        bv[j] = float(Bs[kk][tx * RN + j]);
      for (int i = 0; i < RM; ++i)
        for (int j = 0; j < RN; ++j)
          acc[i][j] += av[i] * bv[j];
    }
    barrier();
  }

  // Outlier columns as extra K tiles: As gathers A at the outlier k indices, Bs stages the dense
  // fp16 rows, and the identical micro-kernel accumulates them.
  for (int j0 = 0; j0 < pc.nOut; j0 += TK) {
    // idx = tid + t * (TILE * TILE) keeps k = idx % TK fixed per thread (TK divides TILE * TILE),
    // so the outlier k index loads once per tile instead of once per staged element.
    int  kOut    = tid % TK;
    int  gjOut   = j0 + kOut;
    bool outLive = gjOut < pc.nOut;
    int  aCol    = outLive ? oi[gjOut] : 0;
    for (int t = 0; t < (TM * TK) / (TILE * TILE); ++t) {
      int m = (tid + t * (TILE * TILE)) / TK;
      int gm = tileM + m;
      As[kOut][m] = (gm < M && outLive) ? a[aBase + gm * K + aCol] : STORE(0.0);
    }
    for (int t = 0; t < (TK * TN) / (TILE * TILE); ++t) {
      int idx = tid + t * (TILE * TILE);
      int k = idx / TN, n = idx % TN;
      int gj = j0 + k, gn = tileN + n;
      Bs[k][n] = (gj < pc.nOut && gn < N) ? ov[gj * N + gn] : STORE(0.0);
    }
    barrier();
    for (int kk = 0; kk < TK; ++kk) {
      float av[RM], bv[RN];
      for (int i = 0; i < RM; ++i)
        av[i] = float(As[kk][ty * RM + i]);
      for (int j = 0; j < RN; ++j)
        bv[j] = float(Bs[kk][tx * RN + j]);
      for (int i = 0; i < RM; ++i)
        for (int j = 0; j < RN; ++j)
          acc[i][j] += av[i] * bv[j];
    }
    barrier();
  }

  for (int i = 0; i < RM; ++i) {
    int m = tileM + ty * RM + i;
    if (m >= M) continue;
    for (int j = 0; j < RN; ++j) {
      int n = tileN + tx * RN + j;
      if (n >= N) continue;
      int idx = dBase + m * N + n;
#ifdef WQ_BIAS
      float v = acc[i][j] + float(bias[n]);
#ifdef PW_EPI
      d[idx] = TO_STORE(pw_apply(v, idx));
#else
      d[idx] = TO_STORE(v);
#endif
#else
#ifdef PW_EPI
      d[idx] = TO_STORE(pw_apply(acc[i][j], idx));
#else
      d[idx] = TO_STORE(acc[i][j]);
#endif
#endif
    }
  }
}
