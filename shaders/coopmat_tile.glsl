// Shared cooperative-matrix tile core for LDS-staged GEMM-shaped kernels (conv_gemm_cm.comp and
// any future coopmat consumer whose operands are not dense row-major fp16 in global memory —
// NC4HW4 convs, transform-domain Winograd, dequantized sub-byte weights). The direct-load GEMM
// (coopmat_gemm.comp) predates this include and keeps its own body: its operands need no staging.
//
// Contract for the including kernel:
//   - one pinned subgroup per workgroup at the device's native compute width (32 or 64):
//     layout(local_size_x = 32, local_size_x_id = 0) with the host passing the width as spec
//     constant 0 AND as requiredSubgroupSize (the routing rule gates on coopmatSubgroupWidth);
//     panel-fill loops stride by gl_WorkGroupSize.x so both widths cover the tile;
//   - the coopmat extension block declared before this include (GL_KHR_cooperative_matrix,
//     GL_KHR_memory_scope_semantics, 16bit storage + explicit fp16 arithmetic);
//   - panels staged into the shared arrays below by any gather/dequant/layout adaptation, with
//     out-of-range entries written as zero (a zero A/B contribution is exact, so edge tiles need
//     masking only at the scatter);
//   - the fragment loop driven as
//       coopTileAccClear();
//       for (each 16-deep K chunk) { <fill cmA/cmB> barrier(); coopTileStep(); barrier(); }
//       coopTileStoreAcc(); barrier(); <read cmD, apply bias/act, scatter to the output layout>
//
// One workgroup owns a COOP_TM x COOP_TN (32x32) output tile as a 2x2 grid of 16x16x16 fragments;
// K advances COOP_TK = 16 per step. Accumulation is fp32 on the matrix units; cmD holds the fp32
// tile so bias/act apply before the kernel's single TO_STORE narrowing, matching the SSBO conv
// contract (bias added in fp32, one rounding at the store).
//
// Panel layouts (fp16 scalars; A is k-major so an NC4HW4 / im2col / dequant gather lands without
// a transpose):
//   cmA[k][m] : [COOP_TK][COOP_TM] — A fragments load column-major at stride COOP_TM
//   cmB[k][n] : [COOP_TK][COOP_TN] — B fragments load row-major at stride COOP_TN
//   cmD[m][n] : [COOP_TM][COOP_TN] fp32 accumulator tile
// The fragment-to-lane mapping stays opaque (loads/stores only via coopMatLoad/coopMatStore with
// explicit layouts), and each consumer kernel is gated by its own one-time on-device exact
// self-check (coopmat_check.cpp) that disables the path on any driver whose result mismatches.

#define COOP_TM 32
#define COOP_TN 32
#define COOP_TK 16

shared float16_t cmA[COOP_TK * COOP_TM]; // 1 KB
shared float16_t cmB[COOP_TK * COOP_TN]; // 1 KB
shared float cmD[COOP_TM * COOP_TN];     // 4 KB

coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> cmAcc00;
coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> cmAcc01;
coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> cmAcc10;
coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> cmAcc11;

void coopTileAccClear() {
  cmAcc00 = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
  cmAcc01 = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
  cmAcc10 = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
  cmAcc11 = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
}

// One K chunk: loads the staged panels as fragments and issues the 2x2 multiply-accumulate.
// A loads column-major (element (m, k) sits at cmA[k * COOP_TM + m]), B row-major.
void coopTileStep() {
  coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a0, a1;
  coopMatLoad(a0, cmA, 0u, COOP_TM, gl_CooperativeMatrixLayoutColumnMajor);
  coopMatLoad(a1, cmA, 16u, COOP_TM, gl_CooperativeMatrixLayoutColumnMajor);
  coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b0, b1;
  coopMatLoad(b0, cmB, 0u, COOP_TN, gl_CooperativeMatrixLayoutRowMajor);
  coopMatLoad(b1, cmB, 16u, COOP_TN, gl_CooperativeMatrixLayoutRowMajor);
  cmAcc00 = coopMatMulAdd(a0, b0, cmAcc00);
  cmAcc01 = coopMatMulAdd(a0, b1, cmAcc01);
  cmAcc10 = coopMatMulAdd(a1, b0, cmAcc10);
  cmAcc11 = coopMatMulAdd(a1, b1, cmAcc11);
}

// Lands the fp32 accumulator tile in cmD[m][n] for the caller's bias/act/scatter pass.
void coopTileStoreAcc() {
  coopMatStore(cmAcc00, cmD, 0u, COOP_TN, gl_CooperativeMatrixLayoutRowMajor);
  coopMatStore(cmAcc01, cmD, 16u, COOP_TN, gl_CooperativeMatrixLayoutRowMajor);
  coopMatStore(cmAcc10, cmD, 16u * COOP_TN, COOP_TN, gl_CooperativeMatrixLayoutRowMajor);
  coopMatStore(cmAcc11, cmD, 16u * COOP_TN + 16u, COOP_TN, gl_CooperativeMatrixLayoutRowMajor);
}
