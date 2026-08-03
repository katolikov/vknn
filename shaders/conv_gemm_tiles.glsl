// Shared tile geometry of the implicit-GEMM convolution family: conv_gemm.comp,
// conv_gemm_ksplit.comp, and conv_gemm_wv4.comp include this file so the panel shapes their
// common LDS layout and TK-chunked K stepping assume have one definition — a change applied to
// only one of them is the silent-wrong-answer class the shared GridSample machinery
// (gridsample_taps.glsl) guards against. TM itself stays a per-kernel specialization constant
// (constant_id 0, values 16/32/64 from convGemmTileM).
// Host mirrors: kConvGemmTileN / kConvGemmTileK and convGemmTileM (src/backend/vulkan/ops/
// vk_op_common.h); conv.cpp's routing floor counts these tiles (Cout >= kGemmMinCoutTiles *
// kConvGemmTileN = 8*64 on the N axis).
#define TM_MAX 64      // shared storage is sized for the widest variant (4 KB total; no occupancy cost)
#define TN 64
#define TK 16
#define TILE 16        // threads per dim (16x16 = 256)
#define RM (TM / TILE) // output pixels per thread: 1, 2, or 4
#define RN (TN / TILE) // 4 output channels per thread = one NC4 block
