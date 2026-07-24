// FusedDwPw (depthwise conv + 1x1 project) shared limits. The fuser (src/import/fuse_dwpw.cpp),
// the Vulkan gate (src/core/vk_gates.cpp), and the GPU op (src/backend/vulkan/ops/fused_dwpw.cpp)
// all key off these constants; the shader-side mirrors are #defines noted per constant, kept in
// lockstep by the comments on both sides.
#pragma once

namespace vknn {

    // Widest expanded (depthwise) channel count E a fused pair may carry. Both fused_dwpw kernels
    // stage the depthwise result for E channels in fixed-size shared arrays; this cap sizes them.
    // Mirrors MAX_EB (= kDwPwMaxExpanded / 4 channel-blocks) in shaders/fused_dwpw_t.comp and
    // shaders/fused_dwpw_t_fp16.comp. 1152 covers the widest inverted-residual expansion in the
    // eligible CNN set (192 * 6).
    inline constexpr int kDwPwMaxExpanded = 1152;

    // Output-tile edge of the spatial-tiled kernel (fused_dwpw_t*): each workgroup produces a
    // kDwPwTile x kDwPwTile pixel tile, staging the depthwise result for all E channels of those
    // pixels in shared memory. Mirrors TILE in shaders/fused_dwpw_t*.comp.
    inline constexpr int kDwPwTile       = 2;
    inline constexpr int kDwPwTilePixels = kDwPwTile * kDwPwTile;

    // Shared bytes the tiled kernels' arrays occupy at the caps: the fp16 kernel stores the
    // depthwise tile as f16vec4 (2 bytes/lane), the fp32 twin as vec4 (4 bytes/lane). Both sit
    // under the ~34KB fused-attention budget the engine already requires of the device.
    inline constexpr int kDwPwTiledSharedFp16Bytes = kDwPwMaxExpanded * kDwPwTilePixels * 2;
    inline constexpr int kDwPwTiledSharedFp32Bytes = kDwPwMaxExpanded * kDwPwTilePixels * 4;

    // Deterministic kernel-choice rule (no autotuning; see fused_dwpw.cpp): the tiled kernel runs
    // when the output plane has at least this many pixels, so its 4x-coarser grid still fills the
    // GPU; a smaller plane keeps the one-workgroup-per-pixel kernel, whose finer grid is the only
    // parallelism a small-spatial block has.
    inline constexpr int kDwPwTileMinPixels = 64;

} // namespace vknn
