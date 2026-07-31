// Device-scratch footprint of the Resize kernel race, as a shape rule.
//
// The race in src/backend/vulkan/ops/resize.cpp measures the scalar and row-tiled kernels against
// each other on buffers sized like the node's real NC4HW4 tensors, so what it allocates scales with
// the node's geometry and not with anything the race itself bounds. The rule lives in a header so
// the host test suite evaluates the same arithmetic the op does; the buffers it sizes exist only on
// a device.
#pragma once
#include "vknn/nchw.h"
#include <cstddef>
#include <cstdint>

namespace vknn {

    /// Device-only scratch one Resize race may hold at once. The race is a load-time placement
    /// measurement whose result is a pure speed choice - both kernels write identical bytes - and it
    /// allocates while the weights and the activation arena are already resident, which is the
    /// moment a session can least afford a second copy of its largest feature map. An upscaling
    /// Resize sizes two of its three buffers by the OUTPUT map, so the footprint grows with the
    /// square of the scale factor and reaches hundreds of MiB on an ordinary upscale. Past this
    /// budget the pick keeps the deterministic scalar kernel instead of measuring, and stores
    /// nothing, so the same signature races normally wherever the geometry fits.
    constexpr size_t kResizeRaceScratchBudgetBytes = 64u << 20;

    /// Bytes one NC4HW4 tensor of this geometry occupies: one vec4 storage element per
    /// (batch, channel block, pixel), channel-block padding included.
    /// @param batch       N of the tensor.
    /// @param channels    C of the tensor (padded up to whole blocks of kNC4Block).
    /// @param height      Spatial height.
    /// @param width       Spatial width.
    /// @param fp16Storage Whether the tensor is stored as fp16 rather than fp32.
    /// @returns The physical footprint in bytes.
    inline size_t resizeNc4TensorBytes(int64_t batch, int64_t channels, int64_t height, int64_t width, bool fp16Storage) {
        const size_t storageBytes = fp16Storage ? sizeof(uint16_t) : sizeof(float);
        return (size_t) batch * (size_t) cBlocks(channels) * (size_t) height * (size_t) width * (size_t) kNC4Block * storageBytes;
    }

    /// Bytes of device-only scratch the Resize race allocates for one node: a source-sized buffer,
    /// a destination-sized buffer, and (only when a pointwise epilogue is fused, since the epilogue
    /// indexes its operands by output element) a second destination-sized buffer for the operand
    /// filler. Each is the full NC4HW4 physical footprint, channel-block padding included.
    /// @param batch          N of the resized tensor.
    /// @param channels       C of the resized tensor (padded up to whole blocks of kNC4Block).
    /// @param inHeight       Source spatial height.
    /// @param inWidth        Source spatial width.
    /// @param outHeight      Destination spatial height.
    /// @param outWidth       Destination spatial width.
    /// @param epilogueActive Whether a fused pointwise epilogue needs its own operand buffer.
    /// @param fp16Storage    Whether the node's tensors are stored as fp16 rather than fp32.
    /// @returns The total scratch footprint in bytes.
    inline size_t resizeRaceScratchBytes(int64_t batch, int64_t channels, int64_t inHeight, int64_t inWidth, int64_t outHeight, int64_t outWidth, bool epilogueActive, bool fp16Storage) {
        const size_t sourceBytes = resizeNc4TensorBytes(batch, channels, inHeight, inWidth, fp16Storage);
        const size_t destBytes   = resizeNc4TensorBytes(batch, channels, outHeight, outWidth, fp16Storage);
        return sourceBytes + destBytes + (epilogueActive ? destBytes : 0);
    }

    /// Whether the Resize race may allocate its scratch for this geometry.
    /// Arguments are resizeRaceScratchBytes'.
    /// @returns True when the footprint fits kResizeRaceScratchBudgetBytes.
    inline bool resizeRaceScratchFits(int64_t batch, int64_t channels, int64_t inHeight, int64_t inWidth, int64_t outHeight, int64_t outWidth, bool epilogueActive, bool fp16Storage) {
        return resizeRaceScratchBytes(batch, channels, inHeight, inWidth, outHeight, outWidth, epilogueActive, fp16Storage) <= kResizeRaceScratchBudgetBytes;
    }

} // namespace vknn
