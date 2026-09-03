// The input-affine prologue a Conv can carry: a pointwise unit (per-channel scale, per-channel
// shift, activation, in that order) applied to every input element the conv loads, in place of a
// standalone FusedPointwise node that would write the transformed tensor for the conv to read
// back. fusePointwiseChains attaches it when the unit's producer cannot host it (a zero-copy Concat
// view, a graph input); the Vulkan Conv op applies it inline in the kernel families that carry a
// _pro variant (shaders/input_affine.glsl) and through a standalone input_affine dispatch into a
// scratch tensor otherwise; the CPU Conv op applies it to its input before the convolution.
//
// Node attributes (all present together, or none):
//   pro_opbase  first index in node.inputs of the prologue's operand tensors
//   pro_scale   index in node.inputs of the [N,C,1,1] scale tensor, or -1
//   pro_shift   index in node.inputs of the [N,C,1,1] shift tensor, or -1
//   pro_act     ActType wire code applied last (None when the unit has no activation)
//   pro_act_lo / pro_act_hi   the activation's bounds (Clip / Relu6)
// The unit computes in fp32 with no intermediate rounding and the conv consumes the value directly;
// the pass therefore attaches it only in its relaxed mode (a strictly-rounded graph keeps the
// standalone unit).
#pragma once
#include "vknn/node.h"
#include <cstdint>

namespace vknn {

    /// Flag bits of the prologue as the kernels receive them (ConvPC::proFlags; mirrored by the
    /// PRO_* macros in shaders/input_affine.glsl).
    constexpr int kInputAffineHasScale = 1;
    constexpr int kInputAffineHasShift = 2;
    constexpr int kInputAffineBatched  = 8; ///< the scale / shift tensors carry one channel row per batch

    /// Kernel taps (KH * KW) a conv needs before it hosts a prologue. The prologue runs once per
    /// loaded input vec4 and a conv kernel reuses each load over KH*KW*4*OCB multiplies: at nine
    /// taps the row and register-tiled kernels absorb the extra fma and activation (DenseNet-121's
    /// 3x3 convs 0.131 -> 0.122 ms with their standalone units gone), at one tap a pointwise kernel
    /// recomputes it once per output-block group and ran 1.8-2.2x slower than the unit it replaced
    /// (the transition 1x1 convs 0.166 -> 0.360 ms), so a 1x1 consumer keeps the standalone unit.
    constexpr int64_t kInputAffineMinTaps = 9;

    inline bool inputAffineActive(const Node &n) {
        return n.attr.has("pro_opbase");
    }

} // namespace vknn
