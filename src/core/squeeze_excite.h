// The fused Squeeze-Excite op (OpType::FusedSE): GlobalAvgPool -> Conv1x1 -> activation -> Conv1x1
// -> gate collapsed into one node that emits the channel scale [N,C,1,1] from the feature map
// [N,C,H,W]; the broadcast Mul that applies the scale stays a separate op. The node carries the
// FC1 activation in `fusedAct` (Relu / SiLU / HardSwish), the gate in `subOp` as a UnaryType
// (HardSigmoid / Sigmoid) and the HardSigmoid alpha/beta in `actLo` / `actHi`.
//
// The GPU kernel (shaders/fused_se*.comp) keeps the pooled vector and the squeeze activations in
// workgroup-shared memory, so both widths are bounded at compile time; the import pass leaves a
// wider chain unfused. The codes below are what the kernel switches on: SE_ACT_* / SE_GATE_* in
// the shaders mirror them, SE_MAX_CHANNELS / SE_MAX_SQUEEZE mirror the bounds (pinned by
// tools/check_shader_contracts.py).
#pragma once
#include "vknn/act_type.h"
#include "vknn/unary_type.h"
#include <cstdint>

namespace vknn {
    /// Widest channel count the fused kernel pools and gates (shared fp32 avg[] entries, 16 KB).
    constexpr int64_t kSeMaxChannels = 4096;
    /// Widest squeeze (FC1 output) width (shared fp32 s1[] entries, 4 KB).
    constexpr int64_t kSeMaxSqueeze = 1024;

    /// FC1 activation codes (push constant `act`).
    constexpr int kSeActRelu      = 0;
    constexpr int kSeActSiLU      = 1;
    constexpr int kSeActHardSwish = 2;
    /// Gate codes (push constant `gate`).
    constexpr int kSeGateHardSigmoid = 0;
    constexpr int kSeGateSigmoid     = 1;
    /// The activation or gate is not one the fused op implements.
    constexpr int kSeCodeNone = -1;

    inline int seActCode(ActType act) noexcept {
        switch (act)
        {
            case ActType::Relu:
                return kSeActRelu;
            case ActType::SiLU:
                return kSeActSiLU;
            case ActType::HardSwish:
                return kSeActHardSwish;
            default:
                return kSeCodeNone;
        }
    }

    inline int seGateCode(UnaryType gate) noexcept {
        switch (gate)
        {
            case UnaryType::HardSigmoid:
                return kSeGateHardSigmoid;
            case UnaryType::Sigmoid:
                return kSeGateSigmoid;
            default:
                return kSeCodeNone;
        }
    }

    /// Whether a chain of `channels` feature channels and `squeeze` FC1 outputs fits the kernel.
    inline bool seShapeFits(int64_t channels, int64_t squeeze) noexcept {
        return channels >= 1 && channels <= kSeMaxChannels && squeeze >= 1 && squeeze <= kSeMaxSqueeze;
    }
} // namespace vknn
