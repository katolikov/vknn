// Flat (row-major) boolean OR on the GPU: out = (a != 0 || b != 0) ? 1 : 0, with N-D broadcasting at
// any rank. The prepare/record logic and binding layout are shared with Xor (logical_binary_vk.h); the
// kernel is shaders/or.comp.
#include "logical_binary_vk.h"

namespace vknn {
    namespace {

        struct OrVk: LogicalBinaryVk {
            OrVk(): LogicalBinaryVk("or") {
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Or, OrVk);
} // namespace vknn
