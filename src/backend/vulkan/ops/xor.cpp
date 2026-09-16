// Flat (row-major) boolean exclusive OR on the GPU: out = ((a != 0) != (b != 0)) ? 1 : 0, with N-D
// broadcasting at any rank. The prepare/record logic and binding layout are shared with Or
// (logical_binary_vk.h); the kernel is shaders/xor.comp.
#include "logical_binary_vk.h"

namespace vknn {
    namespace {

        struct XorVk: LogicalBinaryVk {
            XorVk(): LogicalBinaryVk("xor") {
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Xor, XorVk);
} // namespace vknn
