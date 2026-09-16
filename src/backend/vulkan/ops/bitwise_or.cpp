// Flat (row-major) BitwiseOr on the GPU: two's-complement bitwise OR of the operands' integer values
// with N-D broadcasting at any rank, on shaders/bitwise.comp with the operator as its specialization
// constant. Geometry, constant operands and precision follow bitwise_vk.h.
#include "bitwise_vk.h"

namespace vknn {
    namespace {

        struct BitwiseOrVk: bitwise_vk::BitwiseBinaryVk {
            BitwiseOrVk(): BitwiseBinaryVk(bitwise_vk::kBitwiseOperatorOr) {
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::BitwiseOr, BitwiseOrVk);
} // namespace vknn
