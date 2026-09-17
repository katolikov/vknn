// Flat (row-major) BitwiseXor on the GPU: two's-complement bitwise XOR of the operands' integer values
// with N-D broadcasting at any rank, on shaders/bitwise.comp with the operator as its specialization
// constant. Geometry, constant operands and precision follow bitwise_vk.h.
#include "bitwise_vk.h"

namespace vknn {
    namespace {

        struct BitwiseXorVk: bitwise_vk::BitwiseBinaryVk {
            BitwiseXorVk(): BitwiseBinaryVk(bitwise_vk::kBitwiseOperatorXor) {
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::BitwiseXor, BitwiseXorVk);
} // namespace vknn
