// Flat (row-major) BitwiseAnd on the GPU: two's-complement bitwise AND of the operands' integer values
// with N-D broadcasting at any rank, on shaders/bitwise.comp with the operator as its specialization
// constant. Geometry, constant operands and precision follow bitwise_vk.h.
#include "bitwise_vk.h"

namespace vknn {
    namespace {

        struct BitwiseAndVk: bitwise_vk::BitwiseBinaryVk {
            BitwiseAndVk(): BitwiseBinaryVk(bitwise_vk::kBitwiseOperatorAnd) {
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::BitwiseAnd, BitwiseAndVk);
} // namespace vknn
