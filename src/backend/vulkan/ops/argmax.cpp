// ONNX ArgMax on the GPU: the index of the largest element along `axis`, first index on ties unless
// select_last_index. Kernel, plan and precision contract: arg_extreme_vk.h, arg_extreme_plan.h and
// shaders/arg_extreme.comp.
#include "arg_extreme_vk.h"

namespace vknn {
    namespace {

        struct ArgMaxVk: ArgExtremeVk {
            static constexpr bool kSelectLargest = true;
            ArgMaxVk(): ArgExtremeVk(kSelectLargest) {
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::ArgMax, ArgMaxVk);
} // namespace vknn
