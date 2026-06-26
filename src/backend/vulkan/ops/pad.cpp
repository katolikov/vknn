// Pad on the GPU — flat row-major (constant/edge/reflect). The layout pass marks Pad's I/O flat and
// inserts ConvertLayout at the NC4HW4 boundary; static pads come from attr "pads" or a constant input.
#include "flat_ops.h"

namespace vknn {
    namespace {

        struct PadOp: VulkanOp {
            flat::Pad impl;
            void      prepare(const Node &node, VkOpEnv &env) override {
                impl.prepare(node, env);
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                impl.record(cmd, node, env);
            }
        };

    } // namespace

    VKNN_REGISTER_VK_OP(OpType::kPad, PadOp);

} // namespace vknn
