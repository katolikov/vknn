// Flat ConstantOfShape on the GPU: fill the plan-time output size with the scalar `value` attr.
// Most ConstantOfShape nodes const-fold; one survives when its element count exceeds the fold
// bound, and this kernel keeps it off the CPU. Integer-valued fills (int64 index tensors) stay on
// the exact CPU op, as does an unresolved output size.
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {

        struct ConstantOfShapeVk: VulkanOp {
            struct PC {
                uint32_t count;
                float    value;
            } pc {};
            std::shared_ptr<vk::ComputePipeline> pipe;

            void prepare(const Node &node, VkOpEnv &env) override {
                pc.count = (uint32_t) numElements(env.graph->desc(node.outputs[0]).shape);
                auto it  = node.attr.map.find("value");
                pc.value = (it != node.attr.map.end() && !it->second.floats.empty()) ? it->second.floats[0] : 0.f;
                pipe     = env.pipeline(shader("constant_of_shape", env.useFp16), 1, sizeof(PC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                pipe->dispatch(cmd, {env.devBuf(node.outputs[0])->handle()}, &pc, sizeof(pc), groups(pc.count, 256));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::ConstantOfShape, ConstantOfShapeVk);
} // namespace vknn
