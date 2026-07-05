// Flat ConstantOfShape on the GPU: fill the plan-time output size with the scalar `value` attr.
// Most ConstantOfShape nodes const-fold; one survives when its element count exceeds the fold
// bound, and this kernel keeps it off the CPU. Integer-valued fills (int64 index tensors) stay on
// the exact CPU op, as does an unresolved output size.
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {

        // 1D workgroup width; matches `layout(local_size_x = 256)` in shaders/constant_of_shape.comp.
        // The dispatch group count is derived from this exact value so the launched thread grid covers
        // every output element with no gap or overshoot.
        constexpr uint32_t kConstantOfShapeLocalSize = 256;

        struct ConstantOfShapeVk: VulkanOp {
            // Mirrors constant_of_shape.comp's push_constant block. `count` is the flat (logical)
            // output element count; `value` stays fp32 even under fp16 storage because the shader casts
            // it once via STORE(pc.value) at write time.
            struct PC {
                uint32_t count;
                float    value;
            } pc {};
            std::shared_ptr<vk::ComputePipeline> pipe;

            void prepare(const Node &node, VkOpEnv &env) override {
                // The shader writes a plain linear buffer (d[i]), so the count is the logical element
                // product, not an NC4HW4 packed count. Missing/empty `value` attr defaults to 0.
                pc.count = (uint32_t) numElements(env.graph->desc(node.outputs[0]).shape);
                auto it  = node.attr.map.find("value");
                pc.value = (it != node.attr.map.end() && !it->second.floats.empty()) ? it->second.floats[0] : 0.f;
                pipe     = env.pipeline(shader("constant_of_shape", env.useFp16), 1, sizeof(PC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                pipe->dispatch(cmd, {env.devBuf(node.outputs[0])->handle()}, &pc, sizeof(pc), groups(pc.count, kConstantOfShapeLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::ConstantOfShape, ConstantOfShapeVk);
} // namespace vknn
