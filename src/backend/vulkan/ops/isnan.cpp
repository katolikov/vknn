// Flat (row-major) IsNaN on the GPU: out = isnan(x) ? 1 : 0, a pure 1:1 elementwise map. Always
// runs on the flat path (gpuFlatNode returns true). No broadcasting and no axis, so it needs no
// geometry SSBO: the push constant carries only the element count and the shader is a straight walk
// with a tail bounds check. The bool result is the canonical fp32 1.0/0.0 the flat comparison ops
// emit. Layout byte-matches shaders/isnan.comp.
#include "flat_ops.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct IsNaNVk: VulkanOp {
            struct PC {
                int total;
            } pc {};
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          hold; // when the input is a constant initializer

            void prepare(const Node &node, VkOpEnv &env) override {
                pc.total = (int) numElements(env.graph->desc(node.outputs[0]).shape);
                // 2 SSBOs: source (binding 0) and destination (binding 1).
                pipe = env.pipeline(shader("isnan", env.useFp16), 2, sizeof(PC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *s = operandBuf(env, node.inputs[0], hold);
                vk::Buffer *d = env.devBuf(node.outputs[0]);
                // One flat invocation per element; isnan.comp is local_size_x=256 == flat::kFlatLocalSize.
                pipe->dispatch(cmd, {s->handle(), d->handle()}, &pc, sizeof(pc), groups(pc.total, flat::kFlatLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::IsNaN, IsNaNVk);
} // namespace vknn
