// MaxPool2D on the GPU (NC4HW4).
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct MaxPoolOp: VulkanOp {
            std::unique_ptr<vk::ComputePipeline> pipe;
            MaxPC                                pc {};
            int64_t                              total = 0;

            void prepare(const Node &node, VkOpEnv &env) override {
                NCHW x    = NCHW::from(env.graph->desc(node.inputs[0]).shape);
                NCHW y    = NCHW::from(env.graph->desc(node.outputs[0]).shape);
                auto ints = [&](const char *k, std::vector<int64_t> d) {
                    const auto &v = node.attr.getints(k);
                    return v.empty() ? d : v;
                };
                auto ks  = ints("kernel_shape", {1, 1});
                auto st  = ints("strides", {1, 1});
                auto pad = ints("pads", {0, 0, 0, 0});
                pc       = {(int) x.n,   (int) x.c,   (int) x.h,   (int) x.w,   (int) y.h,    (int) y.w,
                            (int) ks[0], (int) ks[1], (int) st[0], (int) st[1], (int) pad[0], (int) pad[1]};
                total    = x.n * cBlocks(x.c) * y.h * y.w;
                pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("maxpool", env.useFp16), 2, sizeof(MaxPC), std::vector<uint32_t> {}, env.cache->handle());
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *src = env.devBuf(node.inputs[0]);
                vk::Buffer *dst = env.devBuf(node.outputs[0]);
                pipe->dispatch(cmd, {src->handle(), dst->handle()}, &pc, sizeof(pc), groups(total, 64));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::MaxPool, MaxPoolOp);
} // namespace vknn
