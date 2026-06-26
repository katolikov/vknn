// Einsum on the flat GPU path. Only the "i,j->ij" outer product runs on the GPU (the 102 RoPE
// frequency tables); the batched mat-vec / matmul equations fall back to the CPU op (a handful of
// tiny tensors in the geometry tail). Either operand may be a constant initializer (uploaded flat)
// or an activation.
#include "flat_ops.h"
#include "vk_op_common.h"
#include "vknn/op.h"
#include <string>
#include <vector>

namespace vknn {
    namespace {

        struct EinsumPC {
            uint32_t total;
            int      I, J;
        };

        struct EinsumOp: VulkanOp {
            std::unique_ptr<vk::ComputePipeline> pipe;
            EinsumPC                             pc {};
            std::shared_ptr<vk::Buffer>          constBuf[2];

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g = *env.graph;
                int64_t      I = numElements(g.desc(node.inputs[0]).shape);
                int64_t      J = numElements(g.desc(node.inputs[1]).shape);
                pc             = {(uint32_t) (I * J), (int) I, (int) J};
                for (int e = 0; e < 2; ++e)
                {
                    TensorId t = node.inputs[e];
                    if (g.isInitializer(t))
                    {
                        int64_t            n = numElements(g.desc(t).shape);
                        std::vector<float> v = initFloats(g, t);
                        v.resize(n);
                        constBuf[e] = upload(*env.ctx, v, env.useFp16);
                    }
                }
                pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("einsum_outer", env.useFp16), 3, sizeof(EinsumPC), std::vector<uint32_t> {},
                                                             env.cache->handle());
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                auto buf = [&](int e) {
                    return constBuf[e] ? constBuf[e].get() : env.devBuf(node.inputs[e]);
                };
                pipe->dispatch(cmd, {buf(0)->handle(), buf(1)->handle(), env.devBuf(node.outputs[0])->handle()}, &pc, sizeof(pc), groups(pc.total, 256));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::kEinsum, EinsumOp);
} // namespace vknn
