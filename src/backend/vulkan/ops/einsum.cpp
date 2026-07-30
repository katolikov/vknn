// Einsum on the flat GPU path. Only the "i,j->ij" outer product runs on the GPU (e.g. RoPE
// frequency tables); the batched mat-vec / matmul equations fall back to the CPU op. Either operand
// may be a constant initializer (uploaded flat) or an activation.
#include "flat_ops.h"
#include "vk_op_common.h"
#include "vknn/op.h"
#include <string>
#include <vector>

namespace vknn {
    namespace {

        // Mirrors einsum_outer.comp's push_constant block. total is the flat output element count
        // (I*J) that bounds the 1D grid; I and J are the operand lengths the shader uses to decode a
        // flat output index gid into its (row, col) pair as a[gid / J] * b[gid % J].
        struct EinsumPC {
            uint32_t total;
            int      I, J;
        };

        struct EinsumOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            EinsumPC                             pc {};
            std::shared_ptr<vk::Buffer>          constBuf[2];

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g = *env.graph;
                // The outer product treats each operand as a flat vector, so its element count is the
                // full shape product regardless of rank; the I*J product is the output size.
                int64_t I = numElements(g.desc(node.inputs[0]).shape);
                int64_t J = numElements(g.desc(node.inputs[1]).shape);
                pc        = {(uint32_t) (I * J), (int) I, (int) J};
                // Either operand may be a constant initializer (e.g. a RoPE frequency table). Pack it
                // flat into a device buffer here, matching the fp16/fp32 mode; activation operands are
                // instead bound from env.devBuf at record time. resize(n) pins the blob to n elements.
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
                pipe = env.pipeline(shader("einsum_outer", env.useFp16), 3, sizeof(EinsumPC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                // Bind the prepacked constant buffer when the operand was an initializer; otherwise the
                // operand is an activation resolved from its device buffer.
                auto buf = [&](int e) {
                    return constBuf[e] ? constBuf[e].get() : env.devBuf(node.inputs[e]);
                };
                // One flat 1D grid of total output lanes; einsum_outer.comp is local_size_x=256 == flat::kFlatLocalSize.
                pipe->dispatch(cmd, {buf(0)->handle(), buf(1)->handle(), env.devBuf(node.outputs[0])->handle()}, &pc, sizeof(pc), groups(pc.total, flat::kFlatLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Einsum, EinsumOp);
} // namespace vknn
