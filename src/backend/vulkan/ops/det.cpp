// Det on the GPU: determinant of (batched) square matrices, flat row-major, [..., n, n] -> [...].
// ONE dispatch, one thread per matrix; the det_flat kernel loads the n*n elements at storage
// precision, evaluates the FIXED-ORDER cofactor expansion in fp32 (transcribed 1:1 from the CPU
// oracle in backend/cpu/ops/det.cpp — CPU and GPU agree bitwise in fp32 storage), and rounds once
// at the store. vkNodeGate admits only n <= kDetMaxAnalyticN; larger matrices fall back to the
// CPU's general LU with a named reason.
#include "flat_ops.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {
        struct DetPC {
            int batches, n;
        };
        struct DetOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          hold0; // when the input is a constant initializer
            DetPC                                pc {};
            PwEpi                                epi; // fused pointwise unit applied at the store
            void                                 prepare(const Node &node, VkOpEnv &env) override {
                const Shape  &in      = env.graph->desc(node.inputs[0]).shape;
                const int     rank    = (int) in.size();
                const int64_t n       = rank >= 2 ? in[(size_t) rank - 1] : 0;
                int64_t       batches = 1;
                for (int k = 0; k < rank - 2; ++k)
                {
                    batches *= in[(size_t) k];
                }
                pc = {(int) batches, (int) n};
                epi.prepare(node, env, /*flat=*/true, env.graph->desc(node.outputs[0]).shape);
                pipe = env.pipeline(shader((std::string("det_flat") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(DetPC), std::vector<uint32_t> {env.flatLocalSize});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer           *dst  = env.devBuf(node.outputs[0]);
                std::vector<VkBuffer> bufs = {operandBuf(env, node.inputs[0], hold0)->handle(), dst->handle()};
                epi.append(bufs, node, env, dst->handle());
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(pc.batches, env.flatLocalSize));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Det, DetOp);
} // namespace vknn
