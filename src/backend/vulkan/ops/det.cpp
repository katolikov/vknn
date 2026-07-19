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
            void                                 prepare(const Node &node, VkOpEnv &env) override {
                const Shape &in   = env.graph->desc(node.inputs[0]).shape;
                const int    rank = (int) in.size();
                const int64_t n   = rank >= 2 ? in[(size_t) rank - 1] : 0;
                int64_t       batches = 1;
                for (int k = 0; k < rank - 2; ++k)
                {
                    batches *= in[(size_t) k];
                }
                pc   = {(int) batches, (int) n};
                pipe = env.pipeline(shader("det_flat", env.useFp16), 2, sizeof(DetPC), std::vector<uint32_t> {});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                pipe->dispatch(cmd, {operandBuf(env, node.inputs[0], hold0)->handle(), env.devBuf(node.outputs[0])->handle()}, &pc, sizeof(pc), groups(pc.batches, flat::kFlatLocalSize));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Det, DetOp);
} // namespace vknn
