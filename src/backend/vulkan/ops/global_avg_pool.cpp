// GlobalAveragePool: average each channel over H*W. The reduction itself, both dispatch strategies
// and the epilogue live in nc4_spatial_reduce.h, which the Reduce family's spatial arm shares -- a
// mean is that reduction with REDUCE_MEAN.
#include "nc4_spatial_reduce.h"
#include "vknn/reduce_type.h"

namespace vknn {
    namespace {

        struct GlobalAvgPoolOp: VulkanOp {
            Nc4SpatialReduce impl;

            void prepare(const Node &node, VkOpEnv &env) override {
                impl.prepare(node, env, (int) ReduceType::Mean);
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                impl.record(cmd, node, env);
            }
        };

    } // namespace

    VKNN_REGISTER_VK_OP(OpType::GlobalAvgPool, GlobalAvgPoolOp);

} // namespace vknn
