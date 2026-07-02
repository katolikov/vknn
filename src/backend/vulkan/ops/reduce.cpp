// Reduce family on the GPU (FLAT row-major path): ReduceMean/Sum/Max/Min/Prod/L2 over a set of
// axes. One thread per output element loops the reduced axes (see shaders/flat_reduce.comp). The
// layout pass routes Reduce to the flat path; axes come from the `axes` attr or input[1].
#include "flat_ops.h"
#include "pw_plan.h"
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        struct ReducePCFlat {
            int rank, total, op;
            int inDim[flat::kMaxRank];
            int inStride[flat::kMaxRank];
            int reduce[flat::kMaxRank];
        };
        struct ReduceOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            ReducePCFlat                         pc {};
            PwEpi                                epi;
            void                                 prepare(const Node &node, VkOpEnv &env) override {
                const Graph         &g    = *env.graph;
                Shape                in   = g.desc(node.inputs[0]).shape;
                int                  rank = (int) in.size();
                std::vector<int64_t> axes = readI64Param(g, node, "axes", 1);
                if (axes.empty())
                {
                    for (int k = 0; k < rank; ++k)
                    {
                        axes.push_back(k); // reduce all
                    }
                }
                auto inStride = flat::rowStrides(in);
                pc            = {};
                pc.rank       = rank;
                pc.op         = node.subOp; // ReduceType
                for (int k = 0; k < rank; ++k)
                {
                    pc.inDim[k]    = (int) in[k];
                    pc.inStride[k] = (int) inStride[k];
                    pc.reduce[k]   = 0;
                }
                for (int64_t a: axes)
                {
                    int ax = (int) (a < 0 ? a + rank : a);
                    if (ax >= 0 && ax < rank)
                    {
                        pc.reduce[ax] = 1;
                    }
                }
                pc.total = (int) numElements(g.desc(node.outputs[0]).shape);
                epi.prepare(node, env, /*flat=*/true, g.desc(node.outputs[0]).shape);
                pipe = env.pipeline(shader((std::string("flat_reduce") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(ReducePCFlat), std::vector<uint32_t> {});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                VkBuffer              dst  = env.devBuf(node.outputs[0])->handle();
                std::vector<VkBuffer> bufs = {env.devBuf(node.inputs[0])->handle(), dst};
                epi.append(bufs, node, env, dst);
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(pc.total, 256));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Reduce, ReduceOp);
} // namespace vknn
