// Reduce family on the GPU (FLAT row-major path): ReduceMean/Sum/Max/Min/Prod/L2 over a set of
// axes. One thread per output element loops the reduced axes (see shaders/flat_reduce.comp). The
// layout pass routes Reduce to the flat path; axes come from the `axes` attr or input[1].
#include "flat_ops.h"
#include "pw_plan.h"
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        // Field order/types mirror flat_reduce.comp's push_constant block. `op` is the ReduceType
        // sub-op code (Mean=0, Sum=1, Max=2, Min=3, Prod=4, L2=5) the shader branches on; `reduce`
        // is a 0/1 mask marking which axes collapse. inDim/inStride are the input's row-major shape
        // and strides so the shader can walk both the kept axes (to place the output) and the reduced
        // axes (to accumulate).
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
                // Axes come from the `axes` attr or input[1] (readI64Param checks both); an empty list
                // is the ONNX "reduce over every axis" default, giving a scalar output.
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
                // Build the reduce mask: normalize each ONNX axis (negative counts from the end) and
                // set its lane. Out-of-range axes are ignored so a malformed attr can't index past rank.
                for (int64_t a: axes)
                {
                    int ax = (int) (a < 0 ? a + rank : a);
                    if (ax >= 0 && ax < rank)
                    {
                        pc.reduce[ax] = 1;
                    }
                }
                // One shader invocation per output element, so the 1D grid is sized to the output count.
                pc.total = (int) numElements(g.desc(node.outputs[0]).shape);
                // A pointwise chain can be folded into the reduce's store: epi.suffix() selects the
                // _epi shader variant and epi.extraBufs() adds its operand bindings on top of the two
                // base buffers (input, output).
                epi.prepare(node, env, /*flat=*/true, g.desc(node.outputs[0]).shape);
                pipe = env.pipeline(shader((std::string("flat_reduce") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(ReducePCFlat), std::vector<uint32_t> {});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                // Binding order matches the shader: input at 0, output at 1, then any fused-epilogue
                // operands appended by epi.append (must stay in sync with the extraBufs() count above).
                VkBuffer              dst  = env.devBuf(node.outputs[0])->handle();
                std::vector<VkBuffer> bufs = {env.devBuf(node.inputs[0])->handle(), dst};
                epi.append(bufs, node, env, dst);
                // One flat 1D grid over the output: ceil(total / 256) workgroups of local_size_x=256.
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(pc.total, 256));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Reduce, ReduceOp);
} // namespace vknn
