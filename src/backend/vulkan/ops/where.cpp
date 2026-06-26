// Flat (row-major) Where on the GPU: out = (cond != 0) ? x : y, with N-D broadcasting over all
// three operands (rank <= 6). Always runs on the flat path (gpuFlatNode returns true). Each operand
// may be a constant initializer (uploaded flat in prepare()), like flat::Binary but with three
// inputs. Push-constant block byte-matches shaders/where_select.comp.
#include "flat_ops.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct WhereVk: VulkanOp {
            struct PC {
                int rank, total;
                int outDim[flat::kMaxRank], cStride[flat::kMaxRank], xStride[flat::kMaxRank], yStride[flat::kMaxRank];
            } pc {};
            std::unique_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          constBuf[3];

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g    = *env.graph;
                Shape        out  = g.desc(node.outputs[0]).shape;
                int          rank = (int) out.size();
                pc.rank           = rank;
                pc.total          = (int) numElements(out);
                for (int k = 0; k < rank; ++k)
                {
                    pc.outDim[k] = (int) out[k];
                }
                int *strideOf[3] = {pc.cStride, pc.xStride, pc.yStride};
                auto setup       = [&](TensorId t, int which) {
                    Shape                s = g.desc(t).shape;
                    std::vector<int64_t> ps(rank, 1); // left-pad to out rank
                    for (int k = 0; k < (int) s.size(); ++k)
                    {
                        ps[rank - (int) s.size() + k] = s[k];
                    }
                    auto st = flat::rowStrides(ps);
                    for (int k = 0; k < rank; ++k)
                    {
                        strideOf[which][k] = ps[k] == 1 ? 0 : (int) st[k];
                    }
                    if (g.isInitializer(t))
                    {
                        std::vector<float> cv = initFloats(g, t); // decodes fp16 (fp16 .vxm); fp32 passthrough
                        cv.resize(numElements(s));
                        constBuf[which] = upload(*env.ctx, cv, env.useFp16);
                    }
                };
                setup(node.inputs[0], 0);
                setup(node.inputs[1], 1);
                setup(node.inputs[2], 2);
                pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("where_select", env.useFp16), 4, sizeof(PC), std::vector<uint32_t> {}, env.cache->handle());
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                auto buf = [&](int e) {
                    return constBuf[e] ? constBuf[e].get() : env.devBuf(node.inputs[e]);
                };
                pipe->dispatch(cmd, {buf(0)->handle(), buf(1)->handle(), buf(2)->handle(), env.devBuf(node.outputs[0])->handle()}, &pc, sizeof(pc), groups(pc.total, 256));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::kWhere, WhereVk);
} // namespace vknn
