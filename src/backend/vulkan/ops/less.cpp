// Flat (row-major) Less on the GPU: out = (a < b) ? 1 : 0, with
// N-D broadcasting up to flat::kMaxRank. Always runs on the flat path (gpuFlatNode returns true).
// Constant operands are uploaded flat in prepare(), exactly like Equal. Push-constant block
// byte-matches shaders/less.comp.
#include "flat_ops.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct LessVk: VulkanOp {
            struct PC {
                int rank, total;
                int outDim[flat::kMaxRank], aStride[flat::kMaxRank], bStride[flat::kMaxRank];
            } pc {};
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          constBuf[2];

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
                auto setup = [&](TensorId t, int which) {
                    Shape                s = g.desc(t).shape;
                    std::vector<int64_t> ps(rank, 1); // right-align this operand's shape into the output rank (leading dims padded to 1)
                    for (int k = 0; k < (int) s.size(); ++k)
                    {
                        ps[rank - (int) s.size() + k] = s[k];
                    }
                    auto st  = flat::rowStrides(ps);
                    int *dst = (which == 0 ? pc.aStride : pc.bStride);
                    for (int k = 0; k < rank; ++k)
                    {
                        // Broadcast convention shared with flat::Binary: a size-1 dim gets stride 0 so the
                        // shader reads the same element for every output coordinate along that axis.
                        dst[k] = ps[k] == 1 ? 0 : (int) st[k];
                    }
                    if (g.isInitializer(t))
                    {
                        std::vector<float> cv = initFloats(g, t); // decodes fp16; fp32 passthrough
                        cv.resize((size_t) std::max<int64_t>(1, numElements(s))); // 0-D scalar: keep its 1 element
                        constBuf[which] = upload(*env.ctx, cv, env.useFp16);
                    }
                };
                setup(node.inputs[0], 0);
                setup(node.inputs[1], 1);
                pipe = env.pipeline(shader("less", env.useFp16), 3, sizeof(PC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                auto buf = [&](int e) {
                    return constBuf[e] ? constBuf[e].get() : env.devBuf(node.inputs[e]);
                };
                // One flat invocation per output element; less.comp is local_size_x=256 == flat::kFlatLocalSize.
                pipe->dispatch(cmd, {buf(0)->handle(), buf(1)->handle(), env.devBuf(node.outputs[0])->handle()}, &pc, sizeof(pc), groups(pc.total, flat::kFlatLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Less, LessVk);
} // namespace vknn
