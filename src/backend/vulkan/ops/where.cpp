// Flat (row-major) Where on the GPU: out = (cond != 0) ? x : y, with N-D broadcasting over all
// three operands at any rank. Always runs on the flat path (gpuFlatNode returns true). Each operand
// may be a constant initializer (uploaded flat in prepare()), like flat::Binary but with three
// inputs. The per-axis geometry (outDim/cStride/xStride/yStride) rides a content-deduped SSBO
// (flat::uploadFlatGeom) bound at binding 4; the push constant carries only rank/total. Layout
// byte-matches shaders/where_select.comp.
#include "flat_ops.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct WhereVk: VulkanOp {
            struct PC {
                int rank, total;
            } pc {};
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          geom;
            std::shared_ptr<vk::Buffer>          constBuf[3];

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g    = *env.graph;
                Shape        out  = g.desc(node.outputs[0]).shape;
                int          rank = (int) out.size();
                pc.rank           = rank;
                pc.total          = (int) numElements(out);
                std::vector<int32_t> outDim(rank), cStride(rank), xStride(rank), yStride(rank);
                for (int k = 0; k < rank; ++k)
                {
                    outDim[k] = (int) out[k];
                }
                int32_t *strideOf[3] = {cStride.data(), xStride.data(), yStride.data()};
                auto     setup       = [&](TensorId t, int which) {
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
                        cv.resize((size_t) std::max<int64_t>(1, numElements(s))); // 0-D scalar: keep its 1 element
                        constBuf[which] = upload(*env.ctx, cv, env.useFp16);
                    }
                };
                setup(node.inputs[0], 0);
                setup(node.inputs[1], 1);
                setup(node.inputs[2], 2);
                geom = flat::uploadFlatGeom(env, {outDim, cStride, xStride, yStride});
                pipe = env.pipeline(shader("where_select", env.useFp16), 5, sizeof(PC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                auto buf = [&](int e) {
                    return constBuf[e] ? constBuf[e].get() : env.devBuf(node.inputs[e]);
                };
                // where_select.comp is local_size_x=256 with one invocation per output element, so the
                // 1D grid is ceil(total/256) — the shared flat element-parallel dispatch size.
                pipe->dispatch(cmd, {buf(0)->handle(), buf(1)->handle(), buf(2)->handle(), env.devBuf(node.outputs[0])->handle(), geom->handle()}, &pc, sizeof(pc), groups(pc.total, flat::kFlatLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Where, WhereVk);
} // namespace vknn
