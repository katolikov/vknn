// GridSample on the GPU: NC4HW4 data (input 0) sampled by a flat [N,Hout,Wout,2] grid (input 1). The grid
// is bound at the segment compute precision — a CONSTANT grid is uploaded (fp16/fp32) and a RUNTIME grid
// (the optical-flow warps) is bound directly from its flat activation buffer via operandBuf. The layout
// pass keeps the grid flat (it can't be NC4HW4-packed with its channels-last [.,.,.,2] shape).
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        struct GsPC {
            int N, C, Hin, Win, OH, OW, align;
        };
        struct GridSampleOp: VulkanOp {
            std::unique_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          gridHold; // holds a constant grid; a runtime grid uses devBuf
            GsPC                                 pc {};
            int64_t                              total = 0;
            void                                 prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g  = *env.graph;
                NCHW         x  = NCHW::from(g.desc(node.inputs[0]).shape);
                const Shape &gs = g.desc(node.inputs[1]).shape; // [N,Hout,Wout,2]
                int          OH = (int) gs[1], OW = (int) gs[2];
                pc               = {(int) x.n, (int) x.c, (int) x.h, (int) x.w, OH, OW, (int) node.attr.geti("align_corners", 0)};
                std::string mode = node.attr.gets("mode", "bilinear");
                uint32_t    MODE = (mode == "nearest") ? 1u : 0u;
                std::string pad  = node.attr.gets("padding_mode", "zeros");
                uint32_t    PAD  = pad == "border" ? 1u : (pad == "reflection" ? 2u : 0u);
                total            = (int64_t) x.n * cBlocks(x.c) * OH * OW;
                pipe             = std::make_unique<vk::ComputePipeline>(*env.ctx, shader("gridsample", env.useFp16), 3, sizeof(GsPC), std::vector<uint32_t> {MODE, PAD},
                                                                         env.cache->handle());
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *s    = env.devBuf(node.inputs[0]);
                vk::Buffer *grid = operandBuf(env, node.inputs[1], gridHold); // const upload or runtime buffer
                vk::Buffer *d    = env.devBuf(node.outputs[0]);
                pipe->dispatch(cmd, {s->handle(), grid->handle(), d->handle()}, &pc, sizeof(pc), groups(total, 64));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::GridSample, GridSampleOp);
} // namespace vknn
