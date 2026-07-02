// Resize (spatial) on the GPU (NC4HW4): nearest + bilinear, per output channel-block.
#include "pw_plan.h"
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    int vxResizeMode(const std::string &);
    int vxResizeCoord(const std::string &);
    namespace {
        struct ResizePC {
            int N, C, IH, IW, OH, OW, mode, cm;
        };
        struct ResizeOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            ResizePC                             pc {};
            PwEpi                                epi;
            int64_t                              total = 0;
            void                                 prepare(const Node &node, VkOpEnv &env) override {
                NCHW x = NCHW::from(env.graph->desc(node.inputs[0]).shape);
                NCHW y = NCHW::from(env.graph->desc(node.outputs[0]).shape);
                pc = {(int) x.n, (int) x.c, (int) x.h, (int) x.w, (int) y.h, (int) y.w, vxResizeMode(node.attr.gets("mode", "nearest")), vxResizeCoord(node.attr.gets("coordinate_transformation_mode", "half_pixel"))};
                total = (int64_t) x.n * cBlocks(x.c) * y.h * y.w;
                epi.prepare(node, env, /*flat=*/false, env.graph->desc(node.outputs[0]).shape);
                pipe  = env.pipeline(shader((std::string("resize") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(ResizePC), std::vector<uint32_t> {});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer           *s    = env.devBuf(node.inputs[0]);
                vk::Buffer           *d    = env.devBuf(node.outputs[0]);
                std::vector<VkBuffer> bufs = {s->handle(), d->handle()};
                epi.append(bufs, node, env, d->handle());
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(total, 64));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Resize, ResizeOp);
} // namespace vknn
