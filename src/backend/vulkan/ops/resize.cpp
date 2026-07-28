// Resize (spatial) on the GPU (NC4HW4): nearest + bilinear, per output channel-block.
#include "pw_plan.h"
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    int vxResizeMode(const std::string &);
    int vxResizeCoord(const std::string &);
    namespace {
        // Local workgroup size along x; matches local_size_x in shaders/resize.comp.
        constexpr uint32_t kResizeLocalSize = 64;

        // Byte-matched to shaders/resize.comp's push_constant block { int N, C, IH, IW, OH, OW, mode, cm }.
        // mode is the resolved vxResizeMode() code (0=nearest, 1=bilinear); cm is the resolved
        // vxResizeCoord() coordinate-transform code the shader's coord() switches on.
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
                // One dispatch lane per output NC4HW4 block-pixel: cBlocks(c) = ceil(c/4) ceil-packs
                // channels into groups of 4, so each lane resolves all 4 packed channels at one
                // (n, cb, oy, ox) location. Spatial extent uses the OUTPUT y.h/y.w (resize target size).
                total = (int64_t) x.n * cBlocks(x.c) * y.h * y.w;
                epi.prepare(node, env, /*flat=*/false, env.graph->desc(node.outputs[0]).shape);
                // Base binding count is 2 (src, dst); a fused pointwise epilogue appends its own operand
                // buffers via extraBufs() and swaps in the _epi shader variant through suffix().
                pipe = env.pipeline(shader((std::string("resize") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(ResizePC), std::vector<uint32_t> {});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer           *s    = env.devBuf(node.inputs[0]);
                vk::Buffer           *d    = env.devBuf(node.outputs[0]);
                std::vector<VkBuffer> bufs = {s->handle(), d->handle()};
                epi.append(bufs, node, env, d->handle());
                // kResizeLocalSize matches resize.comp's local_size_x; groups() ceil-divides the per-block-pixel
                // lane count so the 1D grid covers every lane (the shader's own bounds check drops the tail).
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(total, kResizeLocalSize));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Resize, ResizeOp);
} // namespace vknn
