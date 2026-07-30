// Resize (spatial) on the GPU (NC4HW4): nearest + bilinear + cubic, per output channel-block.
#include "core/resize_rule.h"
#include "pw_plan.h"
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        // Local workgroup size along x; matches local_size_x in shaders/resize.comp.
        constexpr uint32_t kResizeLocalSize = 64;

        // Byte-matched to shaders/resize.comp's push_constant block
        // { int N, C, IH, IW, OH, OW, mode, cm; float cubicA; int excludeOutside, perPixel, nm }.
        // mode is the resolved vxResizeMode() code, cm the vxResizeCoord() code the shader's coord()
        // switches on, nm the vxResizeNearestMode() rounding nearestSrc applies; cubicA and
        // excludeOutside carry the ONNX cubic attributes and are read only in cubic mode.
        struct ResizePC {
            int   N, C, IH, IW, OH, OW, mode, cm;
            float cubicA;
            int   excludeOutside, perPixel, nm;
        };

        // Lane map, as a deterministic shape rule. The two maps are byte-identical -- same arithmetic
        // per (block, pixel), only the thread that runs it moves -- so this is a pure placement
        // choice and none == fast == heavy still holds.
        //
        // One lane per output PIXEL evaluates the coordinate transform and the cubic weight fans once
        // instead of once per channel block, which is what a GROWING output map wants. A SHRINKING
        // one wants the opposite: it reads more than it writes, and the block-pixel map keeps a whole
        // wave inside one channel block where the source streams. Measured per output/input pixel
        // ratio at Cb=8, cubic, both directions: growing or equal is -20% to -41% on the per-pixel
        // map, shrinking is +11% to +23% on it. The rule is the sign of that ratio.
        inline bool resizePerPixelLanes(const NCHW &x, const NCHW &y) {
            return (int64_t) y.h * y.w >= (int64_t) x.h * x.w;
        }
        struct ResizeOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            ResizePC                             pc {};
            PwEpi                                epi;
            int64_t                              total = 0;
            void                                 prepare(const Node &node, VkOpEnv &env) override {
                NCHW x = NCHW::from(env.graph->desc(node.inputs[0]).shape);
                NCHW y = NCHW::from(env.graph->desc(node.outputs[0]).shape);
                pc     = {(int) x.n,
                                                          (int) x.c,
                                                          (int) x.h,
                                                          (int) x.w,
                                                          (int) y.h,
                                                          (int) y.w,
                                                          vxResizeMode(node.attr.gets("mode", "nearest")),
                                                          vxResizeCoord(node.attr.gets("coordinate_transformation_mode", "half_pixel")),
                                                          node.attr.getf("cubic_coeff_a", kResizeCubicCoeffDefault),
                                                          (int) node.attr.geti("exclude_outside", 0),
                      resizePerPixelLanes(x, y) ? 1 : 0,
                                                          vxResizeNearestMode(node.attr.gets("nearest_mode", "round_prefer_floor"))};
                // Lane count for the map resizePerPixelLanes chose: one per output pixel (the kernel
                // loops the channel blocks internally) or one per NC4HW4 block-pixel. Spatial extent
                // uses the OUTPUT y.h/y.w (the resize target size).
                total = (int64_t) x.n * y.h * y.w * (pc.perPixel != 0 ? 1 : cBlocks(x.c));
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
