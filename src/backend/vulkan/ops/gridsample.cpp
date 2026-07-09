// GridSample on the GPU: NC4HW4 data (input 0) sampled by a flat [N,Hout,Wout,2] grid (input 1). The grid
// carries normalized COORDINATES, where fp16 quantization costs up to ~0.5 px at 1920-wide inputs, so it
// is kept at fp32 wherever the storage allows: a CONSTANT grid always uploads fp32, and a RUNTIME grid
// (the optical-flow warps) binds its flat activation buffer via operandBuf — fp32 when
// pinGridSampleGridFp32 pinned its storage, else the fp16 bytes it was stored in. The shader reads the
// grid as raw words and decodes per the GRID_FP32 spec constant, so the binding is correct either way.
// The layout pass keeps the grid flat (it can't be NC4HW4-packed with its channels-last [.,.,.,2] shape).
#include "pw_plan.h"
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        // Local workgroup size along x; matches local_size_x in shaders/gridsample.comp.
        constexpr uint32_t kGridSampleLocalSize = 64;

        // Field order/types mirror gridsample.comp's push_constant block { N, C, Hin, Win, OH, OW, align }.
        // align is align_corners (0/1), which shifts the grid-to-pixel coordinate mapping in the shader.
        struct GsPC {
            int N, C, Hin, Win, OH, OW, align;
        };
        struct GridSampleOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          gridHold; // holds a constant grid; a runtime grid uses devBuf
            PwEpi                                epi;
            GsPC                                 pc {};
            int64_t                              total = 0;
            void                                 prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g  = *env.graph;
                NCHW         x  = NCHW::from(g.desc(node.inputs[0]).shape);
                const Shape &gs = g.desc(node.inputs[1]).shape; // [N,Hout,Wout,2]
                int          OH = (int) gs[1], OW = (int) gs[2];
                pc               = {(int) x.n, (int) x.c, (int) x.h, (int) x.w, OH, OW, (int) node.attr.geti("align_corners", 0)};
                // MODE/PAD are baked as spec constants (constant_id 0/1) so the driver specializes the
                // sampler branch at pipeline build. Encoding matches gridsample.comp: MODE 0=bilinear
                // 1=nearest 2=cubic; PAD 0=zeros 1=border 2=reflection.
                std::string mode = node.attr.gets("mode", "bilinear");
                uint32_t    MODE = (mode == "nearest") ? 1u : (mode == "cubic" || mode == "bicubic") ? 2u : 0u;
                std::string pad  = node.attr.gets("padding_mode", "zeros");
                uint32_t    PAD  = pad == "border" ? 1u : (pad == "reflection" ? 2u : 0u);
                // GRID_FP32 (constant_id 2, fp16 shader only) tells the shader how to decode the raw
                // grid words: a constant grid always uploads fp32 (record() below), a runtime grid is
                // fp32 exactly when its storage is pinned (pinGridSampleGridFp32 / --fp32-tensors).
                // Baked from the tensor's FINAL post-pass storage, so shader and binding always agree.
                uint32_t GRID_FP32 = (g.isInitializer(node.inputs[1]) || g.desc(node.inputs[1]).storeFp32) ? 1u : 0u;
                // One dispatch lane per output NC4HW4 block-pixel: cBlocks(c) ceil-packs channels into
                // groups of 4, so each lane resolves all 4 packed channels at one (n, OH, OW) location.
                total            = (int64_t) x.n * cBlocks(x.c) * OH * OW;
                epi.prepare(node, env, /*flat=*/false, g.desc(node.outputs[0]).shape);
                // Binding count is the 3 base buffers (source, grid, dest) plus any extra buffers the
                // fused pointwise epilogue reads; epi.suffix() selects the matching _epi shader variant.
                pipe = env.pipeline(shader((std::string("gridsample") + epi.suffix()).c_str(), env.useFp16), 3 + epi.extraBufs(), sizeof(GsPC), std::vector<uint32_t> {MODE, PAD, GRID_FP32});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                const Graph &g = *env.graph;
                vk::Buffer  *s = env.devBuf(node.inputs[0]);
                vk::Buffer  *grid;
                if (g.isInitializer(node.inputs[1]))
                {
                    // A constant grid uploads fp32 regardless of the segment precision: the grid holds
                    // normalized coordinates whose fp16 quantization drifts the sample point by up to
                    // ~0.5 px at 1920-wide inputs. The shader decodes per GRID_FP32.
                    if (!gridHold)
                    {
                        gridHold = uploadWeight(env, initFloats(g, node.inputs[1]), /*fp16=*/false);
                    }
                    grid = gridHold.get();
                } else
                {
                    grid = env.devBuf(node.inputs[1]); // runtime grid: bound at its storage precision
                }
                vk::Buffer *d = env.devBuf(node.outputs[0]);
                std::vector<VkBuffer> bufs = {s->handle(), grid->handle(), d->handle()};
                epi.append(bufs, node, env, d->handle());
                // Flat 1D grid over the packed output lanes; kGridSampleLocalSize matches gridsample.comp's local_size_x.
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(total, kGridSampleLocalSize));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::GridSample, GridSampleOp);
} // namespace vknn
