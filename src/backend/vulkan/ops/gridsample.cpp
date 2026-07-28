// GridSample on the GPU: NC4HW4 data (input 0) sampled by a flat [N,Hout,Wout,2] grid (input 1). The grid
// carries normalized COORDINATES, where fp16 quantization costs up to ~0.5 px at 1920-wide inputs, so it
// is kept at fp32 wherever the storage allows: a CONSTANT grid always uploads fp32, and a RUNTIME grid
// (the optical-flow warps) binds its flat activation buffer via operandBuf — fp32 when
// pinGridSampleGridFp32 pinned its storage, else the fp16 bytes it was stored in. The shader reads the
// grid as raw words and decodes per the GRID_FP32 spec constant, so the binding is correct either way.
// The layout pass keeps the grid flat (it can't be NC4HW4-packed with its channels-last [.,.,.,2] shape).
//
// A WARP-mode GridSample (fuseGridSampleWarp) folds the scaled-flow coordinate chain into the op: input 1
// is the NCHW flow [N,2,Hout,Wout] in the NC4HW4 activation layout, input 2 is the base grid
// [.,Hout,Wout,2] uploaded fp32, and the attr warp_scale is the fold's scalar. The gridsample_warp shader
// computes coord = base + scale*flow inside its coordinate lookup — the fp16 variant reproduces the
// standalone Mul's fp16-rounded product (an fp32 base add of an fp16-narrowed flow*scale), so the fused
// op is byte-identical to the materialized-grid path it replaces.
#include "pw_plan.h"
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        // Local workgroup size along x; matches local_size_x in shaders/gridsample.comp.
        constexpr uint32_t kGridSampleLocalSize = 64;

        // Field order/types mirror gridsample.comp's push_constant block { N, C, Hin, Win, OH, OW, align }.
        // align is align_corners (0/1), which shifts the grid-to-pixel coordinate mapping in the shader.
        // scale is the warp fold's multiplier (unused by the plain grid shader), baked fp16-narrowed for
        // the fp16 kernel so flow*scale matches the standalone Mul's operand precision; baseNStride is the
        // warp base grid's N stride (0 broadcasts a [1,OH,OW,2] base over the batch).
        struct GsPC {
            int   N, C, Hin, Win, OH, OW, align;
            float scale;
            int   baseNStride;
        };
        struct GridSampleOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          gridHold; // holds a constant grid (plain) or the fp32 base grid (warp)
            PwEpi                                epi;
            GsPC                                 pc {};
            bool                                 warp  = false;
            int64_t                              total = 0;
            // GRID_FP32 (constant_id 2, plain fp16 shader only): a constant grid always uploads fp32, a
            // runtime grid is fp32 exactly when its storage is pinned (pinGridSampleGridFp32 /
            // --fp32-tensors). Baked from the tensor's FINAL post-pass storage so shader and binding agree.
            static uint32_t gridWordsFp32(const Graph &g, const Node &node) {
                return (g.isInitializer(node.inputs[1]) || g.desc(node.inputs[1]).storeFp32) ? 1u : 0u;
            }
            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g = *env.graph;
                NCHW         x = NCHW::from(g.desc(node.inputs[0]).shape);
                warp           = node.attr.has("warp");
                // Output spatial dims: the channels-last grid's [.,OH,OW,.] for the plain op, the NCHW
                // flow's [.,.,OH,OW] for the warp op.
                const Shape &c1 = g.desc(node.inputs[1]).shape;
                int          OH = warp ? (int) c1[2] : (int) c1[1];
                int          OW = warp ? (int) c1[3] : (int) c1[2];
                pc              = {(int) x.n, (int) x.c, (int) x.h, (int) x.w, OH, OW, (int) node.attr.geti("align_corners", 0), 0.f, 0};
                if (warp)
                {
                    // Bake the scale at the flow's operand precision so flow*scale matches the standalone
                    // Mul: fp16 storage narrows the scalar exactly as the Binary op's uploadInit did
                    // (floatToHalfSat, saturating out-of-range like an imported constant), fp32 keeps it
                    // exact. The shader then rounds the product itself.
                    float s  = node.attr.getf("warp_scale", 1.f);
                    pc.scale = env.useFp16 ? halfToFloat(floatToHalfSat(s)) : s;
                    // Base grid [.,OH,OW,2]: N stride 0 broadcasts a single-frame base over the batch.
                    pc.baseNStride = g.desc(node.inputs[2]).shape[0] == 1 ? 0 : OH * OW * 2;
                }
                // MODE/PAD are baked as spec constants (constant_id 0/1) so the driver specializes the
                // sampler branch at pipeline build. Encoding matches gridsample.comp: MODE 0=bilinear
                // 1=nearest 2=cubic; PAD 0=zeros 1=border 2=reflection.
                std::string mode = node.attr.gets("mode", "bilinear");
                uint32_t    MODE = (mode == "nearest") ? 1u : (mode == "cubic" || mode == "bicubic") ? 2u : 0u;
                std::string pad  = node.attr.gets("padding_mode", "zeros");
                uint32_t    PAD  = pad == "border" ? 1u : (pad == "reflection" ? 2u : 0u);
                // One dispatch lane per output NC4HW4 block-pixel: cBlocks(c) ceil-packs channels into
                // groups of 4, so each lane resolves all 4 packed channels at one (n, OH, OW) location.
                total = (int64_t) x.n * cBlocks(x.c) * OH * OW;
                epi.prepare(node, env, /*flat=*/false, g.desc(node.outputs[0]).shape);
                // The warp shader reads its flow (NC4HW4) and base (fp32) from two dedicated bindings
                // (source, flow, base, dest = 4 base buffers); the plain shader has one grid binding
                // (source, grid, dest = 3). The base grid always uploads fp32, so the warp fp16 shader
                // has no GRID_FP32 selector. epi.suffix() selects the matching _epi shader variant.
                std::vector<uint32_t> spec = warp ? std::vector<uint32_t> {MODE, PAD} : std::vector<uint32_t> {MODE, PAD, gridWordsFp32(g, node)};
                int                   nbuf = warp ? 4 : 3;
                pipe = env.pipeline(shader((std::string(warp ? "gridsample_warp" : "gridsample") + epi.suffix()).c_str(), env.useFp16), nbuf + epi.extraBufs(), sizeof(GsPC), spec);
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                const Graph          *g = env.graph;
                vk::Buffer           *s = env.devBuf(node.inputs[0]);
                vk::Buffer           *d = env.devBuf(node.outputs[0]);
                std::vector<VkBuffer> bufs;
                if (warp)
                {
                    // Warp coordinate source: flow rides its NC4HW4 activation buffer; the base grid is a
                    // constant uploaded fp32 (coordinates ride full precision), read as raw fp32 words.
                    vk::Buffer *flow = env.devBuf(node.inputs[1]);
                    if (!gridHold)
                    {
                        gridHold = uploadWeight(env, initFloats(*g, node.inputs[2]), /*fp16=*/false);
                    }
                    bufs = {s->handle(), flow->handle(), gridHold->handle(), d->handle()};
                } else
                {
                    vk::Buffer *grid;
                    if (g->isInitializer(node.inputs[1]))
                    {
                        // A constant grid uploads fp32 regardless of the segment precision: the grid holds
                        // normalized coordinates whose fp16 quantization drifts the sample point by up to
                        // ~0.5 px at 1920-wide inputs. The shader decodes per GRID_FP32.
                        if (!gridHold)
                        {
                            gridHold = uploadWeight(env, initFloats(*g, node.inputs[1]), /*fp16=*/false);
                        }
                        grid = gridHold.get();
                    } else
                    {
                        grid = env.devBuf(node.inputs[1]); // runtime grid: bound at its storage precision
                    }
                    bufs = {s->handle(), grid->handle(), d->handle()};
                }
                epi.append(bufs, node, env, d->handle());
                // Flat 1D grid over the packed output lanes; kGridSampleLocalSize matches gridsample.comp's local_size_x.
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(total, kGridSampleLocalSize));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::GridSample, GridSampleOp);
} // namespace vknn
