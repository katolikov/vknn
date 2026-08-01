// GridSample on the GPU: NC4HW4 data (input 0) sampled by a flat [N,Hout,Wout,2] grid (input 1). The grid
// carries normalized COORDINATES, where fp16 quantization costs up to ~0.5 px at 1920-wide inputs, so it
// is kept at fp32 wherever the storage allows: a CONSTANT grid always uploads fp32, and a RUNTIME grid
// (the optical-flow warps) binds its flat activation buffer via operandBuf — fp32 when
// pinSampleCoordFp32 pinned its storage, else the fp16 bytes it was stored in. The shader reads the
// grid as raw words and decodes per the GRID_FP32 spec constant, so the binding is correct either way.
// The layout pass keeps the grid flat (it can't be NC4HW4-packed with its channels-last [.,.,.,2] shape).
//
// A WARP-mode GridSample (fuseGridSampleWarp) folds the scaled-flow coordinate chain into the op: input 1
// is the NCHW flow [N,2,Hout,Wout] in the NC4HW4 activation layout, input 2 is the base grid
// [.,Hout,Wout,2] uploaded fp32, and the attr warp_scale is the fold's scalar. The gridsample_warp shader
// computes coord = base + scale*flow inside its coordinate lookup, decoding the flow per FLOW_FP32 and
// multiplying it by a scalar baked at the SAME precision (gridsample_rule.h): an fp16 flow reproduces the
// standalone Mul's fp16-rounded product, so the fused op is byte-identical to the materialized-grid path
// it replaces, and a flow pinned fp32 by pinSampleCoordFp32 carries full-precision coordinates end to end.
#include "gridsample_rule.h"
#include "pw_plan.h"
#include "vk_op_common.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        // The lane width rides the shaders' trailing spec constant, resolved at load from exact
        // caps (env.convLocalSize - the per-thread sampler family width); the dispatch divides by
        // the same value.

        // Field order/types mirror gridsample.comp's push_constant block { N, C, Hin, Win, OH, OW, align }.
        // align is align_corners (0/1), which shifts the grid-to-pixel coordinate mapping in the shader.
        // scale is the warp fold's multiplier (unused by the plain grid shader), baked at the flow
        // operand's storage precision; baseNStride is the warp base grid's N stride (0 broadcasts a
        // [1,OH,OW,2] base over the batch).
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
            // Decode precision of the coordinate operand at input 1, baked from the tensor's FINAL
            // post-pass storage so shader and binding agree: GRID_FP32 for the plain kernel's grid,
            // FLOW_FP32 for the warp kernel's flow (gridsample_rule.h holds both rules and the
            // reason they differ). Also selects the precision the warp scalar is baked at, so the
            // spec constant and pc.scale describe the same operand by construction.
            static bool coordWordsFp32(const Graph &g, const Node &node, bool warp) {
                const TensorId coord = node.inputs[1];
                return warp ? gridSampleFlowWordsFp32(g.isInitializer(coord), g.desc(coord).storeFp32) :
                              gridSampleGridWordsFp32(g.isInitializer(coord), g.desc(coord).storeFp32);
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
                // Decode precision of the coordinate operand, shared by the spec constant below and
                // (for the warp fold) by the scalar the kernel multiplies that operand with.
                const bool coordFp32 = coordWordsFp32(g, node, warp);
                if (warp)
                {
                    // Bake the scale at the FLOW's storage precision: an fp16 flow narrows the scalar
                    // exactly as the standalone Mul's initializer was narrowed (the shader then rounds
                    // the product itself), a pinned fp32 flow keeps it exact so the full-precision
                    // coordinates stay full precision. gridSampleWarpScale holds the rule.
                    pc.scale = gridSampleWarpScale(node.attr.getf("warp_scale", 1.f), env.useFp16, coordFp32);
                    // Base grid [.,OH,OW,2]: N stride 0 broadcasts a single-frame base over the batch.
                    pc.baseNStride = g.desc(node.inputs[2]).shape[0] == 1 ? 0 : OH * OW * 2;
                }
                // MODE/PAD are baked as spec constants (constant_id 0/1) so the driver specializes the
                // sampler branch at pipeline build; gridsample_rule.h holds the attribute-to-selector
                // encoding the shaders mirror.
                const uint32_t MODE = gridSampleModeCode(node.attr.gets("mode", "bilinear"));
                const uint32_t PAD  = gridSamplePadCode(node.attr.gets("padding_mode", "zeros"));
                // One dispatch lane per output PIXEL. The kernel loops the NC4HW4 channel blocks
                // internally (cBlocks(c) ceil-packs channels into groups of 4, each resolved as a
                // vec4), so the grid fetch and the coordinate/weight math happen once per pixel
                // instead of once per block-pixel.
                total = (int64_t) x.n * OH * OW;
                epi.prepare(node, env, /*flat=*/false, g.desc(node.outputs[0]).shape);
                // The warp shader reads its flow (NC4HW4) and base (fp32) from two dedicated bindings;
                // the plain shader has one grid binding (kGridSampleWarpBuffers /
                // kGridSamplePlainBuffers). The base grid always uploads fp32, so it needs no decode
                // selector. epi.suffix() selects the matching _epi shader variant. Constant 2 carries
                // the coordinate operand's decode precision either way: GRID_FP32 for the plain
                // kernel's grid, FLOW_FP32 for the warp kernel's flow.
                const uint32_t        coordFp32Spec = coordFp32 ? 1u : 0u;
                std::vector<uint32_t> spec          = {MODE, PAD, coordFp32Spec, env.convLocalSize};
                const int             nbuf          = warp ? kGridSampleWarpBuffers : kGridSamplePlainBuffers;
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
                // Flat 1D grid over the packed output lanes at the load-resolved lane width.
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(total, env.convLocalSize));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::GridSample, GridSampleOp);
} // namespace vknn
