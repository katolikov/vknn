// Resize (spatial) on the GPU (NC4HW4): nearest + bilinear + cubic, per output channel-block.
#include "backend/vulkan/resize_race_scratch.h"
#include "backend/vulkan/vk_tune_race.h"
#include "core/resize_rule.h"
#include "pw_plan.h"
#include "vk_op_common.h"
#include "vknn/logging.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        // Local workgroup size along x; matches local_size_x in shaders/resize.comp and
        // shaders/resize_row.comp.
        constexpr uint32_t kResizeLocalSize = 64;

        // Output pixels one row-kernel lane computes along its row; mirrors RESIZE_ROW_TILE in
        // shaders/resize_row.comp.
        constexpr int kResizeRowTile = 4;
        // Floor on a race scratch allocation. A degenerate geometry sizes a buffer at zero bytes,
        // which is not a legal VkBuffer; one NC4HW4 storage element is the smallest that is.
        constexpr size_t kResizeMinScratchBytes = sizeof(float) * (size_t) kNC4Block;
        // Tune-table values of the resize kernel pick (append-only).
        constexpr int kResizeKernelScalar = 0; ///< resize: one output pixel or block-pixel per lane.
        constexpr int kResizeKernelRow    = 1; ///< resize_row: kResizeRowTile row pixels per lane.

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

        // The row-tiled kernel serves the nearest and bilinear arms only: their per-output work is
        // a pure function of tap columns and weights that a row tile can share loads across without
        // touching any accumulation. The cubic arm's 4x4 tap window would need a 4-column ring per
        // row to share exactly; it stays on the scalar kernel. Same gating role as conv.cpp's
        // tinyOcOk/lds3x3 eligibility flags: the predicate bounds where the race may even offer
        // the entrant.
        inline bool resizeRowKernelOk(const ResizePC &pc) {
            return pc.mode == kResizeModeNearest || pc.mode == kResizeModeLinear;
        }
        // Lane count of the row kernel: one lane per (n, channel block, output row,
        // kResizeRowTile-wide row tile); mirrors the grid bound in shaders/resize_row.comp.
        inline int64_t resizeRowLanes(const ResizePC &pc) {
            return (int64_t) pc.N * cBlocks(pc.C) * pc.OH * ((pc.OW + kResizeRowTile - 1) / kResizeRowTile);
        }

        struct ResizeOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            ResizePC                             pc {};
            PwEpi                                epi;
            int64_t                              total = 0;
            // resize_row (kResizeRowTile consecutive row pixels per lane, tap loads shared across
            // the tile) when the cached race picked it; the scalar kernel otherwise.
            // Byte-identical either way (see pickResizeRow).
            bool rowKernel = false;

            // The scalar and row kernels are byte-interchangeable wherever resizeRowKernelOk
            // holds: the row kernel computes each output pixel from the same taps with the same
            // weights and the same accumulation expression, and only shares WHEN a tap is loaded
            // across its tile (shaders/resize_row.comp carries the per-arm argument). The pick is
            // therefore placement-only and follows the standard cached race (cast.cpp
            // pickCastQuad): Tuning::None keeps the scalar kernel (the deterministic default),
            // Fast/Heavy race both once on dedicated scratch through TuneTimer/raceCandidates and
            // persist the winner; the row challenger must clear vk::kTuneRaceMargin to displace
            // the incumbent. The race dispatches the SAME variant the node will (same epilogue
            // suffix, same binding count — the appendForTiming rule). The sig carries the
            // geometry that sets the access pattern: the interpolation / coordinate /
            // nearest-rounding codes and the full in/out extents (which encode the scale class
            // and with it the tap-reuse rate), N and C (the lane counts), plus the epilogue kind
            // and precision. scalarLanes is the incumbent's lane count under the map
            // resizePerPixelLanes chose — the race times the scalar kernel exactly as the graph
            // would run it.
            bool pickResizeRow(VkOpEnv &env, int64_t scalarLanes) {
                if (!resizeRowKernelOk(pc) || scalarLanes <= 0)
                {
                    return false;
                }
                const int epiKind = !epi.active ? 0 : epi.relax ? 2 : 1;
                char      buf[128];
                snprintf(buf, sizeof(buf), "resizerow_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d", pc.mode, pc.cm, pc.nm, pc.N, pc.C, pc.IH, pc.IW, pc.OH, pc.OW, epiKind, env.useFp16 ? 1 : 0);
                std::string sig = env.gpuTag + "/" + buf;
                int         reuse;
                if (env.reuseTuned(sig, reuse) && (reuse == kResizeKernelScalar || reuse == kResizeKernelRow))
                {
                    return reuse == kResizeKernelRow;
                }
                if (env.tuning == Tuning::None || !env.runner)
                {
                    return false;
                }
                // The scratch is sized by the node's own geometry, so a large upscale asks for two
                // output-sized buffers on top of a resident arena. Past the budget the incumbent
                // stands: nothing is measured and nothing is stored, so a device or a bucket where
                // the same signature fits still races it (resize_race_scratch.h).
                if (!resizeRaceScratchFits(pc.N, pc.C, pc.IH, pc.IW, pc.OH, pc.OW, epi.active, env.useFp16))
                {
                    VKNN_DEBUG << "autotune " << sig << " -> scalar (race scratch " << resizeRaceScratchBytes(pc.N, pc.C, pc.IH, pc.IW, pc.OH, pc.OW, epi.active, env.useFp16) << " B over the " << kResizeRaceScratchBudgetBytes << " B budget)";
                    return false;
                }
                // Scratch sized like the real NC4HW4 tensors (resizeNc4TensorBytes: one vec4
                // storage element per (n, block, pixel)), which is what the budget above accounts
                // for. The epilogue's operand filler is output-sized (the appendForTiming rule: the
                // epilogue indexes operands by output element).
                auto mk = [&](size_t bytes) {
                    return std::make_shared<vk::Buffer>(*env.ctx, std::max(bytes, kResizeMinScratchBytes), vk::MemPref::kDeviceOnly);
                };
                auto                        sSrc = mk(resizeNc4TensorBytes(pc.N, pc.C, pc.IH, pc.IW, env.useFp16));
                auto                        sDst = mk(resizeNc4TensorBytes(pc.N, pc.C, pc.OH, pc.OW, env.useFp16));
                std::shared_ptr<vk::Buffer> sFill;
                if (epi.active)
                {
                    sFill = mk(resizeNc4TensorBytes(pc.N, pc.C, pc.OH, pc.OW, env.useFp16));
                }
                vk::TuneTimer timer(env);
                auto          ms  = vk::raceCandidates(2, [&](int index) {
                    const bool row = index == kResizeKernelRow;
                    auto racePipe = env.pipeline(shader((std::string(row ? "resize_row" : "resize") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(ResizePC), std::vector<uint32_t> {});
                    const int64_t lanes = row ? resizeRowLanes(pc) : scalarLanes;
                    return timer.time([&](VkCommandBuffer cmd) {
                        std::vector<VkBuffer> bufs {sSrc->handle(), sDst->handle()};
                        epi.appendForTiming(bufs, sFill ? sFill->handle() : sDst->handle());
                        racePipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(lanes, kResizeLocalSize));
                    });
                });
                const bool    won = ms.size() == 2 && ms[kResizeKernelRow] < ms[kResizeKernelScalar] * vk::kTuneRaceMargin;
                if (env.weights)
                {
                    env.weights->setTuned(sig, won ? kResizeKernelRow : kResizeKernelScalar, (int) env.tuning);
                }
                VKNN_DEBUG << "autotune " << sig << " -> " << (won ? "row" : "scalar") << vk::raceTimes(ms);
                return won;
            }

            // The geometry both kernels resolve exactly, refused here rather than truncated into
            // the push constants. The extent bound is what keeps the shaders' nearestSrc inside
            // 32-bit arithmetic (kResizeMaxSpatialExtent); a source plane of zero extent has no
            // pixel to sample, so the tap clamps would fold to an empty range and the lane would
            // index a plane that holds nothing. Both are the refusals the CPU oracle makes, so the
            // two backends accept exactly the same set of shapes.
            static void refuseUnresolvableGeometry(const Node &node, const NCHW &x, const NCHW &y) {
                if (x.h > kResizeMaxSpatialExtent || x.w > kResizeMaxSpatialExtent || y.h > kResizeMaxSpatialExtent || y.w > kResizeMaxSpatialExtent)
                {
                    throw Error(Status::Unsupported, "Resize '" + node.name + "': spatial extent " + std::to_string(x.h) + "x" + std::to_string(x.w) + " -> " + std::to_string(y.h) + "x" + std::to_string(y.w) + " exceeds the exactly-resolvable bound " + std::to_string(kResizeMaxSpatialExtent) + " (past it the source index has no exact form in the 32-bit integers the kernels evaluate)");
                }
                if ((x.h <= 0 || x.w <= 0) && y.h > 0 && y.w > 0)
                {
                    throw Error(Status::Unsupported, "Resize '" + node.name + "': input spatial extent " + std::to_string(x.h) + "x" + std::to_string(x.w) + " holds no pixels, so the requested " + std::to_string(y.h) + "x" + std::to_string(y.w) + " output has nothing to sample");
                }
            }

            void prepare(const Node &node, VkOpEnv &env) override {
                NCHW x = NCHW::from(env.graph->desc(node.inputs[0]).shape);
                NCHW y = NCHW::from(env.graph->desc(node.outputs[0]).shape);
                refuseUnresolvableGeometry(node, x, y);
                pc = {(int) x.n,
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
                epi.prepare(node, env, /*flat=*/false, env.graph->desc(node.outputs[0]).shape);
                // Scalar-kernel lane count for the map resizePerPixelLanes chose: one per output
                // pixel (the kernel loops the channel blocks internally) or one per NC4HW4
                // block-pixel. Spatial extent uses the OUTPUT y.h/y.w (the resize target size).
                const int64_t scalarLanes = (int64_t) x.n * y.h * y.w * (pc.perPixel != 0 ? 1 : cBlocks(x.c));
                rowKernel                 = pickResizeRow(env, scalarLanes);
                total                     = rowKernel ? resizeRowLanes(pc) : scalarLanes;
                // Base binding count is 2 (src, dst); a fused pointwise epilogue appends its own operand
                // buffers via extraBufs() and swaps in the _epi shader variant through suffix().
                pipe = env.pipeline(shader((std::string(rowKernel ? "resize_row" : "resize") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(ResizePC), std::vector<uint32_t> {});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer           *s    = env.devBuf(node.inputs[0]);
                vk::Buffer           *d    = env.devBuf(node.outputs[0]);
                std::vector<VkBuffer> bufs = {s->handle(), d->handle()};
                epi.append(bufs, node, env, d->handle());
                // kResizeLocalSize matches both resize kernels' local_size_x; groups() ceil-divides the
                // picked kernel's lane count (total, set in prepare) so the 1D grid covers every lane
                // (the shader's own bounds check drops the tail).
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(total, kResizeLocalSize));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Resize, ResizeOp);
} // namespace vknn
