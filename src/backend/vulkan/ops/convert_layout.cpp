// ConvertLayout: NC4HW4 <-> flat row-major on the GPU. Inserted by the format pass at boundaries
// between NC4HW4-native ops (conv/pool) and the generic flat head ops. node.subOp = direction
// (0: NC4HW4 -> flat, 1: flat -> NC4HW4). Logical NCHW shape is identical on both sides.
// The format pass inserts ConvertLayout NODES only — no splice site builds this pipeline directly —
// so pickClayoutQuad below is the single kernel pick for every dispatch of these shaders.
#include "backend/vulkan/vk_tune_race.h"
#include "flat_ops.h"
#include "vk_op_common.h"
#include "vknn/logging.h"
#include "vknn/op.h"

namespace vknn {
    namespace {

        // Flat elements per lane of the vectorized kernel (whole-vec4 access on the flat side);
        // mirrors CLAYOUT_QUAD in shaders/convert_layout_v4.comp.
        constexpr int kClayoutQuad = kQuadBindingElems;
        // Tune-table values of the convert-layout kernel pick (append-only), index-aligned with the
        // family's shared race entrants (flat::kFlatKernelScalar / flat::kFlatKernelQuad).
        constexpr int kClayoutKernelScalar = flat::kFlatKernelScalar; ///< convert_layout: one NC4 lane-quad per lane.
        constexpr int kClayoutKernelQuad   = flat::kFlatKernelQuad;   ///< convert_layout_v4: one flat vec4 quad per lane.
        // Binding layout shared by both kernels: 0/1 are the scalar views and 2/3 the vec4 views of
        // the SAME src/dst buffers. Each kernel takes the vec4 view on its whole-vec4 side (the NC4
        // side for the scalar kernel, the flat side for the quad twin) and scalars on the other.
        constexpr uint32_t kClayoutBindings = 4;

        struct ConvertPC {
            int N, C, H, W, dir;
        };

        struct ConvertLayoutOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            ConvertPC                            pc {};
            uint32_t                             count = 0; // dispatch lanes of the picked kernel
            // convert_layout_v4 (one whole-vec4 quad of consecutive FLAT elements per lane) when
            // the cached race picked it; the scalar kernel (one NC4 lane-quad per lane, whole-vec4
            // on the NC4 side) otherwise. Byte-identical either way (see pickClayoutQuad).
            bool quadKernel = false;

            // The two kernels are byte-interchangeable wherever the quad is offered: elements are
            // copied, never combined, so byte identity is (src,dst) index-pair identity, and both
            // kernels map flat element (n, c, hw) through the same layout equation
            // a = ((n*Cb + c/4)*HW + hw)*4 + c%4. dir 0: both write every flat element exactly
            // once from that same NC4 position (the scalar kernel additionally LOADS pad lanes it
            // discards; reads do not change bytes). dir 1: the written pairs are again identical,
            // but the scalar kernel also zero-fills the pad lanes of a partial final channel block
            // (downstream NC4 consumers read them), which the quad twin never visits — so the quad
            // is offered only when C fills its blocks exactly. That gate is deterministic geometry
            // policy applied before the cache lookup, not a tuned value.
            //
            // The pick itself is placement-only and follows the standard cached race
            // (fused_pointwise.cpp pickFlatQuad): Tuning::None keeps the scalar kernel (the
            // deterministic default), Fast/Heavy race both once on dedicated scratch through
            // TuneTimer/raceCandidates and persist the winner; the quad challenger must clear
            // vk::kTuneRaceMargin to displace the incumbent. total, C and HW ride the signature
            // because they set the plane-run length (the quad's fast-path rate) and both sides'
            // coalescing classes; dir rides it because it flips which side is the scattered one.
            // convert_layout_v4 addresses BOTH the source and the destination through their vec4
            // views (bindings 2/3), so both are gated on vec4 alignment, ahead of the cache lookup:
            // a cached verdict is keyed on shape and would otherwise reach a node whose own buffers
            // cannot host the wider access.
            bool pickClayoutQuad(const Node &node, VkOpEnv &env, int64_t flatTotal, int64_t nc4Total, int64_t scalarLanes) {
                if (flatTotal <= 0 || !flat::quadBindingsAligned(env, {node.inputs[0], node.outputs[0]}))
                {
                    return false;
                }
                if (pc.dir == 1 && pc.C % (int) kNC4Block != 0)
                {
                    return false; // partial final block: only the scalar kernel zero-fills its pad lanes
                }
                char buf[96];
                snprintf(buf, sizeof(buf), "clayout_%d_%d_%d_%d_%d", (int) flatTotal, pc.C, pc.H * pc.W, pc.dir, env.useFp16 ? 1 : 0);
                std::string sig = env.gpuTag + "/" + buf;
                int         reuse;
                if (env.reuseTuned(sig, reuse) && (reuse == kClayoutKernelScalar || reuse == kClayoutKernelQuad))
                {
                    return reuse == kClayoutKernelQuad;
                }
                if (QuadVerdictMemo::lookup(sig, env.tuning, reuse) && (reuse == kClayoutKernelScalar || reuse == kClayoutKernelQuad))
                {
                    return reuse == kClayoutKernelQuad;
                }
                if (env.tuning == Tuning::None || !env.runner)
                {
                    return false;
                }
                // Dedicated scratch at the NC4 physical footprint, which is >= the flat footprint,
                // so one size covers either side of either direction for both candidates.
                constexpr int kClayoutRaceScratchBuffers = 2; // source, destination
                if (!flat::quadRaceAffordable(env, nc4Total * kClayoutRaceScratchBuffers))
                {
                    return false;
                }
                const size_t es = (size_t) (env.useFp16 ? kFp16StorageBytes : kFp32StorageBytes);
                auto         mk = [&](size_t bytes) {
                    return std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>(bytes, flat::kQuadRaceScratchFloorBytes), vk::MemPref::kDeviceOnly);
                };
                auto          sSrc = mk((size_t) nc4Total * es);
                auto          sDst = mk((size_t) nc4Total * es);
                vk::TuneTimer timer(env);
                auto          ms     = vk::raceCandidates(2, [&](int index) {
                    const bool q = index == kClayoutKernelQuad;
                    auto racePipe = env.pipeline(shader(q ? "convert_layout_v4" : "convert_layout", env.useFp16), kClayoutBindings, sizeof(ConvertPC), std::vector<uint32_t> {env.flatLocalSize});
                    const int64_t lanes = q ? (flatTotal + kClayoutQuad - 1) / kClayoutQuad : scalarLanes;
                    return timer.time([&](VkCommandBuffer cmd) {
                        racePipe->dispatch(cmd, {sSrc->handle(), sDst->handle(), sSrc->handle(), sDst->handle()}, &pc, sizeof(pc), groups(lanes, env.flatLocalSize));
                    });
                });
                const bool    won    = ms.size() == 2 && ms[kClayoutKernelQuad] < ms[kClayoutKernelScalar] * vk::kTuneRaceMargin;
                const int     kernel = won ? kClayoutKernelQuad : kClayoutKernelScalar;
                if (env.weights)
                {
                    env.weights->setTuned(sig, kernel, (int) env.tuning);
                }
                // Also memoized, so a session with no weight cache answers the next node carrying
                // this signature from the verdict instead of racing it again.
                QuadVerdictMemo::store(sig, env.tuning, kernel);
                VKNN_DEBUG << "autotune " << sig << " -> " << (won ? "quad" : "scalar") << vk::raceTimes(ms);
                return won;
            }

            void prepare(const Node &node, VkOpEnv &env) override {
                const Shape &shape = env.graph->desc(node.outputs[0]).shape;
                NCHW         x     = NCHW::from(shape);
                int          dir   = node.subOp; // 0: NC4HW4->flat, 1: flat->NC4HW4
                pc                 = {(int) x.n, (int) x.c, (int) x.h, (int) x.w, dir};
                int64_t Cb = cBlocks(x.c), HW = x.h * x.w;
                // Lane counts: the scalar kernel runs one lane per NC4 lane-quad (both directions);
                // the quad twin one lane per kClayoutQuad consecutive flat elements.
                // Both kernels index their buffers with int32 lane and element arithmetic, so the
                // extents enter the same element-count contract the rest of the flat family
                // enforces: a named refusal at load rather than a silent wrap at dispatch.
                const int64_t scalarLanes = flatElementCount(x.n * Cb * HW, "ConvertLayout NC4HW4 lane count");
                const int64_t flatTotal   = flatElementCount(x.n * x.c * HW, "ConvertLayout flat element count");
                quadKernel                = pickClayoutQuad(node, env, flatTotal, packedElems(shape), scalarLanes);
                count                     = (uint32_t) (quadKernel ? (flatTotal + kClayoutQuad - 1) / kClayoutQuad : scalarLanes);
                // Workgroup width rides spec constant 0, resolved at load (env.flatLocalSize); the
                // dispatch below divides by the same value.
                pipe = env.pipeline(shader(quadKernel ? "convert_layout_v4" : "convert_layout", env.useFp16), kClayoutBindings, sizeof(ConvertPC), std::vector<uint32_t> {env.flatLocalSize});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                // bindings 0/1 = scalar views, 2/3 = vec4 views of the same src/dst buffers
                vk::Buffer *s = env.devBuf(node.inputs[0]);
                vk::Buffer *d = env.devBuf(node.outputs[0]);
                pipe->dispatch(cmd, {s->handle(), d->handle(), s->handle(), d->handle()}, &pc, sizeof(pc), groups(count, env.flatLocalSize));
            }
        };

    } // namespace

    VKNN_REGISTER_VK_OP(OpType::ConvertLayout, ConvertLayoutOp);

} // namespace vknn
