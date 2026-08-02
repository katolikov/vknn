// Blocked reduction of a rank-4 tensor over its spatial plane, shared by GlobalAveragePool and the
// spatial arm of the Reduce family.
//
// A channel block's four lanes are four independent reductions that happen to travel together, so
// one vec4 accumulator serves all four and the NC4HW4 buffer is read exactly as it is stored -- no
// layout convert on either side. The reduction kind rides the push constant (shaders/
// nc4_reduce_codes.glsl), so mean, sum, max, min, prod and L2 share one pair of kernels.
//
// Two dispatch strategies, chosen by the channel-block count:
//   * single-pass (avgpool.comp): one workgroup per (n, channel-block), its threads cooperatively
//     reduce H*W. Optimal when there are many channel-blocks -- they already fill the GPU.
//   * cooperative two-pass (avgpool_partial + avgpool_combine): when there are few channel-blocks (a
//     shallow-channel reduction over a large plane), the single-pass kernel leaves the GPU idle.
//     Pass 1 fans H*W across `groups` workgroups per block (vec4 partials -> scratch); pass 2 folds
//     them and finalises. Same partial-buffer shape as the flat two-pass Reduce and conv.cpp.
#pragma once
#include "flat_ops.h" // flat::laneWidthPow2For (caps-clamped pow2 tree width)
#include "pw_plan.h"
#include "vk_op_common.h"

namespace vknn {

    /// Workgroup width of the blocked reduction kernels (spec 0), mirroring POOL_WG_MAX in
    /// shaders/avgpool*.comp. laneWidthPow2For clamps the family ceiling to the device's
    /// maxComputeWorkGroupInvocations and rounds down to a power of two: the clamp keeps the pipeline
    /// creatable on a device at the Vulkan floor of 128 invocations, and the power of two is what the
    /// shaders' halving fold needs to reach every lane.
    inline uint32_t poolTreeWidth(VkOpEnv &env) {
        return flat::laneWidthPow2For(env.ctx->caps(), flat::kFlatLocalSize);
    }

    /// avgpool_partial/avgpool_combine push constant: PoolPC's geometry plus the per-block group count.
    struct PoolPCGroups {
        int N, C, H, W, groups, op;
    };

    /// Blocked spatial reduction. `prepare` picks the strategy and builds the pipelines; `record`
    /// issues them. The caller supplies the ReduceType code the kernels branch on.
    struct Nc4SpatialReduce {
        std::shared_ptr<vk::ComputePipeline> pipe;        // single-pass
        std::shared_ptr<vk::ComputePipeline> partialPipe; // cooperative pass 1
        std::shared_ptr<vk::ComputePipeline> combinePipe; // cooperative pass 2
        std::shared_ptr<vk::Buffer>          scratch;     // vec4 fp32 partials, N*Cb*groups
        PoolPC                               pc {};
        PoolPCGroups                         pcg {};
        PwEpi                                epi;
        int64_t                              total = 0; // N*Cb dispatch group count
        bool                                 coop  = false;

        // Cooperation thresholds. Below kFewBlocks the block count alone cannot fill the GPU, and a
        // plane of at least kWidePlane elements is what makes splitting it worth a second pass.
        static constexpr int64_t kFewBlocks = 4096;
        static constexpr int64_t kWidePlane = 256;
        static constexpr int64_t kMaxGroups = 64;

        void prepare(const Node &node, VkOpEnv &env, int reduceOp) {
            const NCHW x  = NCHW::from(env.graph->desc(node.inputs[0]).shape);
            const int  Cb = (int) cBlocks(x.c);
            total         = (int64_t) x.n * Cb;
            epi.prepare(node, env, /*flat=*/false, env.graph->desc(node.outputs[0]).shape);

            const int64_t hw = (int64_t) x.h * x.w;
            coop             = (total <= kFewBlocks && hw >= kWidePlane);
            if (coop)
            {
                const int64_t byWork = hw / kWidePlane; // no more groups than chunks
                const int64_t byGrid = kFewBlocks / std::max<int64_t>(total, 1);
                const int     groups = (int) std::max<int64_t>(1, std::min({byWork, byGrid, kMaxGroups}));
                pcg                  = {(int) x.n, (int) x.c, (int) x.h, (int) x.w, groups, reduceOp};
                scratch = std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>((size_t) total * groups * 4 * sizeof(float), 16), vk::MemPref::kDeviceOnly);
                // pass 1 bindings: src(0), scratch(1). No epilogue on the partials.
                partialPipe = env.pipeline(shader("avgpool_partial", env.useFp16), 2, sizeof(PoolPCGroups), std::vector<uint32_t> {poolTreeWidth(env)});
                // pass 2 bindings: scratch(0), dst(1), then epilogue operands.
                combinePipe = env.pipeline(shader((std::string("avgpool_combine") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(PoolPCGroups), std::vector<uint32_t> {poolTreeWidth(env)});
            } else
            {
                // PoolPC is byte-matched to shaders/avgpool.comp's push constant {N, C, H, W, op}.
                pc = {(int) x.n, (int) x.c, (int) x.h, (int) x.w, reduceOp};
                pipe = env.pipeline(shader((std::string("avgpool") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(PoolPC), std::vector<uint32_t> {poolTreeWidth(env)});
            }
        }

        void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) {
            vk::Buffer *src = env.devBuf(node.inputs[0]);
            vk::Buffer *dst = env.devBuf(node.outputs[0]);
            if (coop)
            {
                // Pass 1: spatial partials (N*Cb*groups workgroups; the dispatch spills to y past the
                // device max x-group count, and the shaders fold x,y to a linear index).
                partialPipe->dispatch(cmd, {src->handle(), scratch->handle()}, &pcg, sizeof(pcg), (uint32_t) (total * pcg.groups));
                // Two dispatches in one record() are NOT auto-barriered; pass 2 reads the scratch pass 1
                // wrote (see scatternd.cpp / reduce.cpp).
                vk::computeBarrier(*env.ctx, cmd);
                // Pass 2: fold partials, finalise, run the epilogue, store (one workgroup per block).
                std::vector<VkBuffer> cbufs = {scratch->handle(), dst->handle()};
                epi.append(cbufs, node, env, dst->handle());
                combinePipe->dispatch(cmd, cbufs, &pcg, sizeof(pcg), (uint32_t) total);
            } else
            {
                // Binding order must match avgpool.comp: 0 = src, 1 = dst, then any epilogue operands.
                std::vector<VkBuffer> bufs = {src->handle(), dst->handle()};
                epi.append(bufs, node, env, dst->handle());
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), (uint32_t) total);
            }
        }
    };

} // namespace vknn
