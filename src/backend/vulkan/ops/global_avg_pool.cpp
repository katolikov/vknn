// GlobalAveragePool: average each channel over H*W.
//
// Two dispatch strategies, chosen by the channel-block count:
//   * single-pass (avgpool.comp): one workgroup per (n, channel-block), its 256 threads cooperatively
//     reduce H*W. Optimal when there are many channel-blocks — they already fill the GPU.
//   * cooperative two-pass (avgpool_partial + avgpool_combine): when there are few channel-blocks (a
//     shallow-channel global mean over a large plane), the single-pass kernel leaves the GPU idle. Pass 1
//     fans H*W across `groups` workgroups per block (vec4 partial sums -> scratch); pass 2 folds them and
//     divides. Same partial-buffer shape as the flat two-pass Reduce (reduce.cpp) and conv.cpp.
#include "pw_plan.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        // avgpool_partial/avgpool_combine push constant: PoolPC {N,C,H,W} plus the per-block group count.
        struct PoolPCGroups {
            int N, C, H, W, groups;
        };

        struct GlobalAvgPoolOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;        // single-pass
            std::shared_ptr<vk::ComputePipeline> partialPipe; // cooperative pass 1
            std::shared_ptr<vk::ComputePipeline> combinePipe; // cooperative pass 2
            std::shared_ptr<vk::Buffer>          scratch;     // vec4 fp32 partials, N*Cb*groups
            PoolPC                               pc {};
            PoolPCGroups                         pcg {};
            PwEpi                                epi;
            int64_t                              total  = 0;   // N*Cb dispatch group count
            bool                                 coop   = false;

            void prepare(const Node &node, VkOpEnv &env) override {
                NCHW x  = NCHW::from(env.graph->desc(node.inputs[0]).shape);
                int  Cb = (int) cBlocks(x.c);
                total   = (int64_t) x.n * Cb;
                epi.prepare(node, env, /*flat=*/false, env.graph->desc(node.outputs[0]).shape);

                int64_t hw = (int64_t) x.h * x.w;
                // Cooperate only when there are few channel-blocks to parallelise over and the plane is
                // large; otherwise the many blocks already fill the GPU on the single-pass kernel.
                coop = (total <= 4096 && hw >= 256);
                if (coop)
                {
                    int64_t byWork = hw / 256;                       // no more groups than 256-elem chunks
                    int64_t byGrid = 4096 / std::max<int64_t>(total, 1);
                    int     groups = (int) std::max<int64_t>(1, std::min({byWork, byGrid, (int64_t) 64}));
                    pcg            = {(int) x.n, (int) x.c, (int) x.h, (int) x.w, groups};
                    scratch        = std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>((size_t) total * groups * 4 * sizeof(float), 16), vk::MemPref::kDeviceOnly);
                    // pass 1 bindings: src(0), scratch(1). No epilogue on the partial sums.
                    partialPipe = env.pipeline(shader("avgpool_partial", env.useFp16), 2, sizeof(PoolPCGroups), std::vector<uint32_t> {});
                    // pass 2 bindings: scratch(0), dst(1), then epilogue operands.
                    combinePipe = env.pipeline(shader((std::string("avgpool_combine") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(PoolPCGroups), std::vector<uint32_t> {});
                }
                else
                {
                    // PoolPC is byte-matched to shaders/avgpool.comp's push constant {N, C, H, W}.
                    pc   = {(int) x.n, (int) x.c, (int) x.h, (int) x.w};
                    pipe = env.pipeline(shader((std::string("avgpool") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(PoolPC), std::vector<uint32_t> {});
                }
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *src = env.devBuf(node.inputs[0]);
                vk::Buffer *dst = env.devBuf(node.outputs[0]);
                if (coop)
                {
                    // Pass 1: spatial partials (N*Cb*groups workgroups; dispatch spills to y past the
                    // device max x-group count, the shaders fold x,y to a linear index).
                    partialPipe->dispatch(cmd, {src->handle(), scratch->handle()}, &pcg, sizeof(pcg), (uint32_t) (total * pcg.groups));
                    // Two dispatches in one record() are NOT auto-barriered; pass 2 reads the scratch pass 1
                    // wrote (see scatternd.cpp / reduce.cpp).
                    vk::computeBarrier(*env.ctx, cmd);
                    // Pass 2: fold partials, divide, run the epilogue, store (one workgroup per block).
                    std::vector<VkBuffer> cbufs = {scratch->handle(), dst->handle()};
                    epi.append(cbufs, node, env, dst->handle());
                    combinePipe->dispatch(cmd, cbufs, &pcg, sizeof(pcg), (uint32_t) total);
                }
                else
                {
                    // Binding order must match avgpool.comp: 0 = src, 1 = dst, then any epilogue operands.
                    std::vector<VkBuffer> bufs = {src->handle(), dst->handle()};
                    epi.append(bufs, node, env, dst->handle());
                    pipe->dispatch(cmd, bufs, &pc, sizeof(pc), (uint32_t) total);
                }
            }
        };

    } // namespace

    VKNN_REGISTER_VK_OP(OpType::GlobalAvgPool, GlobalAvgPoolOp);

} // namespace vknn
