// Fused Squeeze-Excite scale on the GPU (core/squeeze_excite.h). Two dispatches replace the
// GlobalAvgPool (one or two), the two 1x1 convs and the activation/gate nodes of the chain:
//   1. avgpool_partial reduces the feature map's spatial plane to fp32 vec4 partial sums, one per
//      (channel block, split) — the same cooperative pass and split rule the standalone pool uses,
//      so it fills the GPU on a shallow-channel, large-plane block and still runs one workgroup per
//      block on a small plane.
//   2. fused_se folds the splits into the pooled vector, runs FC1 -> activation -> FC2 -> gate in
//      shared memory and stores the channel scale, one workgroup per image, on transposed weights
//      so every wave load is contiguous (a one-workgroup kernel is latency-bound: the first form
//      of this kernel, one output per thread over the untransposed rows, cost 0.35 ms on a
//      1152-channel block; this form 0.23 ms).
// The broadcast Mul that applies the scale stays a separate op.
//
// Measured on the release device (EfficientNet-B0, 16 blocks): the fused pair costs 0.025-0.23 ms
// per block against ~0.05 ms for the unfused GlobalAvgPool + two 1x1 convs, which already run at
// the per-dispatch floor on a [N,C,1,1] tensor; the fusion saves at most one dispatch per block
// and pays for it with a latency-bound single-workgroup FC. It therefore stays opt-in (-O2).
#include "core/squeeze_excite.h"
#include "nc4_spatial_reduce.h" // avgpool_partial geometry: poolTreeWidth, kReduce* gates, PoolPCGroups
#include "vknn/op.h"
#include "vknn/reduce_type.h"

namespace vknn {
    namespace {
        // Field order/types mirror fused_se.comp's push_constant block.
        struct SePC {
            int   N, C, Cr, groups, hw, act, gate;
            float alpha, beta;
        };
        // [rows][cols] row-major -> [cols][rows].
        std::vector<float> transposeMatrix(const std::vector<float> &m, int64_t rows, int64_t cols) {
            std::vector<float> t((size_t) (rows * cols), 0.f);
            for (int64_t r = 0; r < rows; ++r)
            {
                for (int64_t c = 0; c < cols; ++c)
                {
                    const size_t src = (size_t) (r * cols + c);
                    if (src < m.size())
                    {
                        t[(size_t) (c * rows + r)] = m[src];
                    }
                }
            }
            return t;
        }
        struct FusedSeOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> partialPipe, pipe;
            std::shared_ptr<vk::Buffer>          scratch, w1, b1, w2, b2;
            PoolPCGroups                         pcg {};
            SePC                                 pc {};
            int64_t                              blocks = 0; // N * channel blocks: partial workgroups per split
            void                                 prepare(const Node &node, VkOpEnv &env) override {
                const Graph  &g  = *env.graph;
                const NCHW    x  = NCHW::from(g.desc(node.inputs[0]).shape);
                const int64_t C  = x.c, Cr = g.desc(node.inputs[1]).shape[0]; // W1 is [Cr][C], row count is Cr
                const int64_t hw = x.h * x.w;
                const int     act  = seActCode(node.fusedAct);
                const int     gate = seGateCode((UnaryType) node.subOp);
                if (act == kSeCodeNone || gate == kSeCodeNone || !seShapeFits(C, Cr))
                {
                    throw Error(Status::Unsupported, "FusedSE '" + node.name + "': activation, gate or widths outside the fused kernel (C " + std::to_string(C) + ", squeeze " + std::to_string(Cr) + ")");
                }
                blocks = x.n * cBlocks(C);
                // The split rule of the standalone pool: split a shallow-channel plane across groups
                // until the dispatch fills the GPU, never below kReduceMinChunk elements per group.
                int groups = 1;
                if (blocks <= kReduceSaturatingGroups && hw >= kReduceMinChunk)
                {
                    const int64_t byWork = hw / kReduceMinChunk;
                    const int64_t byGrid = kReduceSaturatingGroups / std::max<int64_t>(blocks, 1);
                    groups               = (int) std::max<int64_t>(1, std::min({byWork, byGrid, kReduceMaxSplit}));
                }
                pcg = {(int) x.n, (int) C, (int) x.h, (int) x.w, groups};
                pc  = {(int) x.n, (int) C, (int) Cr, groups, (int) hw, act, gate, node.actLo, node.actHi};
                constexpr size_t kMinScratchBytes = 16; // a degenerate plan never asks for a zero-byte buffer
                const size_t     partialBytes     = (size_t) blocks * groups * kNC4Block * sizeof(float);
                scratch                           = std::make_shared<vk::Buffer>(*env.ctx, std::max(partialBytes, kMinScratchBytes), vk::MemPref::kDeviceOnly);
                partialPipe = env.pipeline(shader("avgpool_partial", env.useFp16), 2, sizeof(PoolPCGroups), std::vector<uint32_t> {poolTreeWidth(env), (uint32_t) ReduceType::Mean});
                // Both matrices upload transposed (W1T [C][Cr], W2T [Cr][C]) so the kernel's wave reads
                // consecutive elements at every step (see fused_se.comp).
                w1 = uploadCached(env, node.name + "#w1t", [&] {
                    return transposeMatrix(initFloats(g, node.inputs[1]), Cr, C); // [Cr][C] -> [C][Cr]
                });
                w2 = uploadCached(env, node.name + "#w2t", [&] {
                    return transposeMatrix(initFloats(g, node.inputs[3]), C, Cr); // [C][Cr] -> [Cr][C]
                });
                // The biases are optional on the node but the kernel always reads them: an absent bias
                // uploads as zeros at the exact FC width (b1 has Cr entries, b2 has C).
                b1 = uploadCached(env, node.name + "#b1", [&] {
                    std::vector<float> v((size_t) Cr, 0.f);
                    if (node.inputs[2] != kNoTensor)
                    {
                        std::vector<float> t = initFloats(g, node.inputs[2]);
                        for (int64_t i = 0; i < Cr && i < (int64_t) t.size(); ++i)
                        {
                            v[(size_t) i] = t[(size_t) i];
                        }
                    }
                    return v;
                });
                b2 = uploadCached(env, node.name + "#b2", [&] {
                    std::vector<float> v((size_t) C, 0.f);
                    if (node.inputs[4] != kNoTensor)
                    {
                        std::vector<float> t = initFloats(g, node.inputs[4]);
                        for (int64_t i = 0; i < C && i < (int64_t) t.size(); ++i)
                        {
                            v[(size_t) i] = t[(size_t) i];
                        }
                    }
                    return v;
                });
                pipe = env.pipeline(shader("fused_se", env.useFp16), 6, sizeof(SePC), std::vector<uint32_t> {});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *x = env.devBuf(node.inputs[0]);  // feature map in
                vk::Buffer *s = env.devBuf(node.outputs[0]); // channel scale out
                // Pass 1: spatial partials (blocks * groups workgroups; the dispatch spills to y past
                // the device max x-group count, and the shader folds x,y to a linear index).
                partialPipe->dispatch(cmd, {x->handle(), scratch->handle()}, &pcg, sizeof(pcg), (uint32_t) (blocks * pcg.groups));
                // Two dispatches in one record() are not auto-barriered; pass 2 reads pass 1's scratch.
                vk::computeBarrier(*env.ctx, cmd);
                // Pass 2: pool, FC1, activation, FC2, gate (one workgroup per image).
                pipe->dispatch(cmd, {scratch->handle(), w1->handle(), b1->handle(), w2->handle(), b2->handle(), s->handle()}, &pc, sizeof(pc), (uint32_t) pc.N);
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::FusedSE, FusedSeOp);
} // namespace vknn
