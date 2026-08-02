// Reduce family on the GPU (FLAT row-major path): ReduceMean/Sum/Max/Min/Prod/L2 over a set of axes.
// The layout pass routes Reduce to the flat path; axes come from the `axes` attr or input[1].
//
// Two dispatch strategies, chosen by output size:
//   * scalar (flat_reduce.comp): one thread per output element loops the reduced axes. Optimal when the
//     output is large — the many outputs already fill the GPU.
//   * cooperative two-pass (flat_reduce_partial + flat_reduce_combine): when the output is small and the
//     reduced extent is large (a global ReduceMax/Min/Sum over H*W), the scalar path pins the whole
//     reduction to a handful of lanes. Pass 1 fans the reduced extent across `groups` workgroups per
//     output (fp32 partials -> scratch); pass 2 folds the partials and finalises. This is the same
//     partial-buffer pattern conv.cpp uses, and turns the pathological ~1-lane case into a full-GPU one.
#include "dispatch_extent.h"
#include "flat_ops.h"
#include "nc4_spatial_reduce.h"
#include "pw_plan.h"
#include "vk_op_common.h"
#include "vknn/error.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        // Field order/types mirror flat_reduce.comp's push_constant block. `op` is the ReduceType sub-op
        // code (Mean=0, Sum=1, Max=2, Min=3, Prod=4, L2=5) the shader branches on. inDim/inStride/reduce
        // ride the geometry SSBO (unbounded rank), not the push constant.
        struct ReducePCFlat {
            int rank, total, op;
        };
        // flat_reduce_partial.comp: adds `groups`, the workgroup count per output over which the reduced
        // extent is split. flat_reduce_combine.comp: folds `groups` partials, `rcount` finalises the Mean.
        struct ReducePCPartial {
            int rank, total, op, groups;
        };
        struct ReducePCCombine {
            int total, op, groups, rcount;
        };

        struct ReduceOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;        // scalar path
            std::shared_ptr<vk::ComputePipeline> partialPipe; // cooperative pass 1
            std::shared_ptr<vk::ComputePipeline> combinePipe; // cooperative pass 2
            std::shared_ptr<vk::Buffer>          geom;        // inDim/inStride/reduce, deduped SSBO
            std::shared_ptr<vk::Buffer>          scratch;     // fp32 partials, total*groups (cooperative only)
            PwEpi                                epi;
            ReducePCFlat                         pc {};
            ReducePCPartial                      partialPc {};
            ReducePCCombine                      combinePc {};
            bool                                 coop = false;
            // Set when the layout pass placed this node on the blocked path: a spatial reduction is
            // one reduction per channel, so the four lanes of a channel block reduce independently
            // and the NC4HW4 buffer is read as stored (nc4_spatial_reduce.h).
            bool             nc4 = false;
            Nc4SpatialReduce blocked;

            void prepare(const Node &node, VkOpEnv &env) override {
                nc4 = !opIsFlat(node, env);
                if (nc4)
                {
                    blocked.prepare(node, env, node.subOp);
                    return;
                }
                const Graph &g    = *env.graph;
                Shape        in   = g.desc(node.inputs[0]).shape;
                int          rank = (int) in.size();
                // Axes come from the `axes` attr or input[1] (readI64Param checks both); an empty list is
                // the ONNX "reduce over every axis" default, giving a scalar output.
                std::vector<int64_t> axes = readI64Param(g, node, "axes", 1);
                if (axes.empty())
                {
                    for (int k = 0; k < rank; ++k)
                    {
                        axes.push_back(k); // reduce all
                    }
                }
                auto                 inStride = flat::rowStrides(in);
                std::vector<int32_t> inDim(rank), inStr(rank), reduce(rank, 0);
                // Build the reduce mask: normalize each ONNX axis (negative counts from the end) and set
                // its lane. Out-of-range axes are ignored so a malformed attr can't index past rank.
                for (int64_t a: axes)
                {
                    int ax = (int) (a < 0 ? a + rank : a);
                    if (ax >= 0 && ax < rank)
                    {
                        reduce[ax] = 1;
                    }
                }
                // Every quantity the kernels carry is an int32 field they then do int32 arithmetic on:
                // the geometry SSBO's dims and strides, the output total, and the reduced-element count
                // that finalises a Mean. Each is checked against the family's element domain BEFORE it
                // is narrowed, because a wrapped reduced count fails the combine pass's
                // `pc.rcount > 0` test and makes ReduceMean silently return the raw sum
                // (dispatch_extent.h).
                const int64_t outputElements = numElements(g.desc(node.outputs[0]).shape);
                int64_t       reducedExtent  = 1;
                for (int k = 0; k < rank; ++k)
                {
                    if (reduce[k])
                    {
                        reducedExtent *= in[k];
                    }
                }
                auto refuseExtent = [&](const char *quantity, int64_t extent) {
                    throw Error(Status::Unsupported, flat::dispatchExtentRefusal("Reduce '" + node.name + "'", quantity, extent));
                };
                if (!flat::dispatchExtentFits(outputElements))
                {
                    refuseExtent("output element count", outputElements);
                }
                if (!flat::dispatchExtentFits(reducedExtent))
                {
                    refuseExtent("reduced element count", reducedExtent);
                }
                for (int k = 0; k < rank; ++k)
                {
                    if (!flat::dispatchExtentFits(in[k]))
                    {
                        refuseExtent("input dimension", in[k]);
                    }
                    if (!flat::dispatchExtentFits(inStride[k]))
                    {
                        refuseExtent("input stride", inStride[k]);
                    }
                    inDim[k] = (int) in[k];
                    inStr[k] = (int) inStride[k];
                }
                int total = (int) outputElements;
                geom      = flat::uploadFlatGeom(env, {inDim, inStr, reduce});
                // Cooperate only for the small-output case; large-output reductions already parallelise
                // fully on the scalar path and would pay the two-pass barrier for nothing.
                coop = (total <= kReduceSaturatingGroups && reducedExtent >= kReduceMinChunk);
                // A pointwise chain can be folded into the reduce's store: epi.suffix() selects the _epi
                // shader variant and epi.extraBufs() adds its operand bindings. The epilogue runs at the
                // final store — the scalar kernel, or the combine pass for the cooperative path.
                epi.prepare(node, env, /*flat=*/true, g.desc(node.outputs[0]).shape);

                if (coop)
                {
                    // groups per output, under the three shared occupancy gates (their rationale, and
                    // why they cannot be device-derived, lives with them in nc4_spatial_reduce.h).
                    int64_t byWork = reducedExtent / kReduceMinChunk;
                    int64_t byGrid = kReduceSaturatingGroups / std::max<int64_t>(total, 1);
                    int     groups = (int) std::max<int64_t>(1, std::min({byWork, byGrid, kReduceMaxSplit}));
                    partialPc      = {rank, total, node.subOp, groups};
                    combinePc      = {total, node.subOp, groups, (int) reducedExtent};
                    scratch = std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>((size_t) total * groups * sizeof(float), 16), vk::MemPref::kDeviceOnly);
                    // pass 1 bindings: input(0), scratch(1), geom(2). No epilogue on the partials.
                    partialPipe = env.pipeline(shader("flat_reduce_partial", env.useFp16), 3, sizeof(ReducePCPartial),
                                               std::vector<uint32_t> {flat::laneWidthPow2For(env.ctx->caps(), flat::kFlatLocalSize)});
                    // pass 2 bindings: scratch(0), output(1), then epilogue operands.
                    combinePipe = env.pipeline(shader((std::string("flat_reduce_combine") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(ReducePCCombine),
                                               std::vector<uint32_t> {flat::laneWidthPow2For(env.ctx->caps(), flat::kFlatLocalSize)});
                } else
                {
                    pc = {rank, total, node.subOp};
                    // scalar bindings: input(0), output(1), geom(2), then epilogue operands.
                    pipe = env.pipeline(shader((std::string("flat_reduce") + epi.suffix()).c_str(), env.useFp16), 3 + epi.extraBufs(), sizeof(ReducePCFlat), std::vector<uint32_t> {env.flatLocalSize});
                }
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                if (nc4)
                {
                    blocked.record(cmd, node, env);
                    return;
                }
                VkBuffer dst = env.devBuf(node.outputs[0])->handle();
                if (coop)
                {
                    // Pass 1: input reduced into fp32 partials (total*groups workgroups; dispatch spills
                    // past the device max x-group count and the shaders fold x,y to a linear index).
                    std::vector<VkBuffer> pbufs = {env.devBuf(node.inputs[0])->handle(), scratch->handle(), geom->handle()};
                    partialPipe->dispatch(cmd, pbufs, &partialPc, sizeof(partialPc), (uint32_t) ((int64_t) partialPc.total * partialPc.groups));
                    // Two dispatches in one record() are NOT auto-barriered; pass 2 reads the scratch pass 1
                    // wrote, so a compute->compute barrier is required (see scatternd.cpp).
                    vk::computeBarrier(*env.ctx, cmd);
                    // Pass 2: fold partials, finalise, run the epilogue, store (one workgroup per output).
                    std::vector<VkBuffer> cbufs = {scratch->handle(), dst};
                    epi.append(cbufs, node, env, dst);
                    combinePipe->dispatch(cmd, cbufs, &combinePc, sizeof(combinePc), (uint32_t) combinePc.total);
                } else
                {
                    // Binding order matches flat_reduce.comp: input(0), output(1), geometry(2), epilogue.
                    std::vector<VkBuffer> bufs = {env.devBuf(node.inputs[0])->handle(), dst, geom->handle()};
                    epi.append(bufs, node, env, dst);
                    pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(pc.total, env.flatLocalSize));
                }
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Reduce, ReduceOp);
} // namespace vknn
