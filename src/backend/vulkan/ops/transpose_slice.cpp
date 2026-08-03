// Transpose and Slice on the GPU. Flat row-major, except a channel-last (NCHW->NHWC) Transpose,
// which reads its input in NC4HW4 (transposeReadsNc4) so no ConvertLayout precedes it.
#include "core/slice_bounds.h"
#include "flat_ops.h"

namespace vknn {
    namespace {

        // Both Transpose and Slice are pure index remaps with no arithmetic, so they share one gather
        // kernel: out[i] = in[base + sum_k outCoord_k * inStride_k]. Transpose permutes the per-axis
        // strides (base 0); Slice folds start offsets into base and the step into inStride. flat::Gather
        // builds the matching push constants per op type in its prepare(). The one exception is the
        // channel-last Transpose below, which reads the packed layout instead of a converted copy.

        // Byte-matched to shaders/transpose_nhwc.comp's push_constant block { int N, C, H, W }.
        struct TransposeNhwcPC {
            int N, C, H, W;
        };

        struct TransposeOp: VulkanOp {
            flat::Gather impl;
            // The channel-last path (transposeReadsNc4): the input stays NC4HW4 and one coalesced
            // kernel does the whole reindex, instead of a full-size ConvertLayout plus a per-element
            // flat gather. nc4Pipe is null for every other Transpose.
            std::shared_ptr<vk::ComputePipeline> nc4Pipe;
            TransposeNhwcPC                      nc4Pc {};
            uint32_t                             nc4Count = 0;

            void prepare(const Node &node, VkOpEnv &env) override {
                if (transposeReadsNc4(*env.graph, node))
                {
                    NCHW x   = NCHW::from(env.graph->desc(node.inputs[0]).shape);
                    nc4Pc    = {(int) x.n, (int) x.c, (int) x.h, (int) x.w};
                    nc4Count = (uint32_t) (x.n * x.h * x.w * cBlocks(x.c)); // one lane per (pixel, channel block)
                    // The pipeline's workgroup-size spec constant and record()'s group count both
                    // derive from env.flatLocalSize, so the two can never disagree.
                    nc4Pipe = env.pipeline(shader("transpose_nhwc", env.useFp16), 2, sizeof(TransposeNhwcPC), std::vector<uint32_t> {env.flatLocalSize});
                    return;
                }
                impl.prepare(node, env);
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                if (nc4Pipe)
                {
                    vk::Buffer *s = env.devBuf(node.inputs[0]);
                    vk::Buffer *d = env.devBuf(node.outputs[0]);
                    nc4Pipe->dispatch(cmd, {s->handle(), d->handle()}, &nc4Pc, sizeof(nc4Pc), groups(nc4Count, env.flatLocalSize));
                    return;
                }
                // An identity-perm Transpose is aliased onto its input by the segment planner (the
                // same geometry-as-metadata rule as Reshape); the gather is then a no-op. Same
                // guards as Slice: a fused epilogue must run, and an initializer operand is never
                // aliased.
                if (!node.attr.has("pw_steps") && !node.inputs.empty() && node.inputs[0] != kNoTensor && !env.graph->isInitializer(node.inputs[0]))
                {
                    vk::Buffer *src = env.devBuf(node.inputs[0]);
                    vk::Buffer *dst = env.devBuf(node.outputs[0]);
                    if (src && dst && src->handle() == dst->handle())
                    {
                        return;
                    }
                }
                impl.record(cmd, node, env);
            }
        };
        struct SliceOp: VulkanOp {
            flat::Gather impl;
            // ---- NC4HW4 channel path ----
            // A block-aligned channel slice (sliceIsNc4) takes a contiguous run of channel blocks out
            // of each batch, so it moves buffer ranges instead of gathering element by element -- and
            // when the planner already aliased the output onto that range, it moves nothing at all.
            // Geometry mirrors split.cpp: NC4HW4 bytes are [n][channelBlock][h*w][kNC4Block] * elem.
            bool    nc4_ = false;
            NCHW    x_ {};
            int64_t cbTotal_  = 0; // the input's channel-block count (its per-batch pitch)
            int64_t cbOut_    = 0; // channel blocks this slice takes
            int64_t blockOff_ = 0; // first input block the slice starts at
            int64_t hw_       = 0;
            int64_t elem_     = 0;

            void prepare(const Node &node, VkOpEnv &env) override {
                nc4_ = !opIsFlat(node, env);
                if (!nc4_)
                {
                    impl.prepare(node, env);
                    return;
                }
                const Graph &g               = *env.graph;
                const Shape &in              = g.desc(node.inputs[0]).shape;
                x_                           = NCHW::from(in);
                elem_                        = env.useFp16 ? 2 : 4;
                cbTotal_                     = cBlocks(x_.c);
                hw_                          = x_.h * x_.w;
                const auto            starts = readI64Param(g, node, "starts", 1), ends = readI64Param(g, node, "ends", 2);
                const auto            steps = readI64Param(g, node, "steps", 4);
                const SliceAxisBounds b     = resolveSliceAxis(in[1], starts[0], ends[0], steps.empty() ? 1 : steps[0]);
                blockOff_                   = b.start / kNC4Block;
                cbOut_                      = cBlocks(NCHW::from(g.desc(node.outputs[0]).shape).c);
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                if (nc4_)
                {
                    vk::Buffer *src = env.devBuf(node.inputs[0]);
                    vk::Buffer *dst = env.devBuf(node.outputs[0]);
                    // Zero-copy: the output already IS this block range of the input. Only for a single
                    // batch -- batches interleave in the source, so a multi-batch slice is not one view.
                    if (x_.n == 1 && dst->hazardRoot() == src->hazardRoot() && dst->rootOffset() == src->rootOffset() + (size_t) (blockOff_ * hw_ * kNC4Block * elem_))
                    {
                        return;
                    }
                    for (int64_t n = 0; n < x_.n; ++n)
                    {
                        VkBufferCopy c {};
                        c.srcOffset = (VkDeviceSize) ((n * cbTotal_ + blockOff_) * hw_ * kNC4Block * elem_);
                        c.dstOffset = (VkDeviceSize) ((n * cbOut_) * hw_ * kNC4Block * elem_);
                        c.size      = (VkDeviceSize) (cbOut_ * hw_ * kNC4Block * elem_);
                        vkCmdCopyBuffer(cmd, src->handle(), dst->handle(), 1, &c);
                    }
                    return;
                }
                // A full-range unit-step Slice is aliased onto its input buffer by the segment planner
                // (geometry-as-metadata); input and output then resolve to the same buffer, so the gather
                // is a no-op — skip it. Guards, in order: a folded pointwise epilogue ("pw_steps") still
                // has to run so it cannot be skipped; the data operand must exist; and an initializer
                // operand is never aliased (it lives in its own constant buffer, not the input's).
                if (!node.attr.has("pw_steps") && !node.inputs.empty() && node.inputs[0] != kNoTensor && !env.graph->isInitializer(node.inputs[0]))
                {
                    vk::Buffer *src = env.devBuf(node.inputs[0]);
                    vk::Buffer *dst = env.devBuf(node.outputs[0]);
                    if (src && dst && src->handle() == dst->handle())
                    {
                        return; // identity slice with no epilogue: aliased onto its input, dispatch is a no-op
                    }
                    // Zero-copy: a contiguous unit-step slice whose output the planner made a
                    // sub-buffer view of the input at exactly pc.base elements — bytes already in place.
                    const size_t elemBytes = env.useFp16 ? 2 : 4;
                    if (src && dst && impl.contiguousSlice && dst->hazardRoot() == src->hazardRoot() &&
                        dst->rootOffset() == src->rootOffset() + (size_t) impl.pc.base * elemBytes)
                    {
                        return;
                    }
                }
                impl.record(cmd, node, env);
            }
        };

    } // namespace

    VKNN_REGISTER_VK_OP(OpType::Transpose, TransposeOp);
    VKNN_REGISTER_VK_OP(OpType::Slice, SliceOp);

} // namespace vknn
