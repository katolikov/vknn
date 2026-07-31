// Split on the GPU, two paths:
//  - NC4HW4 channel split (4D, axis=1, every output's channels a multiple of 4): each output is a
//    contiguous range of channel-blocks, a plain buffer copy (keeps YOLO's C2f blocks on the GPU).
//  - Flat row-major split (any other axis, e.g. the encoder's last-axis splits): each output is a
//    Slice along the split axis, dispatched via the flat_gather shader (base = axis offset).
#include "flat_ops.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct SplitOp: VulkanOp {
            bool flat_ = false;
            // ---- NC4HW4 channel path ----
            struct Part {
                int     outIdx;
                int64_t blockOff, cbk;
            };
            std::vector<Part> parts_;
            NCHW              x_ {};
            int               elem_    = 4;
            int64_t           cbTotal_ = 0, hw_ = 0;
            // ---- flat path ----
            // The flat path reuses flat::Gather's push-constant block verbatim rather than mirroring
            // it: the block is the flat_gather.comp contract, and a private copy would silently
            // under-declare the pipeline's push range whenever that contract grows. The per-output
            // outDim/inStride geometry rides a content-deduped SSBO (flat::uploadFlatGeom, binding
            // 2), one per output. A split output is never virtualized, so outPad/outLast stay 0.
            using FPC = flat::Gather::PC;
            std::vector<FPC>                                  fpcs_;
            std::vector<int>                                  foutIdx_;
            std::vector<std::shared_ptr<vk::ComputePipeline>> fpipes_;
            std::vector<std::shared_ptr<vk::Buffer>>          fgeom_;
            std::shared_ptr<vk::Buffer>                       hold0_;
            bool contiguousParts_ = false; // every dim before the axis is 1: outputs are contiguous slabs at pc.base

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g = *env.graph;
                flat_          = !node.outputs.empty() && node.outputs[0] != kNoTensor && g.desc(node.outputs[0]).gpuFlat;
                if (flat_)
                {
                    Shape in   = g.desc(node.inputs[0]).shape;
                    int   rank = (int) in.size();
                    int   axis = (int) node.attr.geti("axis", 0);
                    if (axis < 0)
                    {
                        axis += rank;
                    }
                    {
                        int64_t outer = 1;
                        for (int d = 0; d < axis && d < rank; ++d)
                        {
                            outer *= in[d];
                        }
                        contiguousParts_ = outer == 1;
                    }
                    auto    inStride = flat::rowStrides(in);
                    int64_t offset   = 0; // running start of this output along the split axis, in axis elements
                    for (size_t k = 0; k < node.outputs.size(); ++k)
                    {
                        if (node.outputs[k] == kNoTensor)
                        {
                            continue;
                        }
                        Shape out = g.desc(node.outputs[k]).shape;
                        FPC   pc {};
                        pc.rank  = rank;
                        pc.total = (int) numElements(out);
                        // Each output is a contiguous slice of the input along the split axis. flat_gather reads
                        // in[base + sum_d outCoord_d * inStride_d]; base skips this output's axis offset (offset
                        // rows of the axis stride), and the input strides carry every other axis through unchanged.
                        pc.base = (int) (offset * inStride[axis]);
                        std::vector<int32_t> outDim(rank), inStr(rank);
                        for (int d = 0; d < rank; ++d)
                        {
                            outDim[d] = (int) out[d];
                            inStr[d]  = (int) inStride[d];
                        }
                        fgeom_.push_back(flat::uploadFlatGeom(env, {outDim, inStr}));
                        pc.items = flat::kMovementItemsPerLane; // one element per lane (see flat_ops.h)
                        fpcs_.push_back(pc);
                        foutIdx_.push_back((int) k);
                        offset += out[axis]; // next output starts where this one ends
                        fpipes_.push_back(
                            env.pipeline(shader("flat_gather", env.useFp16), 3, sizeof(FPC), std::vector<uint32_t> {env.flatLocalSize, (uint32_t) pc.items}));
                    }
                    return;
                }
                // NC4HW4 channel split
                x_       = NCHW::from(g.desc(node.inputs[0]).shape);
                elem_    = env.useFp16 ? 2 : 4; // bytes per stored element (fp16 half vs fp32 float)
                cbTotal_ = cBlocks(x_.c);       // input's channel-block count (the source row pitch)
                hw_      = x_.h * x_.w;
                // Each output owns a contiguous run of channel-blocks. blk is the input's first block index
                // for the current output; because every split channel count is a multiple of four here, a
                // block is never shared, so each output is a whole-block copy with no channel remainder.
                int64_t blk = 0;
                for (size_t k = 0; k < node.outputs.size(); ++k)
                {
                    if (node.outputs[k] == kNoTensor)
                    {
                        continue;
                    }
                    int64_t ck  = NCHW::from(g.desc(node.outputs[k]).shape).c;
                    int64_t cbk = cBlocks(ck); // channel-blocks this output takes
                    parts_.push_back({(int) k, blk, cbk});
                    blk += cbk;
                }
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                const size_t elemBytes = env.useFp16 ? 2 : 4;
                if (flat_)
                {
                    vk::Buffer *src = operandBuf(env, node.inputs[0], hold0_);
                    for (size_t i = 0; i < fpcs_.size(); ++i)
                    {
                        vk::Buffer *dst = env.devBuf(node.outputs[foutIdx_[i]]);
                        // Zero-copy: the planner made this output a sub-buffer view of the input at
                        // exactly its slab offset (pc.base elements) — the bytes are already in place.
                        if (contiguousParts_ && dst->hazardRoot() == src->hazardRoot() && dst->rootOffset() == src->rootOffset() + (size_t) fpcs_[i].base * elemBytes)
                        {
                            continue;
                        }
                        // One flat_gather invocation per output element (local_size_x = kFlatLocalSize),
                        // gathering the slice for this output straight out of the shared input buffer.
                        fpipes_[i]->dispatch(cmd, {src->handle(), dst->handle(), fgeom_[i]->handle()}, &fpcs_[i], sizeof(FPC),
                                             groups((fpcs_[i].total + fpcs_[i].items - 1) / fpcs_[i].items, env.flatLocalSize));
                    }
                    return;
                }
                // NC4HW4 buffer bytes are laid out as [n][channelBlock][h*w][kNC4Block] * elem_. One
                // vkCmdCopyBuffer per batch lifts this output's block range out of the input: batches are
                // interleaved in the source (stride cbTotal_ blocks) but packed contiguously in the compact
                // destination (stride p.cbk blocks), so the copies cannot be coalesced across n.
                vk::Buffer *src = env.devBuf(node.inputs[0]);
                for (const Part &p: parts_)
                {
                    vk::Buffer *dst = env.devBuf(node.outputs[p.outIdx]);
                    // Zero-copy: this output is a view of the input's channel-block slice (N==1 only —
                    // batched sources interleave and cannot be a single view).
                    if (x_.n == 1 && dst->hazardRoot() == src->hazardRoot() && dst->rootOffset() == src->rootOffset() + (size_t) p.blockOff * hw_ * kNC4Block * elemBytes)
                    {
                        continue;
                    }
                    for (int64_t n = 0; n < x_.n; ++n)
                    {
                        VkBufferCopy c {};
                        // Skip n full input batches plus p.blockOff blocks to this output's first block.
                        c.srcOffset = (VkDeviceSize) ((n * cbTotal_ + p.blockOff) * hw_ * kNC4Block * elem_);
                        c.dstOffset = (VkDeviceSize) ((n * p.cbk) * hw_ * kNC4Block * elem_);
                        c.size      = (VkDeviceSize) (p.cbk * hw_ * kNC4Block * elem_);
                        vkCmdCopyBuffer(cmd, src->handle(), dst->handle(), 1, &c);
                    }
                }
            }
        };

    } // namespace

    VKNN_REGISTER_VK_OP(OpType::Split, SplitOp);

} // namespace vknn
