// Channel-axis Concat on the GPU (NC4HW4). Each input is copied into the output at its channel-
// block offset. Only valid when every input's channel count is a multiple of 4 (block alignment);
// the backend's supportsNode() enforces that, otherwise it falls back to the CPU concat. A fused
// pointwise unit (pw_steps) applies at each part's stores, indexed in output space, so a unit
// attached to the Concat (e.g. a lowered BatchNorm + activation) costs no extra dispatch. Inputs
// from pwCoreInputs on are the unit's operands, not concatenated parts.
#include "flat_ops.h"
#include "pw_plan.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        // Mirror of the concat shader's push_constant block. Cib/Cob are input/output channel-block
        // counts (channels/4 in NC4HW4), cbOff is this input's starting channel block in the output,
        // and HW is the flattened spatial extent. One invocation copies one [n][cb][hw] vec4 element.
        // Mirrors concat.comp's push_constant block: a part's blocks land at channel-block offset
        // cbOff (channel axis) or its rows/columns at (hOff, wOff) of the output plane (spatial axis).
        struct ConcatPC {
            int N, Cib, Cob, cbOff, HWin, HWout, Win, Wout, hOff, wOff;
        };
        /// The concatenated axis of a blocked Concat, normalized: 1 = channels, 2 = rows, 3 = columns.
        /// A rank-3 [N,C,L] map's axis 2 is its L, which NCHW::from places on the row axis, so the
        /// normalized value already names the NCHW axis for both ranks.
        int blockedConcatAxis(const Node &node, const Shape &out) {
            const int rank = (int) out.size();
            int64_t   axis = node.attr.geti("axis", 1);
            if (axis < 0)
            {
                axis += rank;
            }
            return (int) axis;
        }

        struct ConcatOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::vector<ConcatPC>                parts; // one per concatenated input
            std::vector<int64_t>                 partGroups;
            PwEpi                                epi;
            flat::Concat                         flatImpl;
            bool                                 flat = false;

            void prepare(const Node &node, VkOpEnv &env) override {
                if (opIsFlat(node, env))
                {
                    flat = true;
                    flatImpl.prepare(node, env);
                    return;
                }
                Shape out = env.graph->desc(node.outputs[0]).shape;
                epi.prepare(node, env, false, out);
                NCHW      y     = NCHW::from(out);
                const int axis  = blockedConcatAxis(node, out);
                int       Cob   = (int) cBlocks(y.c), HWout = (int) (y.h * y.w);
                int       cbOff = 0, hOff = 0, wOff = 0;
                size_t    nIn   = (size_t) pwCoreInputs(node);
                for (size_t e = 0; e < nIn && e < node.inputs.size(); ++e)
                {
                    NCHW xi   = NCHW::from(env.graph->desc(node.inputs[e]).shape);
                    int  Cib  = (int) cBlocks(xi.c);
                    int  HWin = (int) (xi.h * xi.w);
                    parts.push_back({(int) y.n, Cib, Cob, cbOff, HWin, HWout, (int) xi.w, (int) y.w, hOff, wOff});
                    // One invocation per [n][cb][hw] vec4; the group count and the pipeline's
                    // workgroup-size spec constant derive from the same device-resolved width.
                    partGroups.push_back(groups((int64_t) y.n * Cib * HWin, env.flatLocalSize));
                    // Advance the cursor along the concatenated axis so the next input lands after this one.
                    if (axis == 1)
                    {
                        cbOff += Cib;
                    } else if (axis == 2)
                    {
                        hOff += (int) xi.h;
                    } else
                    {
                        wOff += (int) xi.w;
                    }
                }
                pipe = env.pipeline(shader((std::string("concat") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(ConcatPC), std::vector<uint32_t> {env.flatLocalSize});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                if (flat)
                {
                    flatImpl.record(cmd, node, env);
                    return;
                }
                vk::Buffer *dst = env.devBuf(node.outputs[0]);
                // A fused unit must run at the stores, so an epi-carrying Concat never skips a part.
                const bool   mayAlias  = !node.attr.has("pw_steps");
                const size_t elemBytes = env.useFp16 ? 2 : 4;
                // Each input writes a disjoint block range (channel axis) or plane range (spatial axis) of the
                // output, so no barriers between them.
                for (size_t i = 0; i < parts.size(); ++i)
                {
                    vk::Buffer *src = env.devBuf(node.inputs[i]);
                    // Zero-copy: the planner made this part a sub-buffer view of the output at exactly
                    // its channel-block slice, so the producer already wrote the bytes in place. Valid
                    // only for the contiguous N==1 tiling the planner links.
                    const bool channelPart = parts[i].hOff == 0 && parts[i].wOff == 0 && parts[i].HWin == parts[i].HWout;
                    if (mayAlias && channelPart && parts[i].N == 1 && src->hazardRoot() == dst->hazardRoot() &&
                        src->rootOffset() == dst->rootOffset() + (size_t) parts[i].cbOff * parts[i].HWout * kNC4Block * elemBytes)
                    {
                        continue;
                    }
                    std::vector<VkBuffer> bufs {src->handle(), dst->handle()};
                    epi.append(bufs, node, env, dst->handle());
                    pipe->dispatch(cmd, bufs, &parts[i], sizeof(ConcatPC), (uint32_t) partGroups[i]);
                }
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Concat, ConcatOp);
} // namespace vknn
