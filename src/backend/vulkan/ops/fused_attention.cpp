// FusedAttention on the flat GPU path: the whole M == 1 decode-attention core — q.k^T scores with
// scale and additive mask, softmax, p.V context — in two dispatches (contract in
// core/fused_attention.h). Pass 1 (shaders/fused_attention.comp) runs one workgroup per
// (KV row, token chunk): the GQA group members share every K/V load, and the chunk's softmax
// partial {max, weight sum, unnormalized context} lands in an fp32 scratch buffer. Pass 2
// (shaders/fused_attention_combine.comp) folds the chunks with the flash rescale and stores the
// context rows. Operands are read through the per-axis strides the fuseDecodeAttention pass
// composed from the folded MatMul operand views, so the GQA KV cache is read in place. Both
// passes accumulate in fp32, so the node matches the CPU oracle to fp32 rounding (cosine), not
// byte-for-byte.
//
// Pass 1 has two interchangeable kernels: the portable base kernel (fused_attention.comp) and a
// subgroup variant (fused_attention_sg.comp) that replaces the shared-memory reduction trees with
// subgroupMax/subgroupAdd, reads each V element once for the whole GQA group, and reads
// element-contiguous 4-aligned K/V rows as vec4 words. prepare() picks the subgroup kernel when
// the device and the node meet its requirements (see sgEligible below); the base kernel and its
// dispatch are byte-unchanged otherwise.
#include "core/fused_attention.h"
#include "flat_ops.h"
#include "vk_op_common.h"
#include "vknn/op.h"
#include <vector>

namespace vknn {
    namespace {

        // Field order/types mirror fused_attention.comp's push_constant block.
        struct FaPartialPC {
            int   rank, kvRows, C, hd;
            int   qK, kN, kK, vN, vK, mN, hasMask, chunks;
            int   groupStrideQ, groupStrideM, groupStrideRow;
            int   pastLen, kNewN, kNewK, vNewN, vNewK;
            float scale, maskScale;
        };
        // Mirrors fused_attention_combine.comp's push_constant block.
        struct FaCombinePC {
            int rows, hd, chunks;
        };

        constexpr int kFaWorkgroupSize = 256;
        constexpr int kFaMaxChunks     = 64; // == FA_MAX_CHUNKS in fused_attention_combine.comp
        // Subgroup-kernel tile: token chunk and workgroup-width target (the width widens to the
        // head dim and rounds up to whole subgroups). A decode model exposes only kv-head-count KV
        // rows, so parallelism comes from the chunk count: the half-width chunk doubles the
        // (kvRows * chunks) grid over the base kernel's, and the full-width workgroup keeps every
        // lane busy in the p.V sweep (measured best among chunk 64..256 x width 64..256 tiles on
        // the reference decode; the differences past this pick are a few percent).
        constexpr int kFaSgChunkTokens    = 128;
        constexpr int kFaSgWorkgroupSize  = 256;

        struct FusedAttentionOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> partialPipe;
            std::shared_ptr<vk::ComputePipeline> combinePipe;
            bool                                 useSgKernel = false; // pass 1 runs fused_attention_sg.comp
            // int8 KV cache (Hint::KvCacheQuant): the segment stored this node's past K/V sources
            // as int8 payload + fp16 per-(head, token)-row scales, so pass 1 runs the _kvq kernel
            // twin that dequantizes them inside the fp32 K-dot / V-apply loops. Derived from the
            // segment's allocation (env.kvqScale), never re-decided here — kernel and buffers
            // cannot disagree. The scale buffers are segment-owned; the raw pointers stay valid
            // for this op instance's lifetime.
            bool                                 useKvq = false;
            vk::Buffer                          *kvqKScale = nullptr, *kvqVScale = nullptr;
            FaPartialPC                          partialPc {};
            FaCombinePC                          combinePc {};
            std::shared_ptr<vk::Buffer>          geom;      // dims + strides + row strides, deduped SSBO
            std::shared_ptr<vk::Buffer>          scratch;   // fp32 chunk partials, rows*chunks*(hd+2)
            std::shared_ptr<vk::Buffer>          hold[6];   // per-operand, set when that operand is a constant initializer
            std::shared_ptr<vk::Buffer>          maskDummy; // bound in the mask slot when the node has no mask

            void prepare(const Node &node, VkOpEnv &env) override {
                const std::vector<int64_t> &dims    = node.attr.getints(kFaDims);
                const std::vector<int64_t> &qStride = node.attr.getints(kFaQStride);
                const std::vector<int64_t> &kStride = node.attr.getints(kFaKStride);
                const std::vector<int64_t> &vStride = node.attr.getints(kFaVStride);
                const std::vector<int64_t> &mStride = node.attr.getints(kFaMStride);
                const int                   rank    = (int) dims.size();

                partialPc.rank    = rank;
                partialPc.C       = (int) node.attr.geti(kFaC);
                partialPc.hd      = (int) node.attr.geti(kFaHd);
                partialPc.qK      = (int) node.attr.geti(kFaQK);
                partialPc.kN      = (int) node.attr.geti(kFaKN);
                partialPc.kK      = (int) node.attr.geti(kFaKK);
                partialPc.vN      = (int) node.attr.geti(kFaVN);
                partialPc.vK      = (int) node.attr.geti(kFaVK);
                partialPc.mN      = (int) node.attr.geti(kFaMN);
                partialPc.hasMask = node.inputs.size() > 3 && node.inputs[3] != kNoTensor ? 1 : 0;
                // Split-KV form: token s >= pastLen reads the new-rows sources (inputs 4/5) through
                // their own strides; the unsplit form sets pastLen = C so the branch never fires.
                const bool split   = node.attr.geti(kFaSplit, 0) != 0 && node.inputs.size() >= 6;
                kvqKScale          = split && env.kvqScale ? env.kvqScale(node.inputs[1]) : nullptr;
                kvqVScale          = split && env.kvqScale ? env.kvqScale(node.inputs[2]) : nullptr;
                if ((kvqKScale != nullptr) != (kvqVScale != nullptr))
                {
                    // The shared eligibility rule admits K and V together or not at all; one-sided
                    // allocation means the segment and this op disagree — a wiring bug, not a state
                    // to run through.
                    throw Error(Status::RuntimeError, "FusedAttention '" + node.name + "': int8 KV cache allocated for only one of the past K/V sources");
                }
                useKvq = kvqKScale != nullptr;
                partialPc.pastLen  = split ? (int) node.attr.geti(kFaPastLen) : partialPc.C;
                partialPc.kNewN    = (int) node.attr.geti(kFaKNewN);
                partialPc.kNewK    = (int) node.attr.geti(kFaKNewK);
                partialPc.vNewN    = (int) node.attr.geti(kFaVNewN);
                partialPc.vNewK    = (int) node.attr.geti(kFaVNewK);
                const float scale = node.attr.getf(kFaScale, 1.f);
                const float mask  = node.attr.getf(kFaMaskScale, 1.f);
                partialPc.scale     = scale;
                partialPc.maskScale = mask;

                int64_t rows = 1;
                for (int64_t d: dims)
                {
                    rows *= d;
                }

                // GQA group axis: K and V do not advance along it (zero stride), so its members
                // share one K/V row set and pass 1 packs them into one workgroup. The first such
                // axis is taken; without one the group is a single member.
                int groupAxis = -1;
                for (int i = 0; i < rank; ++i)
                {
                    if (dims[i] > 1 && kStride[i] == 0 && vStride[i] == 0)
                    {
                        groupAxis = i;
                        break;
                    }
                }
                const int64_t groupSize = groupAxis >= 0 ? dims[groupAxis] : 1;

                // Attention-row strides (row-major over the ORIGINAL dims) recover the flat output
                // row a (KV row, group member) pair addresses.
                std::vector<int64_t> rowStride(rank, 1);
                for (int i = rank - 2; i >= 0; --i)
                {
                    rowStride[(size_t) i] = rowStride[(size_t) i + 1] * dims[(size_t) i + 1];
                }
                partialPc.groupStrideQ   = groupAxis >= 0 ? (int) qStride[(size_t) groupAxis] : 0;
                partialPc.groupStrideM   = groupAxis >= 0 && partialPc.hasMask && groupAxis < (int) mStride.size() ? (int) mStride[(size_t) groupAxis] : 0;
                partialPc.groupStrideRow = groupAxis >= 0 ? (int) rowStride[(size_t) groupAxis] : 0;
                partialPc.kvRows         = (int) (rows / groupSize);

                // Subgroup-kernel eligibility (fused_attention_sg.comp; any miss keeps the base
                // kernel, whose pipeline and dispatch are unchanged): subgroup arithmetic +
                // shuffle, a head dim that is a multiple of 4, a group small enough for the
                // per-lane accumulator arrays, and a workgroup width — chosen against the device's
                // shared-memory budget below — of whole subgroups within the invocation limit that
                // covers the head dim and holds the G * subgroup-count reduction slab.
                const auto   &cap          = env.ctx->caps();
                const int64_t subgroupSize = (int64_t) cap.subgroupSize;
                bool          sgEligible   = cap.subgroupArithmetic && cap.subgroupShuffle && subgroupSize > 0 && partialPc.hd % 4 == 0 && groupSize <= 8;

                // The sg token chunk, independent of the workgroup width: the G*CHUNK score slab
                // stays small enough that several workgroups sit resident per core, and the chunk
                // count scales the (kvRows * chunks) dispatch to the context so a low-KV-row model
                // still fills the GPU — a decode model has only kv-head-count KV rows, so the grid
                // IS the chunk count times that. The combine pass caps the chunk count.
                const int64_t sgChunkTokens = [&] {
                    int64_t tokens = std::max<int64_t>(std::min<int64_t>(4096 / groupSize, kFaSgChunkTokens), 1);
                    int64_t count  = (partialPc.C + tokens - 1) / tokens;
                    return count > kFaMaxChunks ? (partialPc.C + kFaMaxChunks - 1) / kFaMaxChunks : tokens;
                }();

                // The sg workgroup width adapts to the reported shared-memory budget: the widest
                // fitting width from the tile target down to the head-dim floor, halving per step
                // (each candidate rounded up to whole subgroups), where fitting means the shared
                // arrays — sQ4 (G*hd floats as vec4s) + sScores (G*chunk) + sRed (width) + the
                // vec4 sFold (G*width vec4s) — stay within cap.maxSharedMemory. A device with a
                // smaller budget runs a narrower workgroup instead of losing the kernel; only when
                // the narrowest legal width still misses the budget does the node keep the base
                // kernel. The step halves rather than walking subgroup-sized decrements: on the
                // budget-constrained tier the in-between width (an odd subgroup count) measured
                // ~35% slower than the next halved width.
                int64_t wgs = 0;
                if (sgEligible)
                {
                    const int64_t widest    = (std::max<int64_t>(kFaSgWorkgroupSize, partialPc.hd) + subgroupSize - 1) / subgroupSize * subgroupSize;
                    const int64_t narrowest = (std::max<int64_t>(partialPc.hd, subgroupSize) + subgroupSize - 1) / subgroupSize * subgroupSize;
                    for (int64_t width = widest; width >= narrowest;)
                    {
                        const int64_t sharedBytes = (groupSize * partialPc.hd + groupSize * sgChunkTokens + width) * 4 + groupSize * width * 16;
                        if (width <= (int64_t) cap.maxWorkGroupInvocations && groupSize * (width / subgroupSize) <= width && sharedBytes <= (int64_t) cap.maxSharedMemory)
                        {
                            wgs = width;
                            break;
                        }
                        if (width <= narrowest)
                        {
                            break;
                        }
                        const int64_t halved = (width / 2 + subgroupSize - 1) / subgroupSize * subgroupSize;
                        width                = std::max<int64_t>(std::min<int64_t>(halved, width - subgroupSize), narrowest);
                    }
                    sgEligible = wgs > 0;
                }

                // Chunking: the sg chunk from above, or the base kernel's workgroup-size chunk.
                int64_t chunkTokens = sgChunkTokens;
                if (!sgEligible)
                {
                    chunkTokens = std::max<int64_t>(std::min<int64_t>(4096 / groupSize, kFaWorkgroupSize), 1);
                    if ((partialPc.C + chunkTokens - 1) / chunkTokens > kFaMaxChunks)
                    {
                        chunkTokens = (partialPc.C + kFaMaxChunks - 1) / kFaMaxChunks;
                    }
                }
                int64_t chunks = (partialPc.C + chunkTokens - 1) / chunkTokens;
                partialPc.chunks = (int) chunks;
                combinePc        = {(int) rows, partialPc.hd, (int) chunks};

                // Geometry SSBO: the KV-row dims (group axis collapsed to 1) plus the four operand
                // stride arrays and the row strides.
                const std::vector<int64_t> &kNewStride = node.attr.getints(kFaKNewStride);
                const std::vector<int64_t> &vNewStride = node.attr.getints(kFaVNewStride);
                std::vector<int32_t> dimsKv(rank), qs(rank), ks(rank), vs(rank), ms(rank, 0), rs(rank), kn2(rank, 0), vn2(rank, 0);
                for (int i = 0; i < rank; ++i)
                {
                    dimsKv[i] = (int32_t) (i == groupAxis ? 1 : dims[(size_t) i]);
                    qs[i]     = (int32_t) qStride[(size_t) i];
                    ks[i]     = (int32_t) kStride[(size_t) i];
                    vs[i]     = (int32_t) vStride[(size_t) i];
                    rs[i]     = (int32_t) rowStride[(size_t) i];
                    if (partialPc.hasMask && i < (int) mStride.size())
                    {
                        ms[i] = (int32_t) mStride[(size_t) i];
                    }
                    if (split && i < (int) kNewStride.size())
                    {
                        kn2[i] = (int32_t) kNewStride[(size_t) i];
                        vn2[i] = (int32_t) vNewStride[(size_t) i];
                    }
                }
                geom = flat::uploadFlatGeom(env, {dimsKv, qs, ks, vs, ms, rs, kn2, vn2});

                // fp32 chunk partials: {m, l, acc[hd]} per (row, chunk).
                scratch = std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>((size_t) rows * (size_t) chunks * (size_t) (partialPc.hd + 2) * sizeof(float), 16), vk::MemPref::kDeviceOnly);

                if (!partialPc.hasMask || !split)
                {
                    // Descriptor sets bind every declared buffer; a maskless node binds a shared
                    // dummy word in the mask slot and an unsplit node binds it in the new-source
                    // slots — paths the kernel never reads.
                    maskDummy = env.acquireWeight("fattn#dummy", env.useFp16, [&] {
                        const uint32_t zero[4] = {0, 0, 0, 0};
                        return env.uploadWeightDeviceOnly(zero, sizeof zero, sizeof zero);
                    });
                }

                // vec4 K/V fast paths: the row must be element-contiguous along the vectorized
                // axis and every base-offset contribution a multiple of 4 elements, so each vec4
                // word sits on a 4-element boundary of the buffer.
                const auto strides4Aligned = [](const std::vector<int32_t> &strides) {
                    for (int32_t s: strides)
                    {
                        if (s % 4 != 0)
                        {
                            return false;
                        }
                    }
                    return true;
                };
                bool kVec4 = partialPc.kK == 1 && partialPc.kN % 4 == 0 && strides4Aligned(ks);
                bool vVec4 = partialPc.vN == 1 && partialPc.vK % 4 == 0 && strides4Aligned(vs);
                if (split)
                {
                    kVec4 = kVec4 && partialPc.kNewK == 1 && partialPc.kNewN % 4 == 0 && strides4Aligned(kn2);
                    vVec4 = vVec4 && partialPc.vNewN == 1 && partialPc.vNewK % 4 == 0 && strides4Aligned(vn2);
                }

                useSgKernel = sgEligible;
                // Kernel-choice record under Config::verbosity Debug: which pass-1 kernel the node
                // runs and whether the vec4 K/V paths engaged.
                VKNN_DEBUG << "FusedAttention pass 1: " << (sgEligible ? "subgroup" : "base") << (useKvq ? " kvq" : "") << " kernel, kv4=" << kVec4 << " vv4=" << vVec4 << " group=" << groupSize
                           << " hd=" << partialPc.hd << " chunk=" << chunkTokens << " wgs=" << wgs << " kvRows=" << partialPc.kvRows;
                if (useSgKernel)
                {
                    // Spec constants: G, HD, CHUNK, the shared-array products QT4 == G*HD/4,
                    // SCORETOTAL == G*CHUNK and FOLDTOTAL == G*WGS (a spec-constant product cannot
                    // size an array in GLSL, so the host computes them), the workgroup width, and
                    // the vec4 path selectors. The kvq twin shares the whole scheme and appends the
                    // two scale bindings (12/13).
                    const std::vector<uint32_t> spec = {(uint32_t) groupSize,           (uint32_t) partialPc.hd, (uint32_t) chunkTokens, (uint32_t) (groupSize * partialPc.hd / 4),
                                                        (uint32_t) (groupSize * chunkTokens), (uint32_t) wgs,          kVec4 ? 1u : 0u,        vVec4 ? 1u : 0u,
                                                        (uint32_t) (groupSize * wgs)};
                    partialPipe                      = env.pipeline(shader(useKvq ? "fused_attention_sg_kvq" : "fused_attention_sg", env.useFp16), useKvq ? 14u : 12u, sizeof(FaPartialPC), spec);
                } else
                {
                    // Spec constants: G, HD, CHUNK plus the two shared-array products (a spec-constant
                    // product cannot size an array in GLSL, so the host computes them). The kvq twin
                    // shares the scheme and appends the two scale bindings (8/9).
                    const std::vector<uint32_t> spec = {(uint32_t) groupSize, (uint32_t) partialPc.hd, (uint32_t) chunkTokens,
                                                        (uint32_t) (groupSize * partialPc.hd), (uint32_t) (groupSize * chunkTokens)};
                    partialPipe                      = env.pipeline(shader(useKvq ? "fused_attention_kvq" : "fused_attention", env.useFp16), useKvq ? 10u : 8u, sizeof(FaPartialPC), spec);
                }
                combinePipe = env.pipeline(shader("fused_attention_combine", env.useFp16), 2, sizeof(FaCombinePC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                const bool  split = partialPc.pastLen < partialPc.C;
                vk::Buffer *q    = operandBuf(env, node.inputs[0], hold[0]);
                vk::Buffer *k    = operandBuf(env, node.inputs[1], hold[1]);
                vk::Buffer *v    = operandBuf(env, node.inputs[2], hold[2]);
                vk::Buffer *mask = partialPc.hasMask ? operandBuf(env, node.inputs[3], hold[3]) : maskDummy.get();
                vk::Buffer *kNew = split ? operandBuf(env, node.inputs[4], hold[4]) : maskDummy.get();
                vk::Buffer *vNew = split ? operandBuf(env, node.inputs[5], hold[5]) : maskDummy.get();
                vk::Buffer *dst  = env.devBuf(node.outputs[0]);
                // Pass 1: one workgroup per (KV row, chunk); the 1-D grid spills into y and the
                // kernel folds it back through gl_WorkGroupID.y. The subgroup kernel re-binds the
                // K/V (and split new-source) buffers at 8..11 as its vec4 views; a kvq node
                // appends the two per-row scale buffers as the last bindings of either kernel.
                std::vector<VkBuffer> bindings;
                if (useSgKernel)
                {
                    bindings = {q->handle(), k->handle(), v->handle(), scratch->handle(), mask->handle(), geom->handle(), kNew->handle(), vNew->handle(), k->handle(), v->handle(),
                                kNew->handle(), vNew->handle()};
                } else
                {
                    bindings = {q->handle(), k->handle(), v->handle(), scratch->handle(), mask->handle(), geom->handle(), kNew->handle(), vNew->handle()};
                }
                if (useKvq)
                {
                    bindings.push_back(kvqKScale->handle());
                    bindings.push_back(kvqVScale->handle());
                }
                partialPipe->dispatch(cmd, bindings, &partialPc, sizeof(partialPc), (uint32_t) ((int64_t) partialPc.kvRows * partialPc.chunks));
                // Two dispatches in one record() are NOT auto-barriered; the combine reads the
                // scratch pass 1 wrote (same pattern as reduce.cpp).
                vk::computeBarrier(*env.ctx, cmd);
                combinePipe->dispatch(cmd, {scratch->handle(), dst->handle()}, &combinePc, sizeof(combinePc), (uint32_t) combinePc.rows);
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::FusedAttention, FusedAttentionOp);
} // namespace vknn
