// Generic flat (row-major) GPU op implementations for the detection-head graph: Transpose, Slice,
// Concat, broadcasting Binary/Add, non-channel Softmax. The layout pass (insertLayoutConverts)
// stores these tensors as flat buffers and inserts ConvertLayout at the NC4HW4 boundary, so a whole
// YOLO head runs on the GPU. Ops with an existing NC4HW4 kernel (concat/binary/add/softmax) hold
// one of these and dispatch to it when their tensors are flat; Transpose/Slice are flat-only.
#pragma once
#include "backend/vulkan/vk_tune_model.h"
#include "core/slice_bounds.h"
#include "import/passes.h" // readI64Param
#include "pw_plan.h"
#include "vk_op_common.h"

namespace vknn {

    // A node runs the flat path iff the layout pass marked its (first) output as a flat GPU tensor.
    inline bool opIsFlat(const Node &node, VkOpEnv &env) {
        return !node.outputs.empty() && node.outputs[0] != kNoTensor && env.graph->desc(node.outputs[0]).gpuFlat;
    }

    namespace flat {

        // 1D dispatch local size CEILING for the element-parallel flat shaders (gather/pad/broadcast/
        // scatter/binary/converts): the family's shaders declare local_size_x = 256 as the default
        // AND expose it as their workgroup-size spec constant, so the value that actually runs is
        // env.flatLocalSize - resolved once at load by flatLocalSizeFor() below and passed to every
        // family pipeline. 256 remains the ceiling (and the shared-array size of the workgroup-
        // reduction kernels); a device whose caps cannot host it gets the largest whole-subgroup
        // width that fits instead of a pipeline-creation failure.
        constexpr uint32_t kFlatLocalSize = 256;

        // A family workgroup width for this device, from EXACT caps (never a measured probe -
        // see VkOpEnv::flatLocalSize): the family's ceiling clamped to maxWorkGroupInvocations and
        // maxWorkGroupSize[0], rounded down to whole subgroups, at least one subgroup.
        inline uint32_t laneWidthFor(const vk::VulkanCaps &caps, uint32_t ceiling) {
            uint32_t width = ceiling;
            if (caps.maxWorkGroupInvocations != 0u && caps.maxWorkGroupInvocations < width)
            {
                width = caps.maxWorkGroupInvocations;
            }
            if (caps.maxWorkGroupSize[0] != 0u && caps.maxWorkGroupSize[0] < width)
            {
                width = caps.maxWorkGroupSize[0];
            }
            const uint32_t sub = caps.subgroupSize != 0u ? caps.subgroupSize : 64u;
            width              = width / sub * sub;
            return width != 0u ? width : sub;
        }
        inline uint32_t flatLocalSizeFor(const vk::VulkanCaps &caps) {
            return laneWidthFor(caps, kFlatLocalSize);
        }

        // Lane-width ceiling of the per-thread conv/sampler family (VkOpEnv::convLocalSize).
        constexpr uint32_t kConvFamilyLaneWidth = 64;

        // Elements each lane walks, one slot apart, for the element-parallel family (the fused_pw
        // pattern extended to the movement kernels). A pure placement choice - identical values at
        // any count - so it may consume the MEASURED device probe: the floor is the saturation
        // point deviceTuneModel reports (64-wide waves before added waves stop buying throughput)
        // times a fixed headroom multiple, in lanes. Resolved at load (prepare), never at record.
        constexpr int kItemsPerLaneMax       = 8; // independent loads one lane keeps in flight
        constexpr int kLaneFloorWaveHeadroom = 5;
        constexpr int kProbeWaveLanes        = 64; // the probe counts 64-wide waves
        inline int    itemsPerLane(int64_t total, VkOpEnv &env) {
            const double  waves     = vk::deviceTuneModel(env).wavesToSaturate;
            const int64_t laneFloor = (int64_t) (waves * kProbeWaveLanes) * kLaneFloorWaveHeadroom;
            const int64_t byFloor   = laneFloor > 0 ? total / laneFloor : 1;
            return byFloor < 1 ? 1 : (byFloor > kItemsPerLaneMax ? (int) kItemsPerLaneMax : (int) byFloor);
        }
        inline std::vector<int64_t> rowStrides(const Shape &s) {
            std::vector<int64_t> st(s.size(), 1);
            for (int k = (int) s.size() - 2; k >= 0; --k)
            {
                st[k] = st[k + 1] * s[k + 1];
            }
            return st;
        }

        // Rank geometry (the per-axis outDim/inStride/aStride/... arrays a flat kernel walks) lives in a
        // content-deduped SSBO, not the push constant: a push-constant block sized for the largest rank
        // would blow the 128-byte Vulkan floor (a 4-array op at rank 8 is already 144 B), so the flat
        // kernels read geometry from a device buffer and the push constant carries only scalars. This
        // makes the decodable activation rank unbounded.
        //
        // The arrays are packed back-to-back in the order given: array a's element k is g[a*rank + k], so
        // a kernel reads outDim as g[0..rank), the next array as g[rank..2*rank), and so on. uploadPooled
        // dedupes byte-identical geometry (repeated transformer layers share one allocation).
        inline std::shared_ptr<vk::Buffer> uploadFlatGeom(VkOpEnv &env, const std::vector<std::vector<int32_t>> &arrays) {
            std::vector<int32_t> packed;
            for (const auto &a: arrays)
            {
                packed.insert(packed.end(), a.begin(), a.end());
            }
            if (packed.empty())
            {
                packed.push_back(0); // uploadPooled floors nothing; keep a non-empty payload for a rank-0 op
            }
            return env.uploadPooled(packed.data(), packed.size() * sizeof(int32_t));
        }

        // ---- Transpose / Slice: out[i] = in[base + sum outCoord_k * inStride_k] ----
        struct Gather {
            // outPad/outLast are the output's physical and logical last-axis extents, both 0 unless
            // the segment allocated the output as a virtualized activation (VkOpEnv::rowPad).
            struct PC {
                int rank, total, base, outPad, outLast, items = 1;
            } pc {};
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          geom;  // outDim/inStride, deduped SSBO (binding 2, before the epilogue)
            std::shared_ptr<vk::Buffer>          hold0; // when input[0] is a constant initializer
            PwEpi                                epi;   // a pointwise chain folded into the gather's store
            // Slice only: the sliced box is a contiguous flat sub-range of the input (unit steps;
            // leading dims select one index, at most one partial axis, trailing dims full), so the
            // gather reads exactly in[base .. base+total). SliceOp::record may then skip the dispatch
            // when the output buffer is a sub-buffer view of the input at that byte offset.
            bool contiguousSlice = false;
            void prepare(const Node &node, VkOpEnv &env) {
                const Graph &g  = *env.graph;
                Shape        in = g.desc(node.inputs[0]).shape, out = g.desc(node.outputs[0]).shape;
                auto         inStride = rowStrides(in);
                int          rank     = (int) out.size();
                pc.rank               = rank;
                pc.total              = (int) numElements(out);
                pc.items              = itemsPerLane(pc.total, env);
                pc.base               = 0;
                std::vector<int32_t> outDim(rank), inStr(rank);
                // A virtualized output (the segment allocated its buffer with a padded physical last
                // axis so a consuming tiled MatMul can take the vec4-load kernels): the grid covers
                // the PHYSICAL element count and the kernel zero-fills the pad columns. The
                // zero-copy slice shortcut cannot apply — the data bytes are no longer contiguous.
                auto applyOutPad = [&] {
                    const int64_t outPad = env.rowPad ? env.rowPad(node.outputs[0]) : 0;
                    if (outPad <= 0 || out.empty())
                    {
                        return;
                    }
                    pc.outLast      = (int) out.back();
                    pc.outPad       = (int) outPad;
                    pc.total        = (int) (numElements(out) / out.back() * outPad);
                    pc.items        = itemsPerLane(pc.total, env);
                    contiguousSlice = false;
                };
                // A folded movement chain (foldMovementChains) carries its composed per-axis map in
                // view_stride/view_base — the gather geometry verbatim, overriding perm/starts.
                if (node.attr.has("view_stride"))
                {
                    const auto &vs = node.attr.getints("view_stride");
                    if ((int) vs.size() == rank)
                    {
                        for (int k = 0; k < rank; ++k)
                        {
                            outDim[k] = (int) out[k];
                            inStr[k]  = (int) vs[(size_t) k];
                        }
                        pc.base = (int) node.attr.geti("view_base", 0);
                        applyOutPad();
                        geom = uploadFlatGeom(env, {outDim, inStr});
                        epi.prepare(node, env, /*flat=*/true, g.desc(node.outputs[0]).shape);
                        pipe = env.pipeline(shader((std::string("flat_gather") + epi.suffix()).c_str(), env.useFp16), 3 + epi.extraBufs(), sizeof(PC), std::vector<uint32_t> {env.flatLocalSize});
                        return;
                    }
                }
                if (node.type == OpType::Transpose)
                {
                    const auto &perm = node.attr.getints("perm");
                    for (int k = 0; k < rank; ++k)
                    {
                        int p     = perm.empty() ? rank - 1 - k : (int) perm[k];
                        outDim[k] = (int) in[p];
                        inStr[k]  = (int) inStride[p];
                    }
                } else
                { // kSlice
                    int                  r      = (int) in.size();
                    auto                 starts = readI64Param(g, node, "starts", 1), ends = readI64Param(g, node, "ends", 2);
                    auto                 axes = readI64Param(g, node, "axes", 3), steps = readI64Param(g, node, "steps", 4);
                    std::vector<int64_t> start(r, 0), step(r, 1);
                    for (size_t a = 0; a < starts.size() && a < ends.size(); ++a)
                    {
                        int ax = axes.empty() ? (int) a : (int) (axes[a] < 0 ? axes[a] + r : axes[a]);
                        if (ax < 0 || ax >= r)
                        {
                            continue;
                        }
                        // The same ONNX bound rules shape inference used, so `base` lands on the first
                        // element the inferred `out` extent expects — including a reverse slice, whose
                        // signed stride below then walks back down the axis.
                        const int64_t sp = steps.size() > a ? steps[a] : 1;
                        start[ax]        = resolveSliceAxis(in[ax], starts[a], ends[a], sp).start;
                        step[ax]         = sp;
                    }
                    for (int k = 0; k < rank; ++k)
                    {
                        outDim[k] = (int) out[k];
                        inStr[k]  = (int) (inStride[k] * step[k]);
                        pc.base += (int) (start[k] * inStride[k]);
                    }
                    {
                        int partial     = -1;
                        contiguousSlice = rank == r;
                        for (int d = 0; contiguousSlice && d < r; ++d)
                        {
                            if (step[d] != 1)
                            {
                                contiguousSlice = false;
                            } else if (start[d] != 0 || out[d] != in[d])
                            { partial = d; }
                        }
                        for (int d = 0; contiguousSlice && partial >= 0 && d < r; ++d)
                        {
                            if ((d < partial && out[d] != 1) || (d > partial && (start[d] != 0 || out[d] != in[d])))
                            {
                                contiguousSlice = false;
                            }
                        }
                    }
                }
                applyOutPad();
                geom = uploadFlatGeom(env, {outDim, inStr});
                epi.prepare(node, env, /*flat=*/true, g.desc(node.outputs[0]).shape);
                pipe = env.pipeline(shader((std::string("flat_gather") + epi.suffix()).c_str(), env.useFp16), 3 + epi.extraBufs(), sizeof(PC), std::vector<uint32_t> {env.flatLocalSize});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) {
                vk::Buffer           *dst  = env.devBuf(node.outputs[0]);
                std::vector<VkBuffer> bufs = {operandBuf(env, node.inputs[0], hold0)->handle(), dst->handle(), geom->handle()};
                epi.append(bufs, node, env, dst->handle());
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups((pc.total + pc.items - 1) / pc.items, env.flatLocalSize));
            }
        };

        // ---- Pad (constant/edge/reflect): out[i] maps to an input element, pad region filled per mode ----
        // A constant/absent pad value bakes into pc.cval and runs flat_pad (2 buffers). A RUNTIME pad
        // value (a computed scalar, not an initializer) is not known in prepare(), so the op binds it as
        // a single-element SSBO and runs flat_pad_rt (3 buffers); the pad geometry stays static either
        // way (a runtime pads-geometry input is gated to the CPU, since it makes the output shape
        // data-dependent). The PC block is identical for both shaders, so pc.cval is simply unused on the
        // runtime path.
        struct Pad {
            struct PC {
                int   rank, total, mode;
                float cval;
                int   items = 1;
            } pc {};
            bool                                 runtimeVal = false;
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          geom;  // outDim/inDim/inStride/padBegin, deduped SSBO
            std::shared_ptr<vk::Buffer>          hold0; // when input[0] is a constant initializer
            // Zero-copy pad (mirrors the segment planner's rule): a constant-value pad along ONE
            // axis with every dim before it equal to 1 places the data as a contiguous sub-range of
            // the output at viewOffBytes_. When the planner made the data a view of the output at
            // exactly that offset, record() emits only vkCmdFillBuffer for the two pad ranges.
            bool   viewEligible_ = false;
            size_t viewOffBytes_ = 0, dataBytes_ = 0, outBytes_ = 0;
            void   prepare(const Node &node, VkOpEnv &env) {
                const Graph &g        = *env.graph;
                Shape        in       = g.desc(node.inputs[0]).shape;
                Shape        out      = g.desc(node.outputs[0]).shape;
                int          rank     = (int) out.size();
                auto         inStride = rowStrides(in);
                auto         pads     = readI64Param(g, node, "pads", 1); // [begin..., end...]; begins are pads[0..rank-1]
                std::string  mode     = node.attr.gets("mode", "constant");
                float        cval     = node.attr.getf("value", 0.f);
                if (node.inputs.size() > 2 && node.inputs[2] != kNoTensor && g.isInitializer(node.inputs[2]))
                {
                    // The pad value is a 0-D scalar; read its single element from the initializer bytes
                    // directly (numElements is 0 for an empty shape, so initFloats yields an empty vector).
                    const HostBuffer &hb = g.initializers.at(node.inputs[2]);
                    if (g.desc(node.inputs[2]).dtype == DType::Float16)
                    {
                        if (hb.bytes.size() >= sizeof(fp16_t))
                        {
                            cval = halfToFloatAt(hb.bytes.data(), 0);
                        }
                    } else if (hb.bytes.size() >= sizeof(float))
                    { cval = hb.f32()[0]; }
                }
                runtimeVal = node.inputs.size() > 2 && node.inputs[2] != kNoTensor && !g.isInitializer(node.inputs[2]);
                pc.rank    = rank;
                pc.total   = (int) numElements(out);
                pc.items   = itemsPerLane(pc.total, env);
                pc.mode    = (mode == "edge") ? 1 : (mode == "reflect") ? 2 : 0;
                pc.cval    = cval;
                std::vector<int32_t> outDim(rank), inDim(rank), inStr(rank), padBegin(rank);
                for (int k = 0; k < rank; ++k)
                {
                    outDim[k]   = (int) out[k];
                    inDim[k]    = (int) in[k];
                    inStr[k]    = (int) inStride[k];
                    padBegin[k] = pads.empty() ? 0 : (int) pads[k];
                }
                geom = uploadFlatGeom(env, {outDim, inDim, inStr, padBegin});
                pipe = runtimeVal ? env.pipeline(shader("flat_pad_rt", env.useFp16), 4, sizeof(PC), std::vector<uint32_t> {env.flatLocalSize}) : env.pipeline(shader("flat_pad", env.useFp16), 3, sizeof(PC), std::vector<uint32_t> {env.flatLocalSize});
                {
                    const size_t es      = env.useFp16 ? 2 : 4;
                    int          padAxis = -1;
                    bool         padOk   = pc.mode == 0 && !runtimeVal && (int) pads.size() >= 2 * rank;
                    for (int d = 0; d < rank && padOk; ++d)
                    {
                        const int64_t b = pads[(size_t) d], e = pads[(size_t) (rank + d)];
                        if (b < 0 || e < 0)
                        {
                            padOk = false;
                        } else if (b > 0 || e > 0)
                        {
                            padOk   = padAxis < 0;
                            padAxis = d;
                        }
                    }
                    int64_t outer = 1;
                    for (int d = 0; padOk && d < padAxis; ++d)
                    {
                        outer *= out[(size_t) d];
                    }
                    if (padOk && padAxis >= 0 && outer == 1)
                    {
                        size_t stride = 1;
                        for (int d = padAxis + 1; d < rank; ++d)
                        {
                            stride *= (size_t) out[(size_t) d];
                        }
                        viewOffBytes_ = (size_t) pads[(size_t) padAxis] * stride * es;
                        dataBytes_    = (size_t) numElements(in) * es;
                        outBytes_     = (size_t) numElements(out) * es;
                        viewEligible_ = viewOffBytes_ % sizeof(uint32_t) == 0 && (viewOffBytes_ + dataBytes_) % sizeof(uint32_t) == 0 && outBytes_ % sizeof(uint32_t) == 0;
                    }
                }
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) {
                vk::Buffer *srcB = operandBuf(env, node.inputs[0], hold0);
                vk::Buffer *dstB = env.devBuf(node.outputs[0]);
                // Zero-copy: the data already sits inside the output (the producer wrote through the
                // planner's view), so only the pad ranges need bytes — two transfer fills with the
                // constant value's bit pattern (an fp16 value repeated twice per 32-bit word). The
                // gather dispatch MUST NOT run when the buffers alias: it would read the data range
                // while other invocations overwrite it.
                if (viewEligible_ && !node.attr.has("pw_steps") && srcB->hazardRoot() == dstB->hazardRoot() && srcB->rootOffset() == dstB->rootOffset() + viewOffBytes_)
                {
                    const bool fp16 = env.useFp16;
                    uint32_t   pattern;
                    if (fp16)
                    {
                        constexpr unsigned kHalfBits = 8 * sizeof(uint16_t); // fp16 lane width in the u32 fill word
                        const uint16_t     h         = floatToHalf(pc.cval);
                        pattern                      = (uint32_t) h | ((uint32_t) h << kHalfBits);
                    } else
                    {
                        std::memcpy(&pattern, &pc.cval, sizeof(pattern));
                    }
                    if (viewOffBytes_ > 0)
                    {
                        vkCmdFillBuffer(cmd, dstB->handle(), 0, (VkDeviceSize) viewOffBytes_, pattern);
                    }
                    const size_t tailStart = viewOffBytes_ + dataBytes_;
                    if (outBytes_ > tailStart)
                    {
                        vkCmdFillBuffer(cmd, dstB->handle(), (VkDeviceSize) tailStart, (VkDeviceSize) (outBytes_ - tailStart), pattern);
                    }
                    return;
                }
                VkBuffer src = srcB->handle();
                VkBuffer dst = dstB->handle();
                if (runtimeVal)
                {
                    pipe->dispatch(cmd, {src, env.devBuf(node.inputs[2])->handle(), dst, geom->handle()}, &pc, sizeof(pc), groups((pc.total + pc.items - 1) / pc.items, env.flatLocalSize));
                    return;
                }
                pipe->dispatch(cmd, {src, dst, geom->handle()}, &pc, sizeof(pc), groups((pc.total + pc.items - 1) / pc.items, env.flatLocalSize));
            }
        };

        // ---- Expand / Tile: one gather over right-aligned input dims/strides ----
        // One shader (flat_broadcast) for both, selected by `mode`: Tile wraps a coordinate into its
        // source axis (outCoord % inDim), Expand right-aligns the source into the axis and clamps
        // ahead of it. Both reduce to the plain broadcast on a size-1 source axis; they differ only
        // where the target widens an axis the source neither matches nor broadcasts (see
        // src/backend/cpu/ops/expand.cpp). Input dims/strides are right-aligned into the output rank.
        struct Broadcast {
            struct PC {
                int rank, total, mode, items = 1;
            } pc {};
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          geom;     // outDim/inDim/inStride, deduped SSBO
            std::shared_ptr<vk::Buffer>          constBuf; // when the data operand is a constant initializer
            void                                 prepare(const Node &node, VkOpEnv &env) {
                const Graph &g  = *env.graph;
                Shape        in = g.desc(node.inputs[0]).shape, out = g.desc(node.outputs[0]).shape;
                int          rank         = (int) out.size();
                auto         inStrideFull = rowStrides(in);         // strides in input's own rank
                int          pad          = rank - (int) in.size(); // right-align input into output rank
                pc.rank                   = rank;
                pc.total                  = (int) numElements(out);
                pc.items                  = itemsPerLane(pc.total, env);
                pc.mode                   = (node.type == OpType::Tile) ? 1 : 0;
                std::vector<int32_t> outDim(rank), inDim(rank), inStr(rank);
                for (int k = 0; k < rank; ++k)
                {
                    outDim[k] = (int) out[k];
                    int j     = k - pad; // matching input dim, or -1 if padded (size 1)
                    inDim[k]  = (j >= 0) ? (int) in[j] : 1;
                    inStr[k]  = (j >= 0) ? (int) inStrideFull[j] : 0;
                }
                geom = uploadFlatGeom(env, {outDim, inDim, inStr});
                // Expand/Tile of a constant operand (no activation buffer): upload it flat (direct fp16).
                if (g.isInitializer(node.inputs[0]))
                {
                    constBuf = uploadInit(env, node.inputs[0], in);
                }
                pipe = env.pipeline(shader("flat_broadcast", env.useFp16), 3, sizeof(PC), std::vector<uint32_t> {env.flatLocalSize});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) {
                vk::Buffer *src = constBuf ? constBuf.get() : env.devBuf(node.inputs[0]);
                pipe->dispatch(cmd, {src->handle(), env.devBuf(node.outputs[0])->handle(), geom->handle()}, &pc, sizeof(pc), groups((pc.total + pc.items - 1) / pc.items, env.flatLocalSize));
            }
        };

        // ---- Concat: scatter each input into the output at its axis offset ----
        struct Concat {
            struct PC {
                int rank, total, base, items = 1;
            };
            std::vector<std::shared_ptr<vk::ComputePipeline>> pipes;
            std::vector<PC>                                   pcs;
            std::vector<int>                                  inIdx;
            std::vector<std::shared_ptr<vk::Buffer>>          geoms;                   // per-part inDim/outStride, deduped SSBO (binding 2)
            std::vector<std::shared_ptr<vk::Buffer>>          holds;                   // per-input, set when that input is a constant
            PwEpi                                             epi;                     // fused unit applied at each part's stores
            bool                                              contiguousParts = false; // every dim before the axis is 1: parts are contiguous slabs at pc.base
            void                                              prepare(const Node &node, VkOpEnv &env) {
                const Graph &g    = *env.graph;
                Shape        out  = g.desc(node.outputs[0]).shape;
                int          rank = (int) out.size();
                int64_t      axis = node.attr.geti("axis", 1);
                if (axis < 0)
                {
                    axis += rank;
                }
                {
                    int64_t outer = 1;
                    for (int d = 0; d < (int) axis && d < rank; ++d)
                    {
                        outer *= out[d];
                    }
                    contiguousParts = outer == 1;
                }
                epi.prepare(node, env, true, out);
                auto    outStride = rowStrides(out);
                int64_t offset    = 0;
                // Inputs from pwCoreInputs on are the fused unit's operands, not concatenated parts.
                size_t nIn = (size_t) pwCoreInputs(node);
                for (size_t e = 0; e < nIn && e < node.inputs.size(); ++e)
                {
                    if (node.inputs[e] == kNoTensor)
                    {
                        continue;
                    }
                    Shape in = g.desc(node.inputs[e]).shape;
                    PC    pc {};
                    pc.rank  = rank;
                    pc.total = (int) numElements(in);
                    pc.items = itemsPerLane(pc.total, env);
                    pc.base  = (int) (offset * outStride[axis]);
                    std::vector<int32_t> inDim(rank), outStr(rank);
                    for (int k = 0; k < rank; ++k)
                    {
                        inDim[k]  = (int) in[k];
                        outStr[k] = (int) outStride[k];
                    }
                    geoms.push_back(uploadFlatGeom(env, {inDim, outStr}));
                    pcs.push_back(pc);
                    inIdx.push_back((int) e);
                    offset += in[axis];
                    pipes.push_back(env.pipeline(shader((std::string("flat_scatter") + epi.suffix()).c_str(), env.useFp16), 3 + epi.extraBufs(), sizeof(PC), std::vector<uint32_t> {env.flatLocalSize}));
                }
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) {
                vk::Buffer *dst = env.devBuf(node.outputs[0]);
                if (holds.size() < pcs.size())
                {
                    holds.resize(pcs.size());
                }
                // A fused unit must run at the stores, so an epi-carrying Concat never skips a part.
                const bool   mayAlias  = contiguousParts && !node.attr.has("pw_steps");
                const size_t elemBytes = env.useFp16 ? 2 : 4;
                for (size_t i = 0; i < pcs.size(); ++i)
                {
                    vk::Buffer *src = operandBuf(env, node.inputs[inIdx[i]], holds[i]);
                    // Zero-copy: the planner made this part a sub-buffer view of the output at exactly
                    // its slab offset (pc.base elements), so the producer already wrote it in place.
                    if (mayAlias && src->hazardRoot() == dst->hazardRoot() && src->rootOffset() == dst->rootOffset() + (size_t) pcs[i].base * elemBytes)
                    {
                        continue;
                    }
                    std::vector<VkBuffer> bufs {src->handle(), dst->handle(), geoms[i]->handle()};
                    epi.append(bufs, node, env, dst->handle());
                    pipes[i]->dispatch(cmd, bufs, &pcs[i], sizeof(PC), groups((pcs[i].total + pcs[i].items - 1) / pcs[i].items, env.flatLocalSize));
                }
            }
        };

        // ---- broadcasting Binary / Add (handles a constant operand by uploading it flat) ----
        struct Binary {
            struct PC {
                int   rank, total, op;
                int   act;
                float actLo, actHi;
                int   bothFull; // both operands same shape as output => index == gid (skip the stride loop)
                int   items = 1;
            } pc {};
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          geom; // outDim/aStride/bStride, deduped SSBO
            std::shared_ptr<vk::Buffer>          constBuf[2];
            void                                 prepare(const Node &node, VkOpEnv &env) {
                const Graph &g    = *env.graph;
                Shape        out  = g.desc(node.outputs[0]).shape;
                int          rank = (int) out.size();
                pc.rank           = rank;
                pc.total          = (int) numElements(out);
                pc.items          = itemsPerLane(pc.total, env);
                pc.op             = node.type == OpType::Add ? (int) BinaryType::Add : node.subOp;
                // A fused activation (e.g. a Linear+Relu fused into the Add epilogue, as in the camera_head
                // res_conv) is applied here, matching the NC4HW4 add.
                pc.act   = (int) node.fusedAct;
                pc.actLo = node.actLo;
                pc.actHi = node.actHi;
                std::vector<int32_t> outDim(rank), aStride(rank), bStride(rank);
                for (int k = 0; k < rank; ++k)
                {
                    outDim[k] = (int) out[k];
                }
                auto setup = [&](TensorId t, int which) {
                    Shape                s = g.desc(t).shape;
                    std::vector<int64_t> ps(rank, 1); // left-pad to out rank
                    for (int k = 0; k < (int) s.size(); ++k)
                    {
                        ps[rank - (int) s.size() + k] = s[k];
                    }
                    auto st = rowStrides(ps);
                    for (int k = 0; k < rank; ++k)
                    {
                        int stride                                        = ps[k] == 1 ? 0 : (int) st[k];
                        (which == 0 ? aStride.data() : bStride.data())[k] = stride;
                    }
                    if (g.isInitializer(t))
                    {
                        constBuf[which] = uploadInit(env, t, s);
                    }
                };
                setup(node.inputs[0], 0);
                setup(node.inputs[1], 1);
                pc.bothFull = (g.desc(node.inputs[0]).shape == out && g.desc(node.inputs[1]).shape == out) ? 1 : 0;
                geom        = uploadFlatGeom(env, {outDim, aStride, bStride});
                pipe = env.pipeline(shader("flat_binary", env.useFp16), 4, sizeof(PC), std::vector<uint32_t> {env.flatLocalSize});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) {
                auto buf = [&](int e) {
                    return constBuf[e] ? constBuf[e].get() : env.devBuf(node.inputs[e]);
                };
                pipe->dispatch(cmd, {buf(0)->handle(), buf(1)->handle(), env.devBuf(node.outputs[0])->handle(), geom->handle()}, &pc, sizeof(pc), groups((pc.total + pc.items - 1) / pc.items, env.flatLocalSize));
            }
        };

        // ---- Softmax over an arbitrary axis ----
        struct Softmax {
            // Deterministic row-mapping rule: a row this narrow leaves most of a 128-lane workgroup
            // idle and pays ~14 barrier rounds for a 16-element reduction (a detection head's DFL
            // distribution), so it runs one THREAD per row instead. The per-row reduction order
            // differs between the mappings, so the choice is a shape rule, never a timing race.
            // Attention rows (axis >= a context length) stay on the workgroup mapping.
            static constexpr int kThreadRowMaxAxis = 32;
            // outPad is the output's physical last-axis extent, 0 unless the segment allocated the
            // output as a virtualized activation (VkOpEnv::rowPad). Only a LAST-axis softmax
            // (inner == 1) can carry it: padding widens the last axis, which is the reduced one.
            struct PC {
                int outer, axis, inner, outPad;
            } pc {};
            bool                                 threadRow = false;
            std::shared_ptr<vk::ComputePipeline> pipe;
            PwEpi                                epi;
            void                                 prepare(const Node &node, VkOpEnv &env) {
                Shape   s    = env.graph->desc(node.inputs[0]).shape;
                int     rank = (int) s.size();
                int64_t axis = node.attr.geti("axis", -1);
                if (axis < 0)
                {
                    axis += rank;
                }
                int64_t outer = 1, inner = 1;
                for (int k = 0; k < (int) axis; ++k)
                {
                    outer *= s[k];
                }
                for (int k = (int) axis + 1; k < rank; ++k)
                {
                    inner *= s[k];
                }
                pc                   = {(int) outer, (int) s[axis], (int) inner, 0};
                const int64_t outPad = env.rowPad ? env.rowPad(node.outputs[0]) : 0;
                if (outPad > 0 && inner == 1)
                {
                    pc.outPad = (int) outPad; // rows start at o * outPad; the kernel zeroes the pad
                }
                threadRow = s[axis] <= kThreadRowMaxAxis;
                epi.prepare(node, env, /*flat=*/true, env.graph->desc(node.outputs[0]).shape);
                pipe = env.pipeline(shader((std::string("flat_softmax") + epi.suffix()).c_str(), env.useFp16), 2 + epi.extraBufs(), sizeof(PC), std::vector<uint32_t> {(uint32_t) (threadRow ? 1 : 0)});
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) {
                // ROW_MODE 0: one workgroup per row (LDS reduction across the workgroup).
                // ROW_MODE 1: one thread per row (rows packed 128 to a workgroup).
                VkBuffer              dst  = env.devBuf(node.outputs[0])->handle();
                std::vector<VkBuffer> bufs = {env.devBuf(node.inputs[0])->handle(), dst};
                epi.append(bufs, node, env, dst);
                int64_t rows   = (int64_t) pc.outer * pc.inner;
                int64_t groups = threadRow ? (rows + 127) / 128 : rows;
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), (uint32_t) groups);
            }
        };

    } // namespace flat
} // namespace vknn
