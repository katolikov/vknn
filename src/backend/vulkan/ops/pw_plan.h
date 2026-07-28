// Shared pointwise-unit plan builder: turns a node's pw_steps/pw_params/pw_outs attrs into the
// device PwPlan block (shaders/pw_epilogue.glsl) plus the ordered operand tensor list. Each step is
// an 8-int record (kind, code, srcA, srcB, srcC, dst, bcast, bcastSrc) whose sources reference the
// accumulator, the unit's entry value, one of kPwMaxRegs registers, or a tensor operand
// (kPwRefOp0 - i). Operand references arrive indexing node.inputs and are mapped here to a dense
// physical slot (deduped via slotOf) so the same plan format serves both the standalone
// FusedPointwise op (operands at inputs[1..]) and a producer's fused epilogue (operands appended at
// inputs[pw_opbase..]). pw_outs lists the step whose value each extra output stream stores
// (kPwRefEntry exports the entry value itself).
#pragma once
#include "vk_op_common.h"
#include "vknn/op.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace vknn {

    // Byte-identical to the std430 PwPlan block in shaders/pw_epilogue.glsl. `flags` bit 0
    // (pw_relax attr) selects the fp32-chained discipline: steps run unrounded and the unit rounds
    // once per stored stream instead of per step.
    struct PwPlanCPU {
        int32_t numSteps, rank, worldFlat, numOuts, flags;
        int32_t outDim[kPwMaxRank];
        int32_t step[kPwMaxSteps * 8]; // kind, code, srcA, srcB, srcC, dst, bcast, bcastSrc
        int32_t stride[kPwMaxSteps * kPwMaxRank];
        float   p0[kPwMaxSteps];
        float   p1[kPwMaxSteps];
        int32_t outStep[kPwMaxOuts];
    };
    static_assert(sizeof(PwPlanCPU) == 948, "PwPlanCPU must match the std430 PwPlan block");

    // Is `ref` a tensor-operand reference (kPwRefOp0 - i)?
    inline bool pwRefIsOperand(int ref) {
        return ref <= kPwRefOp0;
    }

    // Build the plan + the ordered operand tensor list (physical slot i -> operands[i-1]) from
    // node.attr pw_steps/pw_params/pw_outs. `out` is the producer/output shape; `flat` picks the
    // index math (row-major broadcast strides vs. NC4HW4 channel-broadcast). `total` is the dispatch
    // element count in the op's own convention (a caller with its own dispatch geometry may ignore
    // it).
    inline void buildPwPlan(const Graph &g, const Node &node, bool flat, const Shape &out, PwPlanCPU &plan, std::vector<TensorId> &operands, int &total) {
        const auto &st     = node.attr.getints("pw_steps");
        const auto &pr     = node.attr.getfloats("pw_params");
        int         nSteps = (int) (st.size() / 8);
        plan               = PwPlanCPU {};
        plan.numSteps      = nSteps;
        plan.worldFlat     = flat ? 1 : 0;
        plan.flags         = node.attr.geti("pw_relax", 0) != 0 ? 1 : 0;
        operands.clear();
        auto slotOf = [&](TensorId t) -> int {
            for (size_t i = 0; i < operands.size(); ++i)
            {
                if (operands[i] == t)
                {
                    return (int) i;
                }
            }
            operands.push_back(t);
            return (int) operands.size() - 1;
        };
        // Remap a source reference from node.inputs index space to dense physical slot space;
        // accumulator/entry/register references pass through unchanged.
        auto mapRef = [&](int ref) -> int {
            if (!pwRefIsOperand(ref))
            {
                return ref;
            }
            return kPwRefOp0 - slotOf(node.inputs[kPwRefOp0 - ref]);
        };
        // The operand tensor a step's broadcast geometry applies to (bcastSrc selects the field),
        // or kNoTensor for a step whose operands are all same-shape / non-tensor.
        auto bcastOperand = [&](int s) -> TensorId {
            int bsrc = (int) st[s * 8 + 7];
            if (bsrc < 1 || bsrc > 3)
            {
                return kNoTensor;
            }
            int ref = (int) st[s * 8 + 1 + bsrc]; // 1=srcA(2), 2=srcB(3), 3=srcC(4) -> field index
            return pwRefIsOperand(ref) ? node.inputs[kPwRefOp0 - ref] : kNoTensor;
        };

        for (int s = 0; s < nSteps; ++s)
        {
            plan.step[s * 8]     = (int32_t) st[s * 8];     // kind
            plan.step[s * 8 + 1] = (int32_t) st[s * 8 + 1]; // code
            plan.step[s * 8 + 2] = (int32_t) mapRef((int) st[s * 8 + 2]);
            plan.step[s * 8 + 3] = (int32_t) mapRef((int) st[s * 8 + 3]);
            plan.step[s * 8 + 4] = (int32_t) mapRef((int) st[s * 8 + 4]);
            plan.step[s * 8 + 5] = (int32_t) st[s * 8 + 5]; // dst register
            plan.step[s * 8 + 6] = (int32_t) st[s * 8 + 6]; // bcast mode
            plan.step[s * 8 + 7] = (int32_t) st[s * 8 + 7]; // bcast source field
            plan.p0[s]           = pr[s * 2];
            plan.p1[s]           = pr[s * 2 + 1];
        }

        if (flat)
        {
            int rank  = std::min((int) out.size(), kPwMaxRank);
            plan.rank = rank;
            for (int k = 0; k < rank; ++k)
            {
                plan.outDim[k] = (int) out[(int) out.size() - rank + k];
            }
            total = (int) numElements(out);
            for (int s = 0; s < nSteps; ++s)
            {
                TensorId opd = bcastOperand(s);
                if (opd == kNoTensor)
                {
                    continue;
                }
                // Right-align the operand shape into `rank` dims (ps), build its row-major strides
                // (ss), then emit a NumPy-style broadcast stride per dim: a size-1 dim gets stride 0
                // so the kernel repeats element 0 along that axis.
                Shape                os = g.desc(opd).shape;
                std::vector<int64_t> ps(rank, 1);
                for (int k = 0; k < (int) os.size() && k < rank; ++k)
                {
                    ps[rank - 1 - k] = os[(int) os.size() - 1 - k];
                }
                // A valid broadcast requires each right-aligned operand axis to be 1 or equal to the
                // output extent (outDim), and the operand rank cannot exceed the fused output rank; a
                // non-conforming extent is a malformed graph that the strides below would turn into an
                // out-of-bounds shader read, so it is rejected here at plan-build time.
                if ((int) os.size() > rank)
                {
                    throw Error(Status::InvalidArgument, "FusedPointwise (" + node.name + ") operand tensor " + std::to_string(opd) + " shape " + shapeStr(os) + " has higher rank than fused output " + shapeStr(out));
                }
                for (int k = 0; k < rank; ++k)
                {
                    if (ps[k] != 1 && ps[k] != plan.outDim[k])
                    {
                        throw Error(Status::InvalidArgument, "FusedPointwise (" + node.name + ") operand tensor " + std::to_string(opd) + " shape " + shapeStr(os) + " is not broadcast-compatible with output " + shapeStr(out) + " (axis " + std::to_string(k) + ": " + std::to_string(ps[k]) + " vs " + std::to_string(plan.outDim[k]) + ")");
                    }
                }
                std::vector<int64_t> ss(rank, 1);
                for (int k = rank - 2; k >= 0; --k)
                {
                    ss[k] = ss[k + 1] * ps[k + 1];
                }
                for (int k = 0; k < rank; ++k)
                {
                    plan.stride[s * kPwMaxRank + k] = (ps[k] == 1) ? 0 : (int) ss[k];
                }
            }
        } else
        {
            // NC4HW4 world: the kernel indexes packed channel blocks, so the plan carries only the
            // spatial extent (rank 1, outDim[0] = H*W). `total` is the thread count: one thread per
            // (n, channel-block, hw) triple = n * ceil(c/4) * H*W, each thread handling the 4 packed
            // channel lanes as a vec4. Blocks for c not a multiple of 4 (padded lanes) are included.
            NCHW y         = NCHW::from(out);
            int  HW        = (int) (y.h * y.w);
            plan.rank      = 1;
            plan.outDim[0] = HW;
            total          = (int) ((int64_t) y.n * ((y.c + 3) / 4) * HW);
        }

        const auto &po = node.attr.getints("pw_outs");
        plan.numOuts   = std::min((int) po.size(), kPwMaxOuts);
        for (int o = 0; o < plan.numOuts; ++o)
        {
            plan.outStep[o] = (int32_t) po[o];
        }
        for (int o = plan.numOuts; o < kPwMaxOuts; ++o)
        {
            plan.outStep[o] = kPwRefNone;
        }
    }

    // Upload the plan, content-deduped: nodes with byte-identical plans (same steps + shapes, the
    // common case for repeated transformer layers) share one device buffer/allocation.
    inline std::shared_ptr<vk::Buffer> uploadPwPlan(VkOpEnv &env, const PwPlanCPU &plan) {
        return env.uploadPooled(&plan, sizeof(plan));
    }

    // Resolve a unit operand to a device buffer. A runtime activation already lives in the unit's
    // world; a CONSTANT operand uploads flat for a flat-world unit, but must be NC4HW4-packed for
    // an NC4 unit — the kernel indexes operands with packed indices, and packing also materializes
    // the padded dead lanes a whole-block load touches (a flat upload would be misordered for
    // C % 4 != 0 or H*W > 1, and read out of bounds on the pad lanes).
    inline vk::Buffer *pwOperandBuf(VkOpEnv &env, TensorId t, std::shared_ptr<vk::Buffer> &hold, bool flatWorld) {
        const Graph &g = *env.graph;
        if (!g.isInitializer(t))
        {
            return env.devBuf(t);
        }
        if (!hold)
        {
            // A pw operand always carries at least one element; an empty payload means the .vxm is
            // broken (a compiler that dropped rank-0 scalars). Refuse it — the splat/pack below
            // would otherwise upload silent zeros or an undefined buffer.
            if (g.initializers.at(t).bytes.empty())
            {
                throw Error(Status::RuntimeError, "pw operand initializer tensor " + std::to_string(t) + " has an empty payload; recompile the .vxm");
            }
            if (flatWorld)
            {
                hold = uploadInit(env, t, g.desc(t).shape);
            } else if (numElements(g.desc(t).shape) <= 1)
            {
                // scalar splat (bcast mode 3): the kernel reads element 0 only
                std::vector<float> v = initFloats(g, t);
                hold                 = upload(*env.ctx, std::vector<float>(4, v.empty() ? 0.f : v[0]), env.useFp16);
            } else
            {
                std::vector<float> v = initFloats(g, t);
                // Right-align a rank<4 constant into NCHW before packing: a [C,1,1] (or [1,C,1,1])
                // channel operand broadcasts by right-alignment, so its packed layout must match
                // the [1,C,1,1] interpretation the kernel's channel-block indexing assumes.
                Shape sh = g.desc(t).shape;
                while (sh.size() < 4)
                {
                    sh.insert(sh.begin(), 1);
                }
                NCHW               s  = NCHW::from(sh);
                int64_t            Cb = cBlocks(s.c), HW = s.h * s.w;
                std::vector<float> p((size_t) (s.n * Cb * 4 * HW), 0.f);
                for (int64_t n = 0; n < s.n; ++n)
                {
                    for (int64_t c = 0; c < s.c; ++c)
                    {
                        for (int64_t hw = 0; hw < HW; ++hw)
                        {
                            p[(size_t) ((((n * Cb + c / 4) * HW) + hw) * 4 + (c & 3))] = v[(size_t) ((n * s.c + c) * HW + hw)];
                        }
                    }
                }
                hold = upload(*env.ctx, p, env.useFp16);
            }
        }
        return hold.get();
    }

    // Wires a node's attached pointwise unit (pw_steps) into the op's own kernel. Usage:
    //   prepare():  epi.prepare(node, env, flat, outShape);
    //               pipe = env.pipeline(shader((base + epi.suffix()).c_str(), fp16), nbuf + epi.extraBufs(), ...);
    //   record():   epi.append(bufs, node, env, dstHandle);   // after the kernel's own buffers
    // The _epi shader variant executes the unit at its store via shaders/pw_epilogue.glsl; the plan
    // SSBO, the operand buffers, and the extra output streams bind at PW_EPI_BASE = the kernel's own
    // buffer count.
    struct PwEpi {
        std::shared_ptr<vk::Buffer>              plan;
        std::vector<TensorId>                    operands;
        std::vector<std::shared_ptr<vk::Buffer>> holds;
        bool                                     active    = false;
        bool                                     flatWorld = true;
        bool                                     relax     = false;

        void prepare(const Node &node, VkOpEnv &env, bool flat, const Shape &out) {
            active = node.attr.has("pw_steps");
            if (!active)
            {
                return;
            }
            flatWorld = flat;
            relax     = node.attr.geti("pw_relax", 0) != 0;
            PwPlanCPU p {};
            int       total = 0;
            buildPwPlan(*env.graph, node, flat, out, p, operands, total);
            plan = uploadPwPlan(env, p);
            holds.assign(operands.size(), nullptr);
        }
        // The rounding discipline is compiled into the SPIR-V (see shaders/pw_epilogue.glsl):
        // "_epi" carries the strict per-step-rounded appliers, "_epi_rx" the fp32-chained ones.
        const char *suffix() const {
            return !active ? "" : relax ? "_epi_rx" : "_epi";
        }
        uint32_t extraBufs() const {
            return active ? 1u + (uint32_t) kPwMaxOperands + (uint32_t) kPwMaxOuts : 0u;
        }
        // Binding set for a load-time timing race: the real plan SSBO plus `filler` in every
        // operand and extra-output slot. A race runs on dedicated scratch buffers, so there are no
        // live operands to resolve and the results are discarded - the operand VALUES do not matter.
        // What does matter is that the race dispatches the SAME kernel variant the graph will, at
        // the same binding count: the epilogue's register demand is part of what decides which conv
        // tile is fastest, and a race run on the plain stem ranks the tiles for a kernel that is
        // never dispatched. `filler` must be at least output-sized, since the epilogue indexes its
        // operands by output element.
        void appendForTiming(std::vector<VkBuffer> &bufs, VkBuffer filler) const {
            if (!active)
            {
                return;
            }
            bufs.push_back(plan->handle());
            for (int slot = 0; slot < kPwMaxOperands + kPwMaxOuts; ++slot)
            {
                bufs.push_back(filler);
            }
        }
        void append(std::vector<VkBuffer> &bufs, const Node &node, VkOpEnv &env, VkBuffer dummy) {
            if (!active)
            {
                return;
            }
            bufs.push_back(plan->handle());
            for (int k = 0; k < kPwMaxOperands; ++k)
            {
                bufs.push_back(k < (int) operands.size() ? pwOperandBuf(env, operands[k], holds[k], flatWorld)->handle() : dummy);
            }
            // Extra output streams (pw_outs): the fused unit stores exported step values to
            // node.outputs[1..]. Unused slots bind the dummy; the plan's outStep entries are
            // kPwRefNone there, so the kernel never writes them.
            for (int o = 0; o < kPwMaxOuts; ++o)
            {
                bool live = 1 + o < (int) node.outputs.size() && node.outputs[1 + o] != kNoTensor;
                bufs.push_back(live ? env.devBuf(node.outputs[1 + o])->handle() : dummy);
            }
        }
    };

} // namespace vknn
