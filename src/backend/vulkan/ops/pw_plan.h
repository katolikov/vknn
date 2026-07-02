// Shared pointwise-chain plan builder: turns a node's pw_steps/pw_params attrs into the device
// PwPlan block (shaders/pw_epilogue.glsl) plus the ordered operand tensor list. Each step's
// absolute operandInputIdx (an index into node.inputs) is mapped to a dense physical operand slot
// (1..K, deduped via slotOf) so the same plan format serves both the standalone FusedPointwise op
// (operands at inputs[1..]) and a producer's fused epilogue (operands appended at inputs[opbase..]).
#pragma once
#include "vk_op_common.h"
#include "vknn/op.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace vknn {

    // Byte-identical to the std430 PwPlan block in shaders/pw_epilogue.glsl.
    struct PwPlanCPU {
        int32_t numSteps, rank, worldFlat, pad;
        int32_t outDim[kPwMaxRank];
        int32_t step[kPwMaxSteps * 4]; // kind, code, opSlot (dense 1..K or 0), bcast
        int32_t stride[kPwMaxSteps * kPwMaxRank];
        float   p0[kPwMaxSteps];
        float   p1[kPwMaxSteps];
    };
    static_assert(sizeof(PwPlanCPU) == 352, "PwPlanCPU must match the std430 PwPlan block");

    // Build the plan + the ordered operand tensor list (physical slot i -> operands[i-1]) from
    // node.attr pw_steps/pw_params. `out` is the producer/output shape; `flat` picks the index math
    // (row-major broadcast strides vs. NC4HW4 channel-broadcast). `total` is the dispatch element
    // count in the op's own convention (a caller with its own dispatch geometry may ignore it).
    inline void buildPwPlan(const Graph &g, const Node &node, bool flat, const Shape &out, PwPlanCPU &plan, std::vector<TensorId> &operands, int &total) {
        const auto &st     = node.attr.getints("pw_steps");
        const auto &pr     = node.attr.getfloats("pw_params");
        int         nSteps = (int) (st.size() / 4);
        plan               = PwPlanCPU {};
        plan.numSteps      = nSteps;
        plan.worldFlat     = flat ? 1 : 0;
        operands.clear();
        auto slotOf = [&](TensorId t) -> int {
            for (size_t i = 0; i < operands.size(); ++i)
            {
                if (operands[i] == t)
                {
                    return (int) i + 1;
                }
            }
            operands.push_back(t);
            return (int) operands.size();
        };
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
                int kind = (int) st[s * 4], code = (int) st[s * 4 + 1], oi = (int) st[s * 4 + 2], bc = (int) st[s * 4 + 3];
                plan.step[s * 4]     = kind;
                plan.step[s * 4 + 1] = code;
                plan.step[s * 4 + 3] = bc;
                plan.p0[s]           = pr[s * 2];
                plan.p1[s]           = pr[s * 2 + 1];
                if (kind == 0)
                {
                    TensorId opd            = node.inputs[oi];
                    plan.step[s * 4 + 2]    = slotOf(opd);
                    Shape                os = g.desc(opd).shape;
                    std::vector<int64_t> ps(rank, 1);
                    for (int k = 0; k < (int) os.size() && k < rank; ++k)
                    {
                        ps[rank - 1 - k] = os[(int) os.size() - 1 - k];
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
                } else
                {
                    plan.step[s * 4 + 2] = 0;
                }
            }
        } else
        {
            NCHW y         = NCHW::from(out);
            int  HW        = (int) (y.h * y.w);
            plan.rank      = 1;
            plan.outDim[0] = HW;
            total          = (int) ((int64_t) y.n * ((y.c + 3) / 4) * HW);
            for (int s = 0; s < nSteps; ++s)
            {
                int kind = (int) st[s * 4], code = (int) st[s * 4 + 1], oi = (int) st[s * 4 + 2], bc = (int) st[s * 4 + 3];
                plan.step[s * 4]     = kind;
                plan.step[s * 4 + 1] = code;
                plan.step[s * 4 + 3] = bc;
                plan.p0[s]           = pr[s * 2];
                plan.p1[s]           = pr[s * 2 + 1];
                plan.step[s * 4 + 2] = (kind == 0) ? slotOf(node.inputs[oi]) : 0;
            }
        }
    }

    // Upload the plan, content-deduped: nodes with byte-identical plans (same steps + shapes, the
    // common case for repeated transformer layers) share one device buffer/allocation.
    inline std::shared_ptr<vk::Buffer> uploadPwPlan(VkOpEnv &env, const PwPlanCPU &plan) {
        return env.uploadPooled(&plan, sizeof(plan));
    }

    // Resolve a chain operand to a device buffer. A runtime activation already lives in the chain's
    // world; a CONSTANT operand uploads flat for a flat-world chain, but must be NC4HW4-packed for
    // an NC4 chain — the kernel indexes operands with packed indices, and packing also materializes
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
                std::vector<float> v  = initFloats(g, t);
                NCHW               s  = NCHW::from(g.desc(t).shape);
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

    // Wires a node's attached pointwise chain (pw_steps) into the op's own kernel. Usage:
    //   prepare():  epi.prepare(node, env, flat, outShape);
    //               pipe = env.pipeline(shader((base + epi.suffix()).c_str(), fp16), nbuf + epi.extraBufs(), ...);
    //   record():   epi.append(bufs, node, env, dstHandle);   // after the kernel's own buffers
    // The _epi shader variant executes the chain at its store via shaders/pw_epilogue.glsl; the plan
    // SSBO + operand buffers bind at PW_EPI_BASE = the kernel's own buffer count.
    struct PwEpi {
        std::shared_ptr<vk::Buffer>              plan;
        std::vector<TensorId>                    operands;
        std::vector<std::shared_ptr<vk::Buffer>> holds;
        bool                                     active = false;
        bool                                     flatWorld = true;

        void prepare(const Node &node, VkOpEnv &env, bool flat, const Shape &out) {
            active = node.attr.has("pw_steps");
            if (!active)
            {
                return;
            }
            flatWorld = flat;
            PwPlanCPU p {};
            int       total = 0;
            buildPwPlan(*env.graph, node, flat, out, p, operands, total);
            plan = uploadPwPlan(env, p);
            holds.assign(operands.size(), nullptr);
        }
        const char *suffix() const {
            return active ? "_epi" : "";
        }
        uint32_t extraBufs() const {
            return active ? 1u + (uint32_t) kPwMaxOperands : 0u;
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
        }
    };

} // namespace vknn
