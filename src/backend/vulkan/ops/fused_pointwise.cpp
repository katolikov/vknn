// Standalone FusedPointwise: dispatch the pw_steps/pw_params chain (produced by
// fusePointwiseChains) as one kernel instead of one dispatch per original op. The plan (steps +
// broadcast strides) is uploaded once as a small SSBO read by shaders/pw_epilogue.glsl; the extra
// step operands bind at consecutive slots after it (see PW_EPI_BASE in the .comp files).
#include "vk_op_common.h"
#include "vknn/op.h"
#include "flat_ops.h"
#include <algorithm>
#include <cstdint>

namespace vknn {
    namespace {
        // Byte-identical to the std430 PwPlan block in shaders/pw_epilogue.glsl.
        struct PwPlanCPU {
            int32_t numSteps, rank, worldFlat, pad;
            int32_t outDim[kPwMaxRank];
            int32_t step[kPwMaxSteps * 4];
            int32_t stride[kPwMaxSteps * kPwMaxRank];
            float   p0[kPwMaxSteps];
            float   p1[kPwMaxSteps];
        };
        static_assert(sizeof(PwPlanCPU) == 352, "PwPlanCPU must match the std430 PwPlan block");

        struct FusedPointwiseOp: VulkanOp {
            std::unique_ptr<vk::ComputePipeline>     pipe;
            std::shared_ptr<vk::Buffer>               planBuf;
            std::vector<std::shared_ptr<vk::Buffer>>  holds;
            int                                       total = 0;
            bool                                       flat  = false;

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g = *env.graph;
                flat           = opIsFlat(node, env);
                Shape out      = g.desc(node.outputs[0]).shape;
                const auto &st = node.attr.getints("pw_steps");
                const auto &pr = node.attr.getfloats("pw_params");
                int         nSteps = (int) (st.size() / 4);

                holds.assign(node.inputs.size(), nullptr);
                PwPlanCPU plan {};
                plan.numSteps  = nSteps;
                plan.worldFlat = flat ? 1 : 0;

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
                        plan.step[s * 4 + 2] = oi;
                        plan.step[s * 4 + 3] = bc;
                        plan.p0[s]           = pr[s * 2];
                        plan.p1[s]           = pr[s * 2 + 1];
                        if (kind == 0)
                        {
                            Shape                os = g.desc(node.inputs[oi]).shape;
                            std::vector<int64_t> ps(rank, 1);
                            for (int k = 0; k < (int) os.size() && k < rank; ++k)
                            {
                                ps[rank - 1 - k] = os[(int) os.size() - 1 - k];
                            }
                            auto ss = flat::rowStrides(ps);
                            for (int k = 0; k < rank; ++k)
                            {
                                plan.stride[s * kPwMaxRank + k] = (ps[k] == 1) ? 0 : (int) ss[k];
                            }
                            if (g.isInitializer(node.inputs[oi]))
                            {
                                holds[oi] = uploadInit(env, node.inputs[oi], os);
                            }
                        }
                    }
                } else
                {
                    NCHW y   = NCHW::from(out);
                    int  HW  = (int) (y.h * y.w);
                    plan.rank    = 1;
                    plan.outDim[0] = HW;
                    total = (int) ((int64_t) y.n * ((y.c + 3) / 4) * HW);
                    for (int s = 0; s < nSteps; ++s)
                    {
                        int kind = (int) st[s * 4], code = (int) st[s * 4 + 1], oi = (int) st[s * 4 + 2], bc = (int) st[s * 4 + 3];
                        plan.step[s * 4]     = kind;
                        plan.step[s * 4 + 1] = code;
                        plan.step[s * 4 + 2] = oi;
                        plan.step[s * 4 + 3] = bc;
                        plan.p0[s]           = pr[s * 2];
                        plan.p1[s]           = pr[s * 2 + 1];
                        if (kind == 0 && g.isInitializer(node.inputs[oi]))
                        {
                            holds[oi] = uploadInit(env, node.inputs[oi], g.desc(node.inputs[oi]).shape);
                        }
                    }
                }

                planBuf = std::make_shared<vk::Buffer>(*env.ctx, sizeof(PwPlanCPU), vk::MemPref::kAuto);
                planBuf->upload(&plan, sizeof(plan));
                pipe = std::make_unique<vk::ComputePipeline>(*env.ctx, shader(flat ? "fused_pw_flat" : "fused_pw_nc4", env.useFp16),
                                                              2 + 1 + kPwMaxOperands, sizeof(int), std::vector<uint32_t> {}, env.cache->handle());
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *dst = env.devBuf(node.outputs[0]);
                std::vector<VkBuffer> bufs;
                bufs.push_back(env.devBuf(node.inputs[0])->handle());
                bufs.push_back(dst->handle());
                bufs.push_back(planBuf->handle());
                for (int k = 0; k < kPwMaxOperands; ++k)
                {
                    int idx = 1 + k;
                    if (idx < (int) node.inputs.size() && node.inputs[idx] != kNoTensor)
                    {
                        bufs.push_back(operandBuf(env, node.inputs[idx], holds[idx])->handle());
                    } else
                    {
                        bufs.push_back(dst->handle());
                    }
                }
                int pc = total;
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(total, 256));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::FusedPointwise, FusedPointwiseOp);
} // namespace vknn
