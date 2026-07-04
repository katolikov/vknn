// Standalone FusedPointwise: dispatch the pw_steps/pw_params chain (produced by
// fusePointwiseChains) as one kernel instead of one dispatch per original op. The plan (steps +
// broadcast strides) is uploaded once as a small SSBO read by shaders/pw_epilogue.glsl; the extra
// step operands bind at consecutive slots after it (see PW_EPI_BASE in the .comp files).
#include "flat_ops.h"
#include "pw_plan.h"
#include "vk_op_common.h"
#include "vknn/op.h"
#include <algorithm>
#include <cstdint>

namespace vknn {
    namespace {
        struct FusedPointwiseOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline>     pipe;
            std::shared_ptr<vk::Buffer>              planBuf;
            std::vector<TensorId>                    operands;
            std::vector<std::shared_ptr<vk::Buffer>> holds;
            int                                      total = 0;
            bool                                     flat  = false;

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g = *env.graph;
                flat           = opIsFlat(node, env);
                Shape out      = g.desc(node.outputs[0]).shape;

                PwPlanCPU plan {};
                buildPwPlan(g, node, flat, out, plan, operands, total);
                holds.assign(operands.size(), nullptr);

                planBuf = uploadPwPlan(env, plan);
                // Spec-constant specialization (flat only): bake the chain structure (numSteps + each
                // step's kind/code/opSlot/bcast) into specialization constants so the driver folds the
                // interpreter loop/branches to straight-line ISA (shaders/pw_epilogue.glsl #ifdef PW_SPEC).
                // The pipeline pool spec-keys these, so chains with the same structure share one pipeline.
                // Byte-identical to the generic interpreter; the numeric plan (p0/p1/stride) stays in the SSBO.
                bool                  useSpec = flat && env.config && env.config->specializePointwise;
                std::vector<uint32_t> spec;
                if (useSpec)
                {
                    spec.reserve(1 + (size_t) kPwMaxSteps * 4);
                    spec.push_back((uint32_t) plan.numSteps);
                    for (int s = 0; s < kPwMaxSteps; ++s)
                    {
                        for (int f = 0; f < 4; ++f)
                        {
                            spec.push_back((uint32_t) plan.step[s * 4 + f]);
                        }
                    }
                }
                const char *base = flat ? (useSpec ? "fused_pw_flat_spec" : "fused_pw_flat") : "fused_pw_nc4";
                pipe = env.pipeline(shader(base, env.useFp16), 2 + 1 + kPwMaxOperands, sizeof(int), spec);
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer           *dst = env.devBuf(node.outputs[0]);
                std::vector<VkBuffer> bufs;
                bufs.push_back(env.devBuf(node.inputs[0])->handle());
                bufs.push_back(dst->handle());
                bufs.push_back(planBuf->handle());
                for (int k = 0; k < kPwMaxOperands; ++k)
                {
                    if (k < (int) operands.size())
                    {
                        bufs.push_back(pwOperandBuf(env, operands[k], holds[k], flat)->handle());
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
