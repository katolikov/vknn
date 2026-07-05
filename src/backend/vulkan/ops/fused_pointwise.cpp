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
                // Buffer count = 2 (primary input + output) + 1 (plan SSBO) + kPwMaxOperands operand
                // slots + kPwMaxOuts extra output streams. The shaders (fused_pw_flat/nc4.comp)
                // statically declare every slot via pw_epilogue.glsl, so the descriptor set must
                // always size for the full count even when the plan uses fewer. Push constant is a
                // single int (the element count `total`), matching the shaders' `PC { int total; }`.
                pipe = env.pipeline(shader(flat ? "fused_pw_flat" : "fused_pw_nc4", env.useFp16), 2 + 1 + kPwMaxOperands + kPwMaxOuts, sizeof(int), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                // Binding order matches the shaders: 0 = primary input, 1 = output, 2 = plan SSBO
                // (PW_EPI_BASE), the kPwMaxOperands step-operand slots (PW_EPI_BASE+1..), then the
                // kPwMaxOuts extra output streams (PW_EPI_BASE+1+kPwMaxOperands..). Unused slots bind
                // `dst` as a harmless placeholder: the plan never references them, so the kernel
                // never reads (or writes) them.
                vk::Buffer           *dst = env.devBuf(node.outputs[0]);
                std::vector<VkBuffer> bufs;
                bufs.push_back(env.devBuf(node.inputs[0])->handle());
                bufs.push_back(dst->handle());
                bufs.push_back(planBuf->handle());
                for (int k = 0; k < kPwMaxOperands; ++k)
                {
                    bufs.push_back(k < (int) operands.size() ? pwOperandBuf(env, operands[k], holds[k], flat)->handle() : dst->handle());
                }
                for (int o = 0; o < kPwMaxOuts; ++o)
                {
                    bool live = 1 + o < (int) node.outputs.size() && node.outputs[1 + o] != kNoTensor;
                    bufs.push_back(live ? env.devBuf(node.outputs[1 + o])->handle() : dst->handle());
                }
                // One int push constant: the element count guarding the 1D grid (local_size_x=256).
                int pc = total;
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(total, 256));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::FusedPointwise, FusedPointwiseOp);
} // namespace vknn
