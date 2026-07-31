// Standalone FusedPointwise: dispatch the pw_steps/pw_params chain (produced by
// fusePointwiseChains) as one kernel instead of one dispatch per original op. The plan (steps +
// broadcast strides) is uploaded once as a small SSBO read by shaders/pw_epilogue.glsl; the extra
// step operands bind at consecutive slots after it (see PW_EPI_BASE in the .comp files).
#include "backend/vulkan/vk_tune_model.h"
#include "backend/vulkan/vk_tune_race.h"
#include "flat_ops.h"
#include "pw_plan.h"
#include "vk_op_common.h"
#include "vknn/logging.h"
#include "vknn/op.h"
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace vknn {
    namespace {
        // Items-per-lane and its device-derived lane floor live in flat::itemsPerLane
        // (flat_ops.h) - shared with the movement kernels, resolved at load.

        // Elements per lane-quad of the vectorized flat kernel; mirrors PW_FLAT_QUAD in
        // shaders/fused_pw_flat_v4.comp.
        constexpr int kPwFlatQuad = 4;
        // Tune-table values of the flat-kernel pick (append-only).
        constexpr int kPwFlatKernelScalar = 0; ///< fused_pw_flat: one element per lane step.
        constexpr int kPwFlatKernelQuad   = 1; ///< fused_pw_flat_v4: one whole-vec4 quad per lane step.

        struct FusedPointwiseOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline>     pipe;
            std::shared_ptr<vk::Buffer>              planBuf;
            std::vector<TensorId>                    operands;
            std::vector<std::shared_ptr<vk::Buffer>> holds;
            int                                      total = 0;
            int                                      items = 1; // per-lane walk count, resolved at load in prepare()
            int                                      units = 0; // lane-step count the grid covers: total (scalar) or quads (vec4)
            bool                                     flat  = false;

            // The scalar and quad flat kernels are byte-interchangeable: identical per-element
            // values, op order and rounding — the quad twin only widens the load/store
            // transactions to whole vec4s over four consecutive elements. The pick is therefore
            // placement-only and follows the standard cached race: Tuning::None keeps the scalar
            // kernel (the deterministic default), Fast/Heavy race both once on dedicated scratch
            // through TuneTimer/raceCandidates and persist the winner; the quad challenger must
            // clear vk::kTuneRaceMargin to displace the incumbent.
            bool pickFlatQuad(VkOpEnv &env, const std::vector<uint32_t> &spec, bool relax, int numSteps) {
                char buf[96];
                snprintf(buf, sizeof(buf), "pwflat_%d_%d_%d_%d_%d", total, numSteps, (int) operands.size(), relax ? 1 : 0, env.useFp16 ? 1 : 0);
                std::string sig = env.gpuTag + "/" + buf;
                int         reuse;
                if (env.reuseTuned(sig, reuse) && (reuse == kPwFlatKernelScalar || reuse == kPwFlatKernelQuad))
                {
                    return reuse == kPwFlatKernelQuad;
                }
                if (env.tuning == Tuning::None || !env.runner)
                {
                    return false;
                }
                // Dedicated scratch: entry/output/operand-filler buffers of the unit's element
                // count (the epilogue indexes operands by output element, so the filler must be at
                // least output-sized — the appendForTiming rule). The real plan SSBO makes the
                // race walk the unit's actual classes and step chain.
                const size_t es = (size_t) (env.useFp16 ? 2 : 4);
                auto         mk = [&](size_t bytes) {
                    return std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>(bytes, 16), vk::MemPref::kDeviceOnly);
                };
                auto          sPrim = mk((size_t) total * es);
                auto          sDst  = mk((size_t) total * es);
                auto          sFill = mk((size_t) total * es);
                vk::TuneTimer timer(env);
                const int     quadTotal   = (total + kPwFlatQuad - 1) / kPwFlatQuad;
                const int     itemsScalar = flat::itemsPerLane(total, env);
                const int     itemsQuad   = flat::itemsPerLane(quadTotal, env);
                auto          ms          = vk::raceCandidates(2, [&](int index) {
                    const bool  quad = index == kPwFlatKernelQuad;
                    std::string base = quad ? "fused_pw_flat_v4" : "fused_pw_flat";
                    if (relax)
                    {
                        base += "_rx";
                    }
                    auto pipe = env.pipeline(shader(base.c_str(), env.useFp16), 2 + 1 + kPwMaxOperands + kPwMaxOuts, sizeof(int) * 2, spec);
                    struct {
                        int total, items;
                    } rpc {total, quad ? itemsQuad : itemsScalar};
                    const int laneSteps = quad ? quadTotal : total;
                    const int lanes     = (laneSteps + rpc.items - 1) / rpc.items;
                    return timer.time([&](VkCommandBuffer cmd) {
                        std::vector<VkBuffer> bufs {sPrim->handle(), sDst->handle(), planBuf->handle()};
                        for (int k = 0; k < kPwMaxOperands + kPwMaxOuts; ++k)
                        {
                            bufs.push_back(sFill->handle());
                        }
                        pipe->dispatch(cmd, bufs, &rpc, sizeof(rpc), groups(lanes, env.flatLocalSize));
                    });
                });
                const bool    quad        = ms.size() == 2 && ms[kPwFlatKernelQuad] < ms[kPwFlatKernelScalar] * vk::kTuneRaceMargin;
                if (env.weights)
                {
                    env.weights->setTuned(sig, quad ? kPwFlatKernelQuad : kPwFlatKernelScalar, (int) env.tuning);
                }
                VKNN_DEBUG << "autotune " << sig << " -> " << (quad ? "quad" : "scalar") << vk::raceTimes(ms);
                return quad;
            }

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g = *env.graph;
                flat           = opIsFlat(node, env);
                Shape out      = g.desc(node.outputs[0]).shape;

                PwPlanCPU plan {};
                buildPwPlan(g, node, flat, out, plan, operands, total);
                holds.assign(operands.size(), nullptr);

                planBuf = uploadPwPlan(env, plan);
                // Chains of up to 8 steps compile a monomorphized pipeline: 25 spec words
                // {NS, K[0..7] = (kind << 16) | (code & 0xffff), p0[0..7] bits, p1[0..7] bits} bind
                // constant_id 0..24 in the shader, which unrolls the step loop and folds the kind
                // dispatch and params at pipeline creation. Units with identical steps share one
                // pipeline (the session pool keys on the spec words). Longer chains — and an empty
                // spec vector — leave the shader's NS = 0 default in place, selecting the runtime
                // plan-SSBO interpreter.
                // 26 words always: 0..24 carry the plan (all-zero = NS 0 = the interpreter, the
                // same selection an absent spec made before), 25 carries the device-resolved
                // workgroup width the shaders declare as local_size_x_id = 25.
                std::vector<uint32_t> spec(26, 0u);
                spec[25] = env.flatLocalSize;
                if (plan.numSteps >= 1 && plan.numSteps <= 8)
                {
                    spec[0] = (uint32_t) plan.numSteps;
                    for (int s = 0; s < plan.numSteps; ++s)
                    {
                        spec[1 + s] = ((uint32_t) plan.step[s * 8] << 16) | ((uint32_t) plan.step[s * 8 + 1] & 0xffffu);
                        std::memcpy(&spec[9 + s], &plan.p0[s], sizeof(uint32_t));
                        std::memcpy(&spec[17 + s], &plan.p1[s], sizeof(uint32_t));
                    }
                }
                // Buffer count = 2 (primary input + output) + 1 (plan SSBO) + kPwMaxOperands operand
                // slots + kPwMaxOuts extra output streams. The shaders (fused_pw_flat/nc4.comp)
                // statically declare every slot via pw_epilogue.glsl, so the descriptor set must
                // always size for the full count even when the plan uses fewer. The push constant is
                // two ints (the element count and the items-per-lane rule), matching `PC { int total, items; }`.
                // The rounding discipline is compiled in: "_rx" = fp32-chained (pw_relax units),
                // base name = strict per-step-rounded.
                bool       relax = node.attr.geti("pw_relax", 0) != 0;
                const bool quad  = flat && pickFlatQuad(env, spec, relax, plan.numSteps);
                units            = quad ? (total + kPwFlatQuad - 1) / kPwFlatQuad : total;
                items            = flat::itemsPerLane(units, env); // device-probe consult happens at load, never at record
                std::string base = flat ? (quad ? "fused_pw_flat_v4" : "fused_pw_flat") : "fused_pw_nc4";
                if (relax)
                {
                    base += "_rx";
                }
                pipe = env.pipeline(shader(base.c_str(), env.useFp16), 2 + 1 + kPwMaxOperands + kPwMaxOuts, sizeof(int) * 2, spec);
                // Which of the two appliers this unit lands on is the difference between an unrolled
                // chain with its params folded at pipeline creation and a per-element walk of the
                // plan SSBO, so it is the first thing to know about a unit that costs more than its
                // memory traffic.
                VKNN_DEBUG << "FusedPointwise '" << node.name << "': " << plan.numSteps << " step(s), " << operands.size() << " operand(s), " << (flat ? (quad ? "flat-quad" : "flat") : "NC4HW4") << ", " << (plan.numSteps >= 1 && plan.numSteps <= 8 ? "monomorphized" : "plan-SSBO interpreter") << ", " << total << " lanes";
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
                // Push constants: the element count guarding the grid and the per-lane walk; the
                // workgroup width matches spec word 25 set at prepare. `units` is the lane-step
                // count the grid covers — elements for the scalar kernels, whole quads for the
                // vectorized flat kernel (which re-derives its quad count from pc.total).
                struct {
                    int total, items;
                } pc {total, items};
                const int lanes = (units + pc.items - 1) / pc.items;
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(lanes, env.flatLocalSize));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::FusedPointwise, FusedPointwiseOp);
} // namespace vknn
