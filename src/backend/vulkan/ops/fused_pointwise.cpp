// Standalone FusedPointwise: dispatch the pw_steps/pw_params chain (produced by
// fusePointwiseChains) as one kernel instead of one dispatch per original op. The plan (steps +
// broadcast strides) is uploaded once as a small SSBO read by shaders/pw_epilogue.glsl; the extra
// step operands bind at consecutive slots after it (see PW_EPI_BASE in the .comp files).
#include "backend/vulkan/vk_tune_model.h"
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
        // Elements each lane processes, one slot apart (pc.items in both fused_pw_*.comp), as a
        // deterministic shape rule -- the map is a pure placement choice, identical values either way.
        //
        // Several items per lane cover the applier's memory latency, which is what bounds a big unit.
        // A SMALL unit has the opposite problem: dividing its lane count by the item count leaves too
        // few workgroups to fill the device, and the latency it was hiding is replaced by idle cores.
        // Measured: an image-resolution unit is -21% at eight items, while a classifier's late 7x7
        // units (a few thousand lanes) are +6% -- so the rule is simply whether enough lanes survive
        // the division.
        //
        // The lane floor is DEVICE-DERIVED at load, not a constant: it is the saturation point the
        // TuneModel probe measures on this GPU (deviceTuneModel; 64-wide waves before added waves
        // stop buying throughput) times a fixed headroom multiple, in lanes. A wider GPU keeps more
        // items per lane profitable exactly that much longer; a narrower one drops to one item
        // sooner instead of idling. The multiple reproduces the measured -21% choice on the
        // calibration device (5 x 400 waves x 64 lanes ~= the 128 Ki-lane floor it was tuned at).
        constexpr int kPwMaxItemsPerLane      = 8; // independent loads one lane keeps in flight
        constexpr int kPwLaneFloorWaveHeadroom = 5;
        constexpr int kPwProbeWaveLanes        = 64; // the probe counts 64-wide waves

        // Items per lane for a dispatch of `total` elements: as many as kPwMaxItemsPerLane, while
        // leaving at least the device's saturation lane count.
        inline int pwItemsPerLane(int total, VkOpEnv &env) {
            const double waves     = vk::deviceTuneModel(env).wavesToSaturate;
            const int    laneFloor = (int) (waves * kPwProbeWaveLanes) * kPwLaneFloorWaveHeadroom;
            const int    byFloor   = laneFloor > 0 ? total / laneFloor : 1;
            return byFloor < 1 ? 1 : (byFloor > kPwMaxItemsPerLane ? kPwMaxItemsPerLane : byFloor);
        }

        struct FusedPointwiseOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline>     pipe;
            std::shared_ptr<vk::Buffer>              planBuf;
            std::vector<TensorId>                    operands;
            std::vector<std::shared_ptr<vk::Buffer>> holds;
            int                                      total = 0;
            int                                      items = 1; // per-lane walk count, resolved at load in prepare()
            bool                                     flat  = false;

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g = *env.graph;
                flat           = opIsFlat(node, env);
                Shape out      = g.desc(node.outputs[0]).shape;

                PwPlanCPU plan {};
                buildPwPlan(g, node, flat, out, plan, operands, total);
                items = pwItemsPerLane(total, env); // device-probe consult happens at load, never at record
                holds.assign(operands.size(), nullptr);

                planBuf = uploadPwPlan(env, plan);
                // Chains of up to 8 steps compile a monomorphized pipeline: 25 spec words
                // {NS, K[0..7] = (kind << 16) | (code & 0xffff), p0[0..7] bits, p1[0..7] bits} bind
                // constant_id 0..24 in the shader, which unrolls the step loop and folds the kind
                // dispatch and params at pipeline creation. Units with identical steps share one
                // pipeline (the session pool keys on the spec words). Longer chains — and an empty
                // spec vector — leave the shader's NS = 0 default in place, selecting the runtime
                // plan-SSBO interpreter.
                std::vector<uint32_t> spec;
                if (plan.numSteps >= 1 && plan.numSteps <= 8)
                {
                    spec.assign(25, 0u);
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
                bool        relax = node.attr.geti("pw_relax", 0) != 0;
                std::string base  = flat ? "fused_pw_flat" : "fused_pw_nc4";
                if (relax)
                {
                    base += "_rx";
                }
                pipe = env.pipeline(shader(base.c_str(), env.useFp16), 2 + 1 + kPwMaxOperands + kPwMaxOuts, sizeof(int) * 2, spec);
                // Which of the two appliers this unit lands on is the difference between an unrolled
                // chain with its params folded at pipeline creation and a per-element walk of the
                // plan SSBO, so it is the first thing to know about a unit that costs more than its
                // memory traffic.
                VKNN_DEBUG << "FusedPointwise '" << node.name << "': " << plan.numSteps << " step(s), " << operands.size() << " operand(s), " << (flat ? "flat" : "NC4HW4") << ", " << (spec.empty() ? "plan-SSBO interpreter" : "monomorphized") << ", " << total << " lanes";
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
                // One int push constant: the element count guarding the 1D grid. The fused_pw_flat/nc4
                // shaders are local_size_x=256 == flat::kFlatLocalSize.
                struct {
                    int total, items;
                } pc {total, items};
                // Both kernels walk pc.items elements per lane (see their main()), so the grid covers
                // that fraction of the element count.
                const int lanes = (total + pc.items - 1) / pc.items;
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(lanes, flat::kFlatLocalSize));
            }
        };
    } // namespace
    VKNN_REGISTER_VK_OP(OpType::FusedPointwise, FusedPointwiseOp);
} // namespace vknn
