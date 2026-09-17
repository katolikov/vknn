// ArgMax / ArgMin on the GPU (flat row-major path), shared by ops/argmax.cpp and ops/argmin.cpp: one
// shaders/arg_extreme.comp invocation per output element scans its axis slice and writes the int64
// index as an fp32 value (exact over every axis the plan accepts). The plan (arg_extreme_plan.h)
// decides the geometry, the data-precision variant and the preconditions; the op builds the pipeline
// with ArgMax-vs-ArgMin and select_last_index as specialization constants and dispatches.
#pragma once
#include "arg_extreme_plan.h"
#include "flat_ops.h"
#include "vk_op_common.h"
#include <algorithm>

namespace vknn {

    class ArgExtremeVk: public VulkanOp {
      public:
        explicit ArgExtremeVk(bool selectLargest): selectLargest_(selectLargest) {
        }

        void prepare(const Node &node, VkOpEnv &env) override {
            const Graph &g = *env.graph;
            plan_          = planArgExtreme(g, node, env.baseFp16);
            if (plan_.constantData)
            {
                // A constant data operand uploads fp32 to device-only memory (the plan then selects the
                // fp32 variant), decoded from any initializer dtype the way the CPU oracle reads it. The
                // plan has verified the payload still holds every element.
                static constexpr bool kUploadFp16 = false;
                std::vector<float>    values      = initFloats(g, node.inputs[0]);
                values.resize((size_t) std::max<int64_t>(1, numElements(g.desc(node.inputs[0]).shape)));
                constantData_ = uploadWeight(env, values, kUploadFp16);
            }
            pipe_ = env.pipeline(shader(kArgExtremeShaderStem, plan_.dataFp16), kArgExtremeBufferCount, sizeof(ArgExtremePushConstants),
                                 argExtremeSpecConstants(selectLargest_, plan_.selectLast));
        }

        void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
            vk::Buffer *data = constantData_ ? constantData_.get() : env.devBuf(node.inputs[0]);
            // One invocation per output element; arg_extreme.comp is local_size_x=256 == flat::kFlatLocalSize.
            pipe_->dispatch(cmd, {data->handle(), env.devBuf(node.outputs[0])->handle()}, &plan_.push, sizeof(plan_.push), groups(plan_.push.total, flat::kFlatLocalSize));
        }

      private:
        bool                                 selectLargest_;
        ArgExtremePlan                       plan_;
        std::shared_ptr<vk::ComputePipeline> pipe_;
        std::shared_ptr<vk::Buffer>          constantData_;
    };

} // namespace vknn
