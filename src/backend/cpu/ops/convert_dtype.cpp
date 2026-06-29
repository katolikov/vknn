// ConvertDtype on the CPU backend: host tensors are always fp32, so the fp16<->fp32 storage convert
// (a Vulkan-only concern) is an identity copy here. Present so a graph with selective-fp32 markers
// still runs on / falls back to CPU.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        struct ConvertDtypeCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X = ctx.t(node.inputs[0]);
                RtTensor       &Y = ctx.t(node.outputs[0]);
                float          *y = cpu::allocOut(Y, X.shape);
                const float    *x = X.host.f32();
                int64_t         n = X.elems();
                for (int64_t i = 0; i < n; ++i)
                {
                    y[i] = x[i];
                }
            }
            bool supportsDType(DType) const override {
                return true;
            }
        };
    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::ConvertDtype, ConvertDtypeCpu);
} // namespace vknn
