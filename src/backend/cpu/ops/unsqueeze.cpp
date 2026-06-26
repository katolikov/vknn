// Unsqueeze: insert size-1 dims at the given axes (opset<13 axes attr, opset>=13 axes input).
// Data is untouched - only the shape changes.
#include "backend/cpu/cpu_backend.h"
#include <algorithm>

namespace vknn {
    namespace {

        struct UnsqueezeCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor      &X    = ctx.t(node.inputs[0]);
                RtTensor            &Y    = ctx.t(node.outputs[0]);
                std::vector<int64_t> axes = node.attr.getints("axes");
                if (axes.empty() && node.inputs.size() > 1 && node.inputs[1] != kNoTensor)
                {
                    const RtTensor &A = ctx.t(node.inputs[1]);
                    for (int64_t i = 0; i < A.elems(); ++i)
                    {
                        axes.push_back(A.host.i64()[i]);
                    }
                }
                Shape out = X.shape;
                std::sort(axes.begin(), axes.end());
                for (int64_t ax: axes)
                {
                    if (ax < 0)
                    {
                        ax += (int64_t) out.size() + 1;
                    }
                    out.insert(out.begin() + std::min<int64_t>(ax, out.size()), 1);
                }
                cpu::copyAs(X, Y, out);
            }
            bool supportsDType(DType) const override {
                return true;
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::kUnsqueeze, UnsqueezeCpu);
} // namespace vknn
