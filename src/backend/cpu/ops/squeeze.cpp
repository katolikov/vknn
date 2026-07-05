// Squeeze: remove size-1 dims (given `axes`, or all size-1 dims when axes is absent). Pure metadata
// reshape - data is untouched, so it's a byte copy with a new shape. axes from the attribute
// (opset<13) or the int64 input[1] (opset>=13). dtype-agnostic.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <algorithm>

namespace vknn {
    namespace {

        struct SqueezeCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor      &X    = ctx.t(node.inputs[0]);
                RtTensor            &Y    = ctx.t(node.outputs[0]);
                int                  rank = (int) X.shape.size();
                std::vector<int64_t> axes = node.attr.getints("axes");
                if (axes.empty() && node.inputs.size() > 1 && node.inputs[1] != kNoTensor)
                {
                    const RtTensor &A = ctx.t(node.inputs[1]);
                    axes.assign(A.host.i64(), A.host.i64() + A.elems());
                }
                // `drop[k]` marks axis k for removal; the surviving axes keep their original order.
                std::vector<bool> drop(rank, false);
                if (axes.empty())
                {
                    for (int k = 0; k < rank; ++k)
                    {
                        drop[k] = (X.shape[k] == 1); // no axes given: remove every size-1 dim
                    }
                } else
                {
                    for (int64_t ax: axes)
                    {
                        if (ax < 0)
                        {
                            ax += rank; // ONNX allows negative axes, counted from the end
                        }
                        // Out-of-range axes are silently skipped rather than clamped, so a stale index
                        // never marks an unintended dim; a listed axis whose size is not 1 is still
                        // dropped here (ONNX leaves that case to the shape checker, not this kernel).
                        if (ax >= 0 && ax < rank)
                        {
                            drop[ax] = true;
                        }
                    }
                }
                Shape out;
                for (int k = 0; k < rank; ++k)
                {
                    if (!drop[k])
                    {
                        out.push_back(X.shape[k]);
                    }
                }
                if (out.empty())
                {
                    out.push_back(1); // every dim dropped (rank-0 result): represent the scalar as [1]
                }
                // Element count is unchanged, so this is a byte copy of X's data under the new shape.
                cpu::copyAs(X, Y, out);
            }
            bool supportsDType(DType) const override {
                return true;
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Squeeze, SqueezeCpu);
} // namespace vknn
