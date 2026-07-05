// ONNX Reshape: reinterpret `data` (inputs[0]) under the target shape given by the int64 shape
// tensor `shape` (inputs[1]); the element sequence is preserved verbatim (row-major), so this is a
// pure metadata/copy op that never touches values. Two shape entries carry special meaning:
//   0  copies the input dim at the same position (per the ONNX default allowzero=0), falling back
//      to 1 when `data` has no dim there.
//   -1 marks the single axis whose extent is inferred so the total element count is preserved;
//      at most one -1 may appear.
// The output element count always equals X.elems(); dtype is carried through unchanged, so every
// dtype is supported.
#include "backend/cpu/cpu_backend.h"
#include <algorithm>

namespace vknn {
    namespace {

        struct ReshapeCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X    = ctx.t(node.inputs[0]);
                const RtTensor &S    = ctx.t(node.inputs[1]);
                RtTensor       &Y    = ctx.t(node.outputs[0]);
                // The shape tensor's element count is the output rank; its int64 entries are the
                // target dims. `known` accumulates the product of every fixed dim so a lone -1 can be
                // solved for later; `inferIdx` records that -1's position (-1 = none).
                int64_t         rank = S.elems();
                const int64_t  *sd   = S.host.i64();
                Shape           out(rank);
                int64_t         known = 1, inferIdx = -1;
                for (int64_t i = 0; i < rank; ++i)
                {
                    int64_t d = sd[i];
                    if (d == 0)
                    {
                        // 0 = "keep the input dim at this position"; positions past X's rank yield 1.
                        d = (i < (int64_t) X.shape.size()) ? X.shape[i] : 1;
                    }
                    out[i] = d;
                    if (d == -1)
                    {
                        inferIdx = i;
                    } else
                    {
                        known *= d;
                    }
                }
                if (inferIdx >= 0)
                {
                    // Solve the inferred axis so the dims multiply back to the input element count.
                    // max(known,1) guards the degenerate case where all other dims are 0.
                    out[inferIdx] = X.elems() / std::max<int64_t>(known, 1);
                }
                // Reshape moves no data: copy `X` verbatim under the resolved shape, preserving dtype.
                cpu::copyAs(X, Y, out);
            }
            bool supportsDType(DType) const override {
                return true;
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Reshape, ReshapeCpu);
} // namespace vknn
