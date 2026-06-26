// Tile (ONNX): repeat X `repeats[k]` times along each axis (repeats is an int64 input, len ==
// rank). out.shape[k] = X.shape[k] * repeats[k]; out[i] = in[ sum_k (outCoord_k % inDim_k) *
// inStride_k ]. dtype-agnostic.
#include "backend/cpu/cpu_backend.h"
#include "import/passes.h" // readI64Param
#include "vknn/op.h"
#include <algorithm>

namespace vknn {
    namespace {

        struct TileCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor      &X    = ctx.t(node.inputs[0]);
                RtTensor            &Y    = ctx.t(node.outputs[0]);
                const Shape         &in   = X.shape;
                int                  rank = (int) in.size();
                std::vector<int64_t> reps = readI64Param(*ctx.graph, node, "repeats", 1);
                if (reps.empty() && node.inputs.size() > 1 && node.inputs[1] != kNoTensor)
                {
                    const RtTensor &R = ctx.t(node.inputs[1]);
                    reps.assign(R.host.i64(), R.host.i64() + R.elems());
                }
                Shape out(rank);
                for (int k = 0; k < rank; ++k)
                {
                    int64_t r = (k < (int) reps.size()) ? reps[k] : 1;
                    out[k]    = in[k] * std::max<int64_t>(r, 1);
                }
                std::vector<int64_t> inStride(rank, 1), outStride(rank, 1);
                for (int k = rank - 2; k >= 0; --k)
                {
                    inStride[k]  = inStride[k + 1] * in[k + 1];
                    outStride[k] = outStride[k + 1] * out[k + 1];
                }
                int64_t        elems = numElements(out);
                bool           i64   = X.dtype == DType::kInt64;
                const float   *xf    = i64 ? nullptr : X.host.f32();
                const int64_t *xi    = i64 ? X.host.i64() : nullptr;
                float         *yf    = i64 ? nullptr : cpu::allocOut(Y, out);
                int64_t       *yi    = i64 ? cpu::allocOutI64(Y, out) : nullptr;
                for (int64_t oi = 0; oi < elems; ++oi)
                {
                    int64_t rem = oi, inf = 0;
                    for (int k = 0; k < rank; ++k)
                    {
                        int64_t c = rem / outStride[k];
                        rem %= outStride[k];
                        inf += (c % in[k]) * inStride[k];
                    }
                    if (i64)
                    {
                        yi[oi] = xi[inf];
                    } else
                    {
                        yf[oi] = xf[inf];
                    }
                }
            }
            bool supportsDType(DType) const override {
                return true;
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::kTile, TileCpu);
} // namespace vknn
