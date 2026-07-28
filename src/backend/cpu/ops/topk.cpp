// ONNX TopK: the k largest (largest=1, default) or smallest (largest=0) elements along `axis`
// (default -1), emitted as two outputs — values (input dtype) and their source indices (int64).
// k is the `k` attribute (opset < 10) or the scalar int64 input[1] (opset 10+) and is clamped to
// the axis length. Equal values keep ascending source indices (the ONNX tie rule), implemented by
// breaking comparison ties on the index, which makes the partial sort order-stable. sorted=0
// leaves the order unspecified by ONNX; this kernel emits the same sorted order as sorted=1, so
// the output is deterministic either way. Handles fp32 and int64 data (the int64 path serves
// const-folded shape/index arithmetic).
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <algorithm>
#include <numeric>

namespace vknn {
    namespace {
        struct TopKCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X    = ctx.t(node.inputs[0]);
                RtTensor       &V    = ctx.t(node.outputs[0]);
                int64_t         rank = (int64_t) X.shape.size();
                if (rank == 0)
                {
                    return; // rank-0 input has no axis to select along
                }
                int64_t axis = node.attr.geti("axis", -1);
                if (axis < 0)
                {
                    axis += rank;
                }
                axis        = std::max<int64_t>(0, std::min(axis, rank - 1));
                int64_t dim = X.shape[axis];
                // k: attribute form first (opset < 10), else the scalar int64 input[1] (opset 10+).
                // The positional read is bounded by pwCoreInputs so a fused-chain operand appended
                // past it is never misread as k.
                int64_t k = -1;
                if (node.attr.has("k"))
                {
                    k = node.attr.geti("k", -1);
                } else if ((int) pwCoreInputs(node) > 1 && node.inputs[1] != kNoTensor)
                {
                    const RtTensor &K = ctx.t(node.inputs[1]);
                    if (K.elems() > 0)
                    {
                        k = K.dtype == DType::Int64 ? K.host.i64()[0] : (int64_t) K.host.f32()[0];
                    }
                }
                k            = std::max<int64_t>(0, std::min(k, dim));
                bool largest = node.attr.geti("largest", 1) != 0;
                // Output shape = input with the selected axis shortened to k. Slices along `axis`
                // are strided walks: element j of slice (o, i) sits at (o*dim + j)*inner + i.
                Shape out     = X.shape;
                out[axis]     = k;
                int64_t outer = 1, inner = 1;
                for (int64_t i = 0; i < axis; ++i)
                {
                    outer *= X.shape[i];
                }
                for (int64_t i = axis + 1; i < rank; ++i)
                {
                    inner *= X.shape[i];
                }
                bool           i64 = X.dtype == DType::Int64;
                const float   *xf  = i64 ? nullptr : X.host.f32();
                const int64_t *xi  = i64 ? X.host.i64() : nullptr;
                float         *vf  = i64 ? nullptr : cpu::allocOut(V, out);
                int64_t       *vi  = i64 ? cpu::allocOutI64(V, out) : nullptr;
                int64_t       *ind = nullptr;
                if (node.outputs.size() > 1 && node.outputs[1] != kNoTensor)
                {
                    ind = cpu::allocOutI64(ctx.t(node.outputs[1]), out);
                }
                std::vector<int64_t> idx(dim);
                for (int64_t o = 0; o < outer; ++o)
                {
                    for (int64_t in = 0; in < inner; ++in)
                    {
                        int64_t base = o * dim * inner + in;
                        // Rank the axis indices by value; a tie falls through to the index compare,
                        // so equal values come out in ascending source order (the ONNX rule) and the
                        // partial sort is effectively stable.
                        std::iota(idx.begin(), idx.end(), (int64_t) 0);
                        auto cmp = [&](int64_t a, int64_t b) {
                            if (i64)
                            {
                                int64_t va = xi[base + a * inner], vb = xi[base + b * inner];
                                if (va != vb)
                                {
                                    return largest ? va > vb : va < vb;
                                }
                            } else
                            {
                                float va = xf[base + a * inner], vb = xf[base + b * inner];
                                if (va != vb)
                                {
                                    return largest ? va > vb : va < vb;
                                }
                            }
                            return a < b;
                        };
                        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), cmp);
                        int64_t obase = (o * k) * inner + in;
                        for (int64_t j = 0; j < k; ++j)
                        {
                            if (i64)
                            {
                                vi[obase + j * inner] = xi[base + idx[j] * inner];
                            } else
                            {
                                vf[obase + j * inner] = xf[base + idx[j] * inner];
                            }
                            if (ind)
                            {
                                ind[obase + j * inner] = idx[j];
                            }
                        }
                    }
                }
            }
        };
    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::TopK, TopKCpu);
} // namespace vknn
