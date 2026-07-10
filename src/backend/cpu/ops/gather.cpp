// ONNX Gather along an arbitrary axis: out = data[:axis] + indices.shape + data[axis+1:]. A scalar
// index (rank-0, stored here as [1] with one element) removes the gathered axis. Used both for
// classifier-preamble shape math (axis-0 vectors) and the transformer attention Q/K/V split, which
// gathers a single index along axis 2 of the permuted qkv tensor; honoring `axis` (not assuming
// axis 0) is required for correctness.
#include "backend/cpu/cpu_backend.h"
#include <algorithm>

namespace vknn {
    namespace {

        struct GatherCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &D    = ctx.t(node.inputs[0]);
                const RtTensor &I    = ctx.t(node.inputs[1]);
                RtTensor       &Y    = ctx.t(node.outputs[0]);
                int64_t         rank = (int64_t) D.shape.size();
                int64_t         axis = node.attr.geti("axis", 0);
                if (axis < 0)
                {
                    axis += rank;
                }
                axis = std::max<int64_t>(0, std::min<int64_t>(axis, rank > 0 ? rank - 1 : 0));

                int64_t nidx = cpu::elemCount(I.shape); // a rank-0 scalar index selects its one row
                // Index dtype varies: const int64 (attention Q/K/V) or a runtime float activation (RoPE).
                auto indexAt = [&](int64_t k) -> int64_t {
                    return I.dtype == DType::Int64 ? I.host.i64()[k] : (int64_t) I.host.f32()[k];
                };
                // Collapse `data` into a 3-way [outer, axisSize, inner] view around `axis`: `outer` is
                // the product of the dims before `axis`, `inner` the product of the dims after it (both
                // contiguous in row-major order). Gather then walks `outer` blocks, selecting `nidx`
                // rows of `inner` contiguous elements each, so the whole op reduces to flat row copies.
                int64_t axisSize = (rank > 0) ? D.shape[axis] : 1;
                int64_t outer    = 1;
                for (int64_t i = 0; i < axis; ++i)
                {
                    outer *= D.shape[i];
                }
                int64_t inner = 1;
                for (int64_t i = axis + 1; i < rank; ++i)
                {
                    inner *= D.shape[i];
                }

                // Scalar index (rank-0, or the importer's [1]-of-1 form) removes the axis; otherwise the
                // indices' own shape is spliced in at `axis`.
                bool  scalarIndex = I.shape.empty() || node.attr.geti("idx_scalar", 0) != 0; // only a true rank-0 index removes the axis
                Shape outShape;
                for (int64_t i = 0; i < axis; ++i)
                {
                    outShape.push_back(D.shape[i]);
                }
                if (!scalarIndex)
                {
                    for (int64_t v: I.shape)
                    {
                        outShape.push_back(v);
                    }
                }
                for (int64_t i = axis + 1; i < rank; ++i)
                {
                    outShape.push_back(D.shape[i]);
                }
                if (outShape.empty())
                {
                    outShape = {1};
                }

                // Validate the whole index tensor in one O(nidx) pass before any row copy, so the copy
                // loops below stay free of per-row bounds checks. An out-of-range index (an out-of-vocab
                // token id against an embedding table) is a hard error naming the value, its position,
                // and the valid range — never an out-of-bounds read.
                for (int64_t k = 0; k < nidx; ++k)
                {
                    int64_t ik = indexAt(k);
                    if (ik < -axisSize || ik >= axisSize)
                    {
                        throw Error(Status::InvalidArgument, "Gather '" + node.name + "': index " + std::to_string(ik) + " at position " + std::to_string(k) + " is out of range [" + std::to_string(-axisSize) + ", " + std::to_string(axisSize) + ") for axis " + std::to_string(axis) + " of size " + std::to_string(axisSize));
                    }
                }

                // Copy each gathered row into the output. Source row (o, src) starts at flat offset
                // (o*axisSize + src)*inner; the k-th output row of block `o` at (o*nidx + k)*inner, since
                // the output replaces `axisSize` with `nidx` along the collapsed axis. A negative index
                // wraps by adding `axisSize`, matching ONNX's Python-style indexing.
                auto copy = [&](auto *y, const auto *d) {
                    for (int64_t o = 0; o < outer; ++o)
                    {
                        for (int64_t k = 0; k < nidx; ++k)
                        {
                            int64_t     ik  = indexAt(k);
                            int64_t     src = ik < 0 ? ik + axisSize : ik;
                            const auto *sp  = d + (o * axisSize + src) * inner;
                            auto       *dp  = y + (o * nidx + k) * inner;
                            for (int64_t j = 0; j < inner; ++j)
                            {
                                dp[j] = sp[j];
                            }
                        }
                    }
                };
                if (D.dtype == DType::Int64)
                {
                    int64_t *y = cpu::allocOutI64(Y, outShape);
                    copy(y, D.host.i64());
                } else
                {
                    float *y = cpu::allocOut(Y, outShape);
                    copy(y, D.host.f32());
                }
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Gather, GatherCpu);
} // namespace vknn
