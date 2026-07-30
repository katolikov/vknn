// Transpose / Permute (generic N-D, dtype-agnostic) — the CPU ORACLE for the GPU flat_gather
// kernel (transpose_slice.cpp); runs in canonical NCHW. Not a fallback in any real scenario.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        struct TransposeCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X    = ctx.t(node.inputs[0]);
                RtTensor       &Y    = ctx.t(node.outputs[0]);
                int             rank = (int) X.shape.size();
                if (node.attr.has("view_stride"))
                {
                    cpu::runViewGather(node, ctx);
                    return; // folded movement chain: the composed map replaces the perm entirely
                }
                std::vector<int64_t> perm = node.attr.getints("perm");
                // ONNX default when `perm` is absent (or malformed): reverse the axis order, so output
                // axis i draws from input axis rank-1-i.
                if ((int) perm.size() != rank)
                {
                    perm.clear();
                    for (int i = rank - 1; i >= 0; --i)
                    {
                        perm.push_back(i);
                    }
                }
                // Normalize negative axes (Python-style, counted from the end) to their [0, rank) form.
                for (auto &p: perm)
                {
                    if (p < 0)
                    {
                        p += rank;
                    }
                }
                // Output shape is the input shape read through the permutation: out[i] = X.shape[perm[i]].
                Shape out(rank);
                for (int i = 0; i < rank; ++i)
                {
                    out[i] = X.shape[perm[i]];
                }
                // Row-major (contiguous) strides for the input and output tensors: the stride of an axis
                // is the product of all dimension sizes to its right, so the last axis has stride 1.
                std::vector<int64_t> inStride(rank, 1), outStride(rank, 1);
                for (int i = rank - 2; i >= 0; --i)
                {
                    inStride[i]  = inStride[i + 1] * X.shape[i + 1];
                    outStride[i] = outStride[i + 1] * out[i + 1];
                }
                int64_t elems = numElements(out);
                // Data is copied verbatim (a pure gather), so the same permutation serves any dtype;
                // only the element width differs. Int64 tensors (shape/index math) take the i64 path,
                // everything else the f32 path, each with its own typed source pointer and output alloc.
                bool           i64 = X.dtype == DType::Int64;
                const float   *xf  = i64 ? nullptr : X.host.f32();
                const int64_t *xi  = i64 ? X.host.i64() : nullptr;
                float         *yf  = i64 ? nullptr : cpu::allocOut(Y, out);
                int64_t       *yi  = i64 ? cpu::allocOutI64(Y, out) : nullptr;
                // For each output element, decode its flat index `oi` into per-axis coordinates via the
                // output strides, then re-address the source: output axis i corresponds to input axis
                // perm[i], so coordinate `c` contributes c * inStride[perm[i]] to the input offset `inf`.
                for (int64_t oi = 0; oi < elems; ++oi)
                {
                    int64_t rem = oi, inf = 0;
                    for (int i = 0; i < rank; ++i)
                    {
                        int64_t c = rem / outStride[i];
                        rem %= outStride[i];
                        inf += c * inStride[perm[i]];
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
        };
    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Transpose, TransposeCpu);
} // namespace vknn
