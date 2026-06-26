// ScatterND (reduction='none'): out = copy(data); then for each index row, write the corresponding
// update slice into out at that location. indices has shape [...,q]; each length-q index addresses
// the first q dims of data, and the trailing (rank-q) dims form the update slice. Canonical
// reference used both as the CPU op and as the const-fold/Vulkan-fallback oracle.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"
#include <cstring>

namespace vknn {
    namespace {

        struct ScatterNDCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &D = ctx.t(node.inputs[0]); // data
                const RtTensor &I = ctx.t(node.inputs[1]); // indices (int64)
                const RtTensor &U = ctx.t(node.inputs[2]); // updates
                RtTensor       &Y = ctx.t(node.outputs[0]);

                const Shape &ds = D.shape;
                int          dr = (int) ds.size();
                // q = size of the last index dim; number of index rows = product of the leading index dims.
                int     q    = I.shape.empty() ? 1 : (int) I.shape.back();
                int64_t rows = 1;
                for (size_t i = 0; i + 1 < I.shape.size(); ++i)
                {
                    rows *= I.shape[i];
                }
                if (I.shape.empty())
                {
                    rows = I.elems(); // degenerate: flat index vector
                }

                // Row-major strides of data; sliceSize = elements written per index row.
                std::vector<int64_t> stride(dr, 1);
                for (int k = dr - 2; k >= 0; --k)
                {
                    stride[k] = stride[k + 1] * ds[k + 1];
                }
                int64_t sliceSize = 1;
                for (int k = q; k < dr; ++k)
                {
                    sliceSize *= ds[k];
                }

                const int64_t *idx = I.host.i64();
                bool           i64 = D.dtype == DType::kInt64;

                if (i64)
                {
                    int64_t *y = cpu::allocOutI64(Y, ds);
                    std::memcpy(y, D.host.i64(), (size_t) D.elems() * 8);
                    const int64_t *u = U.host.i64();
                    for (int64_t r = 0; r < rows; ++r)
                    {
                        int64_t off = 0;
                        for (int c = 0; c < q; ++c)
                        {
                            int64_t ix = idx[r * q + c];
                            if (ix < 0)
                            {
                                ix += ds[c];
                            }
                            off += ix * stride[c];
                        }
                        std::memcpy(y + off, u + r * sliceSize, (size_t) sliceSize * 8);
                    }
                } else
                {
                    float *y = cpu::allocOut(Y, ds);
                    std::memcpy(y, D.host.f32(), (size_t) D.elems() * 4);
                    const float *u = U.host.f32();
                    for (int64_t r = 0; r < rows; ++r)
                    {
                        int64_t off = 0;
                        for (int c = 0; c < q; ++c)
                        {
                            int64_t ix = idx[r * q + c];
                            if (ix < 0)
                            {
                                ix += ds[c];
                            }
                            off += ix * stride[c];
                        }
                        std::memcpy(y + off, u + r * sliceSize, (size_t) sliceSize * 4);
                    }
                }
            }
            bool supportsDType(DType) const override {
                return true;
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::kScatterND, ScatterNDCpu);
} // namespace vknn
