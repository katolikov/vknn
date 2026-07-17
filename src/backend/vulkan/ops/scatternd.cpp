// ScatterND on the FLAT row-major GPU path. Two dispatches: (1) copy data -> out, (2) scatter the
// updates into out at each index row. The indices operand feeds one float IDX binding either way: a
// constant int64/float initializer is converted to float and uploaded here, and a runtime float index
// activation is read straight from its device buffer (the kernel truncates each element to int). Both run
// on the GPU — nothing falls back for a runtime index (see the gate in vk_gates.cpp).
#include "flat_ops.h"
#include "vk_op_common.h"
#include "vknn/op.h"
#include <vector>

namespace vknn {
    namespace {

        struct CopyPC {
            uint32_t count;
        };
        // dataDim/stride ride the geometry SSBO (flat::uploadFlatGeom, binding 3 in scatternd.comp),
        // so the decodable data rank is unbounded; the push constant carries only the scatter scalars.
        struct ScatterPC {
            uint32_t total;
            int      q, sliceSize, rank;
        };

        struct ScatterNDOp: VulkanOp {
            std::shared_ptr<vk::ComputePipeline> copyPipe;
            std::shared_ptr<vk::ComputePipeline> scatterPipe;
            std::shared_ptr<vk::Buffer>          idxBuf; // const index uploaded as float; null when index is activation
            std::shared_ptr<vk::Buffer>          geom;   // dataDim/stride, deduped SSBO
            std::shared_ptr<vk::Buffer>          holdData;
            std::shared_ptr<vk::Buffer>          holdUpd;
            CopyPC                               copyPc {};
            ScatterPC                            pc {};

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g  = *env.graph;
                const Shape &ds = g.desc(node.inputs[0]).shape;
                const Shape &is = g.desc(node.inputs[1]).shape;
                int          dr = (int) ds.size();

                // ONNX ScatterND: `indices` has shape [...,q]. `q` = its last dim = how many leading
                // data axes each index tuple addresses; `rows` = product of all the other index dims =
                // the number of tuples. Each tuple thus overwrites one contiguous slice of the trailing
                // (dr - q) data axes.
                int     q    = is.empty() ? 1 : (int) is.back();
                int64_t rows = 1;
                for (size_t i = 0; i + 1 < is.size(); ++i)
                {
                    rows *= is[i];
                }

                // Row-major element strides of `data`: stride[k] = product of dims below axis k. The
                // kernel dots an index tuple against stride[0..q) to reach a slice's base offset.
                std::vector<int64_t> stride(dr, 1);
                for (int k = dr - 2; k >= 0; --k)
                {
                    stride[k] = stride[k + 1] * ds[k + 1];
                }
                // Elements per scattered slice = product of the trailing, unindexed data axes [q, dr).
                int64_t sliceSize = 1;
                for (int k = q; k < dr; ++k)
                {
                    sliceSize *= ds[k];
                }

                // Pass-2 dispatch domain: one GPU thread per scattered element (rows tuples x sliceSize
                // elements each). copyPc.count below is the full data element count for pass 1.
                pc.total     = (uint32_t) (rows * sliceSize);
                pc.q         = q;
                pc.sliceSize = (int) sliceSize;
                pc.rank      = dr;
                std::vector<int32_t> dataDim(dr), strideArr(dr);
                for (int k = 0; k < dr; ++k)
                {
                    dataDim[k]   = (int) ds[k];
                    strideArr[k] = (int) stride[k];
                }
                geom         = flat::uploadFlatGeom(env, {dataDim, strideArr});
                copyPc.count = (uint32_t) numElements(ds);

                // Index: a constant initializer is uploaded as float; a runtime float index activation (e.g.
                // the decoder/camera ScatterNDs) is read straight from its device buffer. Both feed the
                // kernel's float IDX binding (the kernel truncates to int), so one kernel serves both.
                TensorId iid = node.inputs[1];
                if (g.isInitializer(iid))
                {
                    const HostBuffer  &ib   = g.initializers.at(iid);
                    DType              idt  = g.desc(iid).dtype;
                    int64_t            nIdx = (int64_t) numElements(is);
                    // Floor the staging vector at 4 elements: a scalar/empty index tensor gives nIdx == 0,
                    // and upload() (which also floors the device buffer at 4) must not be handed an empty
                    // vector whose data() could be null. Only the first nIdx entries are ever read.
                    std::vector<float> idxf((size_t) std::max<int64_t>(nIdx, 4), 0.f);
                    for (int64_t i = 0; i < nIdx; ++i)
                    {
                        if (idt == DType::Int64)
                        {
                            idxf[(size_t) i] = (float) ib.i64()[i];
                        } else if (idt == DType::Float16)
                        {
                            idxf[(size_t) i] = halfToFloatAt(ib.bytes.data(), i);
                        } else
                        {
                            idxf[(size_t) i] = ib.f32()[i];
                        }
                    }
                    idxBuf = upload(*env.ctx, idxf, env.useFp16);
                }

                copyPipe = env.pipeline(shader("scatternd_copy", env.useFp16), 2, sizeof(CopyPC), std::vector<uint32_t> {});
                scatterPipe =
                    env.pipeline(shader("scatternd", env.useFp16), 4, sizeof(ScatterPC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *data    = operandBuf(env, node.inputs[0], holdData);
                vk::Buffer *updates = operandBuf(env, node.inputs[2], holdUpd);
                vk::Buffer *idx     = idxBuf ? idxBuf.get() : env.devBuf(node.inputs[1]);
                vk::Buffer *out     = env.devBuf(node.outputs[0]);
                // Pass 1: out = copy(data). scatternd_copy.comp is local_size_x=256 == flat::kFlatLocalSize.
                copyPipe->dispatch(cmd, {data->handle(), out->handle()}, &copyPc, sizeof(copyPc), groups(copyPc.count, flat::kFlatLocalSize));
                // The framework only barriers BETWEEN nodes (read-after-write across ops); two dispatches
                // inside one record() are NOT auto-barriered. Pass 2 scatters into the SAME `out` buffer pass 1
                // wrote, so without this compute->compute barrier the dispatches can overlap and read stale
                // data.
                vk::computeBarrier(*env.ctx, cmd);
                // Pass 2: scatter updates into out at the index rows. scatternd.comp is local_size_x=256 ==
                // flat::kFlatLocalSize.
                if (pc.total > 0)
                {
                    scatterPipe->dispatch(cmd, {updates->handle(), idx->handle(), out->handle(), geom->handle()}, &pc, sizeof(pc), groups(pc.total, flat::kFlatLocalSize));
                }
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::ScatterND, ScatterNDOp);
} // namespace vknn
