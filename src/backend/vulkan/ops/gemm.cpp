// Gemm / fully-connected classifier. The pooled input is NC4HW4 with H=W=1, so the packed
// buffer is just the channel vector. Weights are stored row-major [Cout][Cin].
#include "pw_plan.h"
#include "vk_op_common.h"

namespace vknn {
    namespace {

        struct GemmOp: VulkanOp {
            // Threads per workgroup, byte-matched to fc.comp's `layout(local_size_x = 64)`.
            static constexpr uint32_t kLocalSize = 64;

            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          wbuf, bbuf;
            PwEpi                                epi;
            FcPC                                 pc {};
            int64_t                              Cout = 0;

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph &g      = *env.graph;
                const Shape &ws     = g.desc(node.inputs[1]).shape; // [Cout,Cin] when transB, else [Cin,Cout]
                int64_t      transB = node.attr.geti("transB", 0);
                int64_t      Cin, CoutL;
                if (transB)
                {
                    CoutL = ws[0];
                    Cin   = ws[1];
                } else
                {
                    Cin   = ws[0];
                    CoutL = ws[1];
                }
                Cout                    = CoutL;
                std::vector<float> wv   = initFloats(g, node.inputs[1]);
                const float       *wsrc = wv.data();
                // Prepack weights into the row-major [Cout][Cin] the fc shader indexes as wt[oc*Cin+ic].
                // transB=1: the source is already [Cout][Cin], copy straight through. transB=0 (the ONNX
                // default): the source is [Cin][Cout], so transpose by reading wsrc[ic*Cout+oc].
                wbuf                    = uploadCached(env, node.name + "#w", [&] {
                    std::vector<float> wp((size_t) CoutL * Cin);
                    for (int64_t oc = 0; oc < CoutL; ++oc)
                    {
                        for (int64_t ic = 0; ic < Cin; ++ic)
                        {
                            wp[oc * Cin + ic] = transB ? wsrc[oc * Cin + ic] : wsrc[ic * CoutL + oc];
                        }
                    }
                    return wp;
                });
                std::vector<float> bv;
                if (pwCoreInputs(node) > 2 && node.inputs[2] != kNoTensor)
                {
                    bv = initFloats(g, node.inputs[2]);
                }
                const float *bsrc = bv.data();
                // The fc shader always seeds its accumulator with bias[oc], so materialize a full-length
                // bias buffer even when Gemm has no C input: default to zeros, filled only when present.
                // bsrc is dereferenced solely inside the same has-bias guard, so an empty bv is safe here.
                bbuf              = uploadCached(env, node.name + "#b", [&] {
                    std::vector<float> bias(CoutL, 0.f);
                    if (pwCoreInputs(node) > 2 && node.inputs[2] != kNoTensor)
                    {
                        for (int64_t i = 0; i < CoutL; ++i)
                        {
                            bias[i] = bsrc[i];
                        }
                    }
                    return bias;
                });
                // M = batch rows: classifiers have M=1, the YoNoSplat camera head M=2 (two views). Per-row
                // stride differs by layout: NC4HW4 pads channels to a multiple of 4 (H=W=1), a gpuFlat
                // operand is exactly C.
                int64_t M = Cin > 0 ? numElements(g.desc(node.inputs[0]).shape) / Cin : 1;
                if (M < 1)
                {
                    M = 1;
                }
                auto pad4 = [](int64_t c) {
                    return ((c + 3) / 4) * 4;
                };
                int srcStride = (int) (g.desc(node.inputs[0]).gpuFlat ? Cin : pad4(Cin));
                int dstStride = (int) (g.desc(node.outputs[0]).gpuFlat ? CoutL : pad4(CoutL));
                pc            = {(int) Cin, (int) CoutL, (int) M, srcStride, dstStride, (int) node.fusedAct, node.actLo, node.actHi};
                epi.prepare(node, env, g.desc(node.outputs[0]).gpuFlat, g.desc(node.outputs[0]).shape);
                pipe = env.pipeline(shader((std::string("fc") + epi.suffix()).c_str(), env.useFp16), 4 + epi.extraBufs(), sizeof(FcPC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer           *src  = env.devBuf(node.inputs[0]);
                vk::Buffer           *dst  = env.devBuf(node.outputs[0]);
                std::vector<VkBuffer> bufs = {src->handle(), wbuf->handle(), bbuf->handle(), dst->handle()};
                epi.append(bufs, node, env, dst->handle());
                // One thread per output element: Cout channels * M rows, ceil-divided into kLocalSize-wide
                // workgroups. The shader linearizes gid across a 2D grid, so an overflowing X count folds
                // into Y — no per-op 65535-limit handling needed here.
                pipe->dispatch(cmd, bufs, &pc, sizeof(pc), groups(Cout * pc.M, kLocalSize));
            }
        };

    } // namespace

    VKNN_REGISTER_VK_OP(OpType::Gemm, GemmOp);

} // namespace vknn
