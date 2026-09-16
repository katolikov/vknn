// Flat (row-major) ONNX Mod on the GPU with N-D broadcasting at any rank: d = the exact remainder of
// shaders/mod.comp (fmod 1: std::fmod; fmod 0: the remainder whose sign follows the divisor, 0 for a
// zero divisor), bit-identical to the CPU oracle's float path. Always runs on the flat path
// (descriptor row L::Flat). pinIntegerResultsFp32 pins the output of an integer Mod (fmod 0, or an
// integer-typed operand) to fp32 storage at load, so env.useFp16 is false for such a node; fmod 1 on
// float operands runs at the segment's precision. Constant operands are uploaded flat at the node's
// storage precision in prepare() (initFloats decodes every initializer dtype, a rank-0 scalar keeps
// its one element). The per-axis outDim/aStride/bStride geometry rides a content-deduped SSBO
// (flat::uploadFlatGeom) at binding 3; the push constant carries only scalars. Layout byte-matches
// shaders/mod.comp.
#include "flat_ops.h"
#include "vk_op_common.h"
#include <algorithm>
#include <string>

namespace vknn {
    namespace {

        /// ONNX Mod `fmod` values (the attribute's default is the floor remainder); shaders/mod.comp
        /// names the floor remainder kModFloorRemainder.
        constexpr int kModFloorRemainder = 0;
        constexpr int kModTruncRemainder = 1;
        /// Operand count of ONNX Mod: dividend (input 0) and divisor (input 1).
        constexpr int kModOperandCount = 2;
        /// Storage buffers bound by shaders/mod.comp: dividend, divisor, destination, geometry.
        constexpr int kModBufferCount = 4;

        struct ModVk: VulkanOp {
            struct PC {
                int rank, total, fmodMode;
            } pc {};
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          geom;
            std::shared_ptr<vk::Buffer>          constBuf[kModOperandCount];

            void prepare(const Node &node, VkOpEnv &env) override {
                const Graph  &g        = *env.graph;
                const int64_t fmodMode = node.attr.geti("fmod", kModFloorRemainder);
                if (fmodMode != kModFloorRemainder && fmodMode != kModTruncRemainder)
                {
                    throw Error(Status::InvalidArgument, "Mod '" + node.name + "': fmod must be 0 or 1, got " + std::to_string(fmodMode));
                }
                if ((int) node.inputs.size() < kModOperandCount || node.inputs[0] == kNoTensor || node.inputs[1] == kNoTensor)
                {
                    throw Error(Status::InvalidArgument, "Mod '" + node.name + "': needs a dividend and a divisor");
                }
                const Shape out  = g.desc(node.outputs[0]).shape;
                const int   rank = (int) out.size();
                pc.rank          = rank;
                // A rank-0 output carries one element (numElements reports 0 for an empty shape); a
                // zero-extent output dispatches nothing.
                pc.total    = (int) (out.empty() ? 1 : numElements(out));
                pc.fmodMode = (int) fmodMode;
                std::vector<int32_t> outDim(rank), aStride(rank), bStride(rank);
                for (int k = 0; k < rank; ++k)
                {
                    outDim[k] = (int) out[k];
                }
                auto setup = [&](TensorId t, int which) {
                    const Shape          s = g.desc(t).shape;
                    std::vector<int64_t> ps(rank, 1); // right-align this operand's shape into the output rank (leading dims padded to 1)
                    for (int k = 0; k < (int) s.size(); ++k)
                    {
                        ps[rank - (int) s.size() + k] = s[k];
                    }
                    auto     st  = flat::rowStrides(ps);
                    int32_t *dst = (which == 0 ? aStride.data() : bStride.data());
                    for (int k = 0; k < rank; ++k)
                    {
                        // Broadcast convention shared with flat::Binary: a size-1 dim gets stride 0 so the
                        // shader reads the same element for every output coordinate along that axis.
                        dst[k] = ps[k] == 1 ? 0 : (int) st[k];
                    }
                    if (g.isInitializer(t))
                    {
                        std::vector<float> cv = initFloats(g, t);                 // decodes fp16/int64/int8/uint8; fp32 passthrough
                        cv.resize((size_t) std::max<int64_t>(1, numElements(s))); // 0-D scalar: keep its 1 element
                        constBuf[which] = upload(*env.ctx, cv, env.useFp16);
                    }
                };
                setup(node.inputs[0], 0);
                setup(node.inputs[1], 1);
                geom = flat::uploadFlatGeom(env, {outDim, aStride, bStride});
                pipe = env.pipeline(shader("mod", env.useFp16), kModBufferCount, sizeof(PC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                auto buf = [&](int which) {
                    return constBuf[which] ? constBuf[which].get() : env.devBuf(node.inputs[which]);
                };
                // One flat invocation per output element; mod.comp is local_size_x=256 == flat::kFlatLocalSize.
                pipe->dispatch(cmd, {buf(0)->handle(), buf(1)->handle(), env.devBuf(node.outputs[0])->handle(), geom->handle()}, &pc, sizeof(pc), groups(pc.total, flat::kFlatLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Mod, ModVk);
} // namespace vknn
