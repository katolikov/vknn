// Flat (row-major) ONNX Mod on the GPU with N-D broadcasting at any rank: d = the exact remainder of
// shaders/mod.comp (fmod 1: std::fmod; fmod 0: the remainder whose sign follows the divisor, 0 for a
// zero divisor), bit-identical to the CPU oracle. Always runs on the flat path (descriptor row L::Flat).
// Integer operands (modOperandsAreInteger, the resolver the CPU kernel uses) select the kernel's integer
// mode: operands read as the CPU int64 path reads them and a zero divisor yields 0 in both modes.
// pinIntegerResultsFp32 pins the output of an integer Mod (fmod 0, or integer operands per the same
// resolver) to fp32 storage at load, so env.useFp16 is false for such a node; fmod 1 on float operands
// runs at the segment's precision. Each operand must broadcast to the output (modAlignedOperandExtents
// rejects the node otherwise). Constant operands are uploaded flat at the node's storage precision in
// prepare() (initFloats decodes every initializer dtype, a rank-0 scalar keeps its one element). The
// per-axis outDim/dividendStride/divisorStride geometry rides a content-deduped SSBO
// (flat::uploadFlatGeom) at binding 3; the push constant carries only scalars. Layout byte-matches
// shaders/mod.comp.
#include "backend/vulkan/ops/mod_operand_geometry.h"
#include "flat_ops.h"
#include "import/mod_integer_operands.h"
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
        /// Operand slots of ONNX Mod.
        constexpr int kDividendSlot = 0;
        constexpr int kDivisorSlot  = 1;
        /// Storage buffers bound by shaders/mod.comp: dividend, divisor, destination, geometry.
        constexpr int kModBufferCount = 4;

        struct ModVk: VulkanOp {
            struct PC {
                int rank, total, fmodMode, integerOperands;
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
                if ((int) node.inputs.size() < kModOperandCount || node.inputs[kDividendSlot] == kNoTensor || node.inputs[kDivisorSlot] == kNoTensor ||
                    node.outputs.empty() || node.outputs[0] == kNoTensor)
                {
                    throw Error(Status::InvalidArgument, "Mod '" + node.name + "': needs a dividend, a divisor and an output");
                }
                const bool integerOperands = modOperandsAreInteger(g, node);
                if (integerOperands && env.useFp16)
                {
                    // Integer values are exact only up to 2^11 in fp16 storage; the load-time pin keeps
                    // every integer Mod at fp32.
                    throw Error(Status::RuntimeError, "Mod '" + node.name + "': integer operands reached an fp16 kernel; pinIntegerResultsFp32 must pin its output to fp32");
                }
                const Shape out  = g.desc(node.outputs[0]).shape;
                const int   rank = (int) out.size();
                pc.rank          = rank;
                // A rank-0 output carries one element (numElements reports 0 for an empty shape); a
                // zero-extent output dispatches nothing.
                pc.total           = (int) (out.empty() ? 1 : numElements(out));
                pc.fmodMode        = (int) fmodMode;
                pc.integerOperands = integerOperands ? 1 : 0;
                std::vector<int32_t> outDim(rank), dividendStride(rank), divisorStride(rank);
                for (int axis = 0; axis < rank; ++axis)
                {
                    outDim[axis] = (int) out[axis];
                }
                auto bindOperand = [&](int operandSlot, std::vector<int32_t> &strideDestination) {
                    const TensorId       operand        = node.inputs[operandSlot];
                    const Shape          operandShape   = g.desc(operand).shape;
                    std::vector<int64_t> alignedShape   = modAlignedOperandExtents(node.name, operandShape, out);
                    auto                 alignedStrides = flat::rowStrides(alignedShape);
                    for (int axis = 0; axis < rank; ++axis)
                    {
                        // Broadcast convention shared with flat::Binary: a size-1 axis gets stride 0 so the
                        // shader reads the same element for every output coordinate along that axis.
                        strideDestination[axis] = alignedShape[axis] == 1 ? 0 : (int) alignedStrides[axis];
                    }
                    if (g.isInitializer(operand))
                    {
                        std::vector<float> constantValues = initFloats(g, operand);                      // decodes fp16/int64/int8/uint8; fp32 passthrough
                        constantValues.resize((size_t) std::max<int64_t>(1, numElements(operandShape))); // 0-D scalar: keep its 1 element
                        constBuf[operandSlot] = upload(*env.ctx, constantValues, env.useFp16);
                    }
                };
                bindOperand(kDividendSlot, dividendStride);
                bindOperand(kDivisorSlot, divisorStride);
                geom = flat::uploadFlatGeom(env, {outDim, dividendStride, divisorStride});
                pipe = env.pipeline(shader("mod", env.useFp16), kModBufferCount, sizeof(PC), std::vector<uint32_t> {});
            }

            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                auto operandBuffer = [&](int operandSlot) {
                    return constBuf[operandSlot] ? constBuf[operandSlot].get() : env.devBuf(node.inputs[operandSlot]);
                };
                // One flat invocation per output element; mod.comp is local_size_x=256 == flat::kFlatLocalSize.
                pipe->dispatch(cmd, {operandBuffer(kDividendSlot)->handle(), operandBuffer(kDivisorSlot)->handle(), env.devBuf(node.outputs[0])->handle(), geom->handle()}, &pc, sizeof(pc), groups(pc.total, flat::kFlatLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Mod, ModVk);
} // namespace vknn
