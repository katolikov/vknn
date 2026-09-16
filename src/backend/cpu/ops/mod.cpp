// ONNX Mod (elementwise remainder, attribute `fmod`: 0 = sign of the divisor, 1 = C fmod) with
// NumPy-style broadcasting. The remainder rules live in backend/cpu/mod_remainder.h.
//  - Int64 path, taken when either operand's runtime dtype is Int64 (the Binary rule) or the node's
//    operands are integers by the model's element types (modOperandsAreInteger: an INT32/INT8/UINT8
//    value is carried in fp32 lanes): int64 output, an fp32-carried operand truncated toward zero,
//    integer remainder (zero divisor -> 0 in both modes, INT64_MIN % -1 -> 0).
//  - Float path otherwise: fp32 output, std::fmod plus the fmod 0 sign fix-up (zero divisor -> 0 for
//    fmod 0, NaN for fmod 1). shaders/mod.comp computes the same bits on the GPU, and its integer mode
//    follows the same resolver.
// Every output element is one independent remainder, so both paths partition across threads.
#include "backend/cpu/broadcast.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/mod_remainder.h"
#include "backend/cpu/parallel.h"
#include "import/mod_integer_operands.h"
#include "vknn/graph.h"
#include "vknn/op.h"
#include <algorithm>
#include <string>
#include <vector>

namespace vknn {
    namespace {

        /// Operand count of ONNX Mod: dividend (input 0) and divisor (input 1).
        constexpr size_t kModOperandCount = 2;
        /// Operand slots of ONNX Mod.
        constexpr size_t kDividendSlot = 0;
        constexpr size_t kDivisorSlot  = 1;

        struct ModCpu: CpuOp {
            // modOperandsAreInteger walks the graph, so its answer is kept for the (graph, node) it was
            // resolved on; every run of the same node reuses it.
            const Graph *resolvedGraph   = nullptr;
            const Node  *resolvedNode    = nullptr;
            bool         integerOperands = false;

            void run(const Node &node, ExecContext &ctx) override {
                if (node.inputs.size() < kModOperandCount || node.inputs[kDividendSlot] == kNoTensor || node.inputs[kDivisorSlot] == kNoTensor || node.outputs.empty() || node.outputs[0] == kNoTensor)
                {
                    throw Error(Status::InvalidArgument, "Mod '" + node.name + "': needs a dividend, a divisor and an output");
                }
                const int64_t fmodMode = node.attr.geti("fmod", cpu::kModFloorRemainder);
                if (fmodMode != cpu::kModFloorRemainder && fmodMode != cpu::kModTruncRemainder)
                {
                    throw Error(Status::InvalidArgument, "Mod '" + node.name + "': fmod must be 0 or 1, got " + std::to_string(fmodMode));
                }
                if (ctx.graph != resolvedGraph || &node != resolvedNode)
                {
                    integerOperands = ctx.graph != nullptr && modOperandsAreInteger(*ctx.graph, node);
                    resolvedGraph   = ctx.graph;
                    resolvedNode    = &node;
                }
                const bool      floorRemainder = fmodMode == cpu::kModFloorRemainder;
                const RtTensor &dividendTensor = ctx.t(node.inputs[kDividendSlot]);
                const RtTensor &divisorTensor  = ctx.t(node.inputs[kDivisorSlot]);
                RtTensor       &outputTensor   = ctx.t(node.outputs[0]);
                const Shape    &dividendShape  = dividendTensor.shape;
                const Shape    &divisorShape   = divisorTensor.shape;
                const size_t    rank           = std::max(dividendShape.size(), divisorShape.size());
                const Shape     out            = cpu::broadcastOutputShape(node, dividendShape, divisorShape);
                // NumPy broadcasting right-aligns shapes: a lower-rank operand is padded on the LEFT with
                // size-1 axes, which extentOf reports for the padded prefix.
                auto extentOf = [&](const Shape &operandShape, size_t axis) -> int64_t {
                    const size_t paddingAxes = rank - operandShape.size();
                    return axis < paddingAxes ? 1 : operandShape[axis - paddingAxes];
                };
                const int64_t elementCount = cpu::elemCount(out); // a rank-0 scalar result carries its one element
                // Per-operand broadcast strides (row-major, built back-to-front): 0 on a size-1 axis so
                // every output index along it re-reads the one source element; otherwise the operand's
                // own packed stride.
                std::vector<int64_t> dividendStrides(rank), divisorStrides(rank);
                int64_t              dividendStride = 1, divisorStride = 1;
                for (int axis = (int) rank - 1; axis >= 0; --axis)
                {
                    dividendStrides[axis] = (extentOf(dividendShape, axis) == 1) ? 0 : dividendStride;
                    divisorStrides[axis]  = (extentOf(divisorShape, axis) == 1) ? 0 : divisorStride;
                    dividendStride *= extentOf(dividendShape, axis);
                    divisorStride *= extentOf(divisorShape, axis);
                }
                // Typed views resolved once, before the partition: the const accessors materialize a
                // mapped payload on first touch, which must not happen concurrently inside a chunk.
                const bool     dividendInt64 = dividendTensor.dtype == DType::Int64;
                const bool     divisorInt64  = divisorTensor.dtype == DType::Int64;
                const float   *dividendFloat = dividendInt64 ? nullptr : dividendTensor.host.f32();
                const int64_t *dividendInt   = dividendInt64 ? dividendTensor.host.i64() : nullptr;
                const float   *divisorFloat  = divisorInt64 ? nullptr : divisorTensor.host.f32();
                const int64_t *divisorInt    = divisorInt64 ? divisorTensor.host.i64() : nullptr;
                const int      threads       = cpu::threadCount(ctx.config);
                const int64_t  minChunk      = cpu::minChunkForWork(1);
                const auto     strideSet     = std::vector<const int64_t *> {dividendStrides.data(), divisorStrides.data()};
                if (dividendInt64 || divisorInt64 || integerOperands)
                {
                    int64_t *output = cpu::allocOutI64(outputTensor, out);
                    cpu::parallelFor(threads, 0, elementCount, minChunk, [&](int64_t chunkBegin, int64_t chunkEnd) {
                        cpu::BroadcastWalk walk(out, strideSet);
                        walk.seek(chunkBegin);
                        for (int64_t linearIndex = chunkBegin; linearIndex < chunkEnd; ++linearIndex, walk.next())
                        {
                            const int64_t dividend = dividendInt64 ? dividendInt[walk.offset(kDividendSlot)] : cpu::modOperandToInt64(dividendFloat[walk.offset(kDividendSlot)]);
                            const int64_t divisor = divisorInt64 ? divisorInt[walk.offset(kDivisorSlot)] : cpu::modOperandToInt64(divisorFloat[walk.offset(kDivisorSlot)]);
                            output[linearIndex] = cpu::modRemainderInt(dividend, divisor, floorRemainder);
                        }
                    });
                    return;
                }
                float *output = cpu::allocOut(outputTensor, out);
                cpu::parallelFor(threads, 0, elementCount, minChunk, [&](int64_t chunkBegin, int64_t chunkEnd) {
                    cpu::BroadcastWalk walk(out, strideSet);
                    walk.seek(chunkBegin);
                    for (int64_t linearIndex = chunkBegin; linearIndex < chunkEnd; ++linearIndex, walk.next())
                    {
                        output[linearIndex] = cpu::modRemainderFloat(dividendFloat[walk.offset(kDividendSlot)], divisorFloat[walk.offset(kDivisorSlot)], floorRemainder);
                    }
                });
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Mod, ModCpu);
} // namespace vknn
