// ONNX Mod (elementwise remainder, attribute `fmod`: 0 = sign of the divisor, 1 = C fmod) with
// NumPy-style broadcasting. The remainder rules live in backend/cpu/mod_remainder.h.
//  - Int64 path, taken when EITHER operand's runtime dtype is Int64 (the Binary rule): int64 output,
//    an fp32-carried operand truncated toward zero, integer remainder (zero divisor -> 0,
//    INT64_MIN % -1 -> 0).
//  - Float path otherwise: fp32 output, std::fmod plus the fmod 0 sign fix-up (zero divisor -> 0 for
//    fmod 0, NaN for fmod 1). shaders/mod.comp computes the same bits on the GPU.
// Every output element is one independent remainder, so both paths partition across threads.
#include "backend/cpu/broadcast.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/mod_remainder.h"
#include "backend/cpu/parallel.h"
#include "vknn/op.h"
#include <algorithm>
#include <string>
#include <vector>

namespace vknn {
    namespace {

        /// Operand count of ONNX Mod: dividend (input 0) and divisor (input 1).
        constexpr size_t kModOperandCount = 2;

        struct ModCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                if (node.inputs.size() < kModOperandCount || node.inputs[0] == kNoTensor || node.inputs[1] == kNoTensor || node.outputs.empty() || node.outputs[0] == kNoTensor)
                {
                    throw Error(Status::InvalidArgument, "Mod '" + node.name + "': needs a dividend, a divisor and an output");
                }
                const int64_t fmodMode = node.attr.geti("fmod", cpu::kModFloorRemainder);
                if (fmodMode != cpu::kModFloorRemainder && fmodMode != cpu::kModTruncRemainder)
                {
                    throw Error(Status::InvalidArgument, "Mod '" + node.name + "': fmod must be 0 or 1, got " + std::to_string(fmodMode));
                }
                const bool      floorRemainder = fmodMode == cpu::kModFloorRemainder;
                const RtTensor &A              = ctx.t(node.inputs[0]);
                const RtTensor &B              = ctx.t(node.inputs[1]);
                RtTensor       &Y              = ctx.t(node.outputs[0]);
                const Shape    &sa = A.shape, &sb = B.shape;
                const size_t    rank = std::max(sa.size(), sb.size());
                Shape           out(rank, 1);
                // NumPy broadcasting right-aligns shapes: a lower-rank operand is padded on the LEFT with
                // size-1 axes, which dimOf reports for the padded prefix.
                auto dimOf = [&](const Shape &s, size_t i) -> int64_t {
                    size_t off = rank - s.size();
                    return i < off ? 1 : s[i - off];
                };
                for (size_t i = 0; i < rank; ++i)
                {
                    const int64_t da = dimOf(sa, i), db = dimOf(sb, i);
                    if (da != db && da != 1 && db != 1)
                    {
                        throw Error(Status::InvalidArgument, "Mod '" + node.name + "': operand shapes " + shapeStr(sa) + " and " + shapeStr(sb) + " do not broadcast (axis " + std::to_string(i) + ")");
                    }
                    out[i] = (da == 0 || db == 0) ? 0 : std::max(da, db); // a 0 dim broadcasts to 0 (NumPy), never to 1
                }
                const int64_t n = cpu::elemCount(out); // a rank-0 scalar result carries its one element
                // Per-operand broadcast strides (row-major, built back-to-front): 0 on a size-1 axis so
                // every output index along it re-reads the one source element; otherwise the operand's
                // own packed stride.
                std::vector<int64_t> oa(rank), ob(rank);
                int64_t              sA = 1, sB = 1;
                for (int i = (int) rank - 1; i >= 0; --i)
                {
                    oa[i] = (dimOf(sa, i) == 1) ? 0 : sA;
                    ob[i] = (dimOf(sb, i) == 1) ? 0 : sB;
                    sA *= dimOf(sa, i);
                    sB *= dimOf(sb, i);
                }
                // Typed views resolved once, before the partition: the const accessors materialize a
                // mapped payload on first touch, which must not happen concurrently inside a chunk.
                const bool     aInt64    = A.dtype == DType::Int64;
                const bool     bInt64    = B.dtype == DType::Int64;
                const float   *aFloat    = aInt64 ? nullptr : A.host.f32();
                const int64_t *aInt      = aInt64 ? A.host.i64() : nullptr;
                const float   *bFloat    = bInt64 ? nullptr : B.host.f32();
                const int64_t *bInt      = bInt64 ? B.host.i64() : nullptr;
                const int      threads   = cpu::threadCount(ctx.config);
                const int64_t  minChunk  = cpu::minChunkForWork(1);
                const auto     strideSet = std::vector<const int64_t *> {oa.data(), ob.data()};
                if (aInt64 || bInt64)
                {
                    int64_t *y = cpu::allocOutI64(Y, out);
                    cpu::parallelFor(threads, 0, n, minChunk, [&](int64_t lo, int64_t hi) {
                        cpu::BroadcastWalk walk(out, strideSet);
                        walk.seek(lo);
                        for (int64_t lin = lo; lin < hi; ++lin, walk.next())
                        {
                            const int64_t dividend = aInt64 ? aInt[walk.offset(0)] : cpu::modOperandToInt64(aFloat[walk.offset(0)]);
                            const int64_t divisor  = bInt64 ? bInt[walk.offset(1)] : cpu::modOperandToInt64(bFloat[walk.offset(1)]);
                            y[lin]                 = cpu::modRemainderInt(dividend, divisor, floorRemainder);
                        }
                    });
                    return;
                }
                float *y = cpu::allocOut(Y, out);
                cpu::parallelFor(threads, 0, n, minChunk, [&](int64_t lo, int64_t hi) {
                    cpu::BroadcastWalk walk(out, strideSet);
                    walk.seek(lo);
                    for (int64_t lin = lo; lin < hi; ++lin, walk.next())
                    {
                        y[lin] = cpu::modRemainderFloat(aFloat[walk.offset(0)], bFloat[walk.offset(1)], floorRemainder);
                    }
                });
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Mod, ModCpu);
} // namespace vknn
