// Cast on the GPU. A float->float cast is a pure buffer copy (the segment runs in one precision, so the
// cast is a no-op). A float->integer cast truncates toward zero and narrows to the target range via
// cast.comp, keeping compute-precision storage — the flat path carries a logical integer as a truncated
// fp32/fp16 value, and the graph boundary repacks it to the declared dtype. The narrowing per target
// matches the CPU Cast op followed by the readback narrowing (session.cpp readbackOutput) bit-for-bit:
// INT8 wraps modulo 2^8 (ONNX Cast to a smaller int is modulo, not saturation), UINT8 saturates to
// [0,255], the wide targets (INT32/INT64/UINT32/UINT64) truncate only. supportsNode runs int64 -> the
// scalar/narrow targets on the GPU (the int64 lanes decode to compute-precision float at the pack
// boundary); INT16/UINT16 are not distinct vknn dtypes (they map to fp32 storage) and take the copy path.
#include "backend/vulkan/vk_tune_race.h"
#include "vk_op_common.h"
#include "vknn/logging.h"
#include "vknn/op.h"
#include <limits>

namespace vknn {
    namespace {

        // Elements per lane-quad of the vectorized cast kernel (whole-vec4 load/store); mirrors
        // CAST_QUAD in shaders/cast_v4.comp.
        constexpr int kCastQuad = 4;
        // Tune-table values of the cast kernel pick (append-only).
        constexpr int kCastKernelScalar = 0; ///< cast: one element per lane.
        constexpr int kCastKernelQuad   = 1; ///< cast_v4: one whole-vec4 quad per lane.

        struct CastOp: VulkanOp {
            struct PC {
                int   total;
                float lo, hi;
                int   mode; // 0 = wide (clamp to fence inf/NaN), 1 = INT8 wrap, 2 = UINT8 saturate
            } pc {};
            bool truncate = false;
            // cast_v4 (one whole-vec4 quad per lane) when the cached race picked it; the scalar
            // kernel otherwise. Byte-identical either way (see pickCastQuad).
            bool                                 quadKernel = false;
            std::shared_ptr<vk::ComputePipeline> pipe;
            std::shared_ptr<vk::Buffer>          hold0; // when input is a constant initializer

            // The scalar and quad cast kernels are byte-interchangeable: castOne is the scalar
            // kernel's body per component, at the same element index, with the same rounding. The
            // pick is therefore placement-only and follows the standard cached race
            // (fused_pointwise.cpp pickFlatQuad): Tuning::None keeps the scalar kernel (the
            // deterministic default), Fast/Heavy race both once on dedicated scratch through
            // TuneTimer/raceCandidates and persist the winner; the quad challenger must clear
            // vk::kTuneRaceMargin to displace the incumbent. raceTotal is the output's logical
            // element count — record()'s physical count can add NC4HW4 channel padding, which
            // moves the raced workload by at most the pad fraction and the bytes not at all (the
            // pick is placement-only).
            bool pickCastQuad(VkOpEnv &env, int raceTotal) {
                if (raceTotal <= 0)
                {
                    return false;
                }
                char buf[64];
                snprintf(buf, sizeof(buf), "cast_%d_%d_%d", raceTotal, pc.mode, env.useFp16 ? 1 : 0);
                std::string sig = env.gpuTag + "/" + buf;
                int         reuse;
                if (env.reuseTuned(sig, reuse) && (reuse == kCastKernelScalar || reuse == kCastKernelQuad))
                {
                    return reuse == kCastKernelQuad;
                }
                if (env.tuning == Tuning::None || !env.runner)
                {
                    return false;
                }
                const size_t es = (size_t) (env.useFp16 ? 2 : 4);
                auto         mk = [&](size_t bytes) {
                    return std::make_shared<vk::Buffer>(*env.ctx, std::max<size_t>(bytes, 16), vk::MemPref::kDeviceOnly);
                };
                auto          sSrc = mk((size_t) raceTotal * es);
                auto          sDst = mk((size_t) raceTotal * es);
                vk::TuneTimer timer(env);
                PC            rpc = pc;
                rpc.total         = raceTotal;
                auto       ms     = vk::raceCandidates(2, [&](int index) {
                    const bool q = index == kCastKernelQuad;
                    auto       racePipe = env.pipeline(shader(q ? "cast_v4" : "cast", env.useFp16), 2, sizeof(PC), std::vector<uint32_t> {env.flatLocalSize});
                    const int  lanes = q ? (raceTotal + kCastQuad - 1) / kCastQuad : raceTotal;
                    return timer.time([&](VkCommandBuffer cmd) {
                        racePipe->dispatch(cmd, {sSrc->handle(), sDst->handle()}, &rpc, sizeof(rpc), groups(lanes, env.flatLocalSize));
                    });
                });
                const bool won    = ms.size() == 2 && ms[kCastKernelQuad] < ms[kCastKernelScalar] * vk::kTuneRaceMargin;
                if (env.weights)
                {
                    env.weights->setTuned(sig, won ? kCastKernelQuad : kCastKernelScalar, (int) env.tuning);
                }
                VKNN_DEBUG << "autotune " << sig << " -> " << (won ? "quad" : "scalar") << vk::raceTimes(ms);
                return won;
            }

            void prepare(const Node &node, VkOpEnv &env) override {
                int64_t to = node.attr.geti("to", 1); // ONNX TensorProto dtype
                pc.mode    = 0;
                // integer targets that need a value truncation (not a same-precision copy)
                switch (to)
                {
                    case 2:
                        truncate = true;
                        pc.mode  = 2;
                        pc.lo    = 0.0f;
                        pc.hi    = 255.0f;
                        break; // UINT8: saturate (matches readbackOutput's uint8 clamp)
                    case 3:
                        truncate = true;
                        pc.mode  = 1;
                        pc.lo    = -128.0f;
                        pc.hi    = 127.0f;
                        break; // INT8: modulo wrap (matches readbackOutput's (int8_t) narrowing)
                    case 4:
                        truncate = true;
                        pc.lo    = 0.0f;
                        pc.hi    = 65535.0f;
                        break; // UINT16 (fp32 output tensor: narrows here, saturate)
                    case 5:
                        truncate = true;
                        pc.lo    = -32768.0f;
                        pc.hi    = 32767.0f;
                        break; // INT16 (fp32 output tensor: narrows here, saturate)
                    case 9:
                        truncate = true;
                        pc.lo    = 0.0f;
                        pc.hi    = 1.0f;
                        break; // BOOL (uint8 output tensor: truncate + clamp to [0,1])
                    case 6:
                    case 7:
                    case 12:
                    case 13:
                        // INT32/INT64/UINT32/UINT64: truncate only. These ranges exceed the
                        // exactly-representable integer span of the compute float, so clamping to the
                        // true dtype bounds would be meaningless; the clamp is widened to the fp32 finite
                        // max (~3.4e38) purely to fence off inf/NaN, and trunc() alone drops the fraction.
                        truncate = true;
                        pc.lo    = -3.4e38f;
                        pc.hi    = 3.4e38f;
                        break;
                    default:
                        // FLOAT/FLOAT16/DOUBLE (and INT16/UINT16/BOOL, which map to fp32/uint8 storage,
                        // not a distinct narrow-int output tensor) -> same-precision copy.
                        truncate = false;
                        break;
                }
                if (truncate)
                {
                    // 2 SSBO bindings (src, dst); the spec constant pins both kernels to the
                    // device-resolved width (spec 0 = env.flatLocalSize, matching groups() below).
                    quadKernel = pickCastQuad(env, (int) numElements(env.graph->desc(node.outputs[0]).shape));
                    pipe       = env.pipeline(shader(quadKernel ? "cast_v4" : "cast", env.useFp16), 2, sizeof(PC), std::vector<uint32_t> {env.flatLocalSize});
                }
            }
            void record(VkCommandBuffer cmd, const Node &node, VkOpEnv &env) override {
                vk::Buffer *src = operandBuf(env, node.inputs[0], hold0);
                vk::Buffer *dst = env.devBuf(node.outputs[0]);
                if (!truncate)
                {
                    // Same-precision cast is a raw byte copy. min() clamps the region to the smaller
                    // buffer so a src/dst allocation-size mismatch (e.g. differing NC4HW4 channel padding)
                    // can never over-read or over-write.
                    VkBufferCopy c {0, 0, std::min(src->bytes(), dst->bytes())};
                    vkCmdCopyBuffer(cmd, src->handle(), dst->handle(), 1, &c);
                    return;
                }
                // Truncate every storage slot of the output buffer (element-wise, layout-agnostic; NC4HW4
                // channel padding is truncated harmlessly). elemSize matches the compute precision.
                int elemSize = env.useFp16 ? 2 : 4;
                pc.total     = (int) (dst->bytes() / elemSize);
                // Lane count: elements for the scalar kernel, whole quads for the vectorized twin
                // (which re-derives its quad count from pc.total).
                const int laneSteps = quadKernel ? (pc.total + kCastQuad - 1) / kCastQuad : pc.total;
                pipe->dispatch(cmd, {src->handle(), dst->handle()}, &pc, sizeof(pc), groups(laneSteps, env.flatLocalSize));
            }
        };

    } // namespace
    VKNN_REGISTER_VK_OP(OpType::Cast, CastOp);
} // namespace vknn
