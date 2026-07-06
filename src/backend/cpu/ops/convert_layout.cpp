// ConvertLayout on the CPU backend: NC4HW4 is a device-only storage format, so on the host (all
// canonical NCHW) the convert is an identity copy. Present so a ConvertLayout forced off the GPU
// (Config::disableVkOps, GPU-island fold) still runs and a post-layout-pass graph can execute
// entirely on the CPU. See ConvertLayoutCpu below for the details.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        /// CPU kernel for ConvertLayout: a shape- and value-preserving element copy.
        ///
        /// insertLayoutConverts splices these nodes wherever the GPU crosses an NC4HW4 <-> flat
        /// row-major boundary; node.subOp carries the direction (0: NC4HW4 -> flat, 1: flat ->
        /// NC4HW4) and the logical NCHW shape is identical on both sides. The physical repack is
        /// the GPU kernel's job on device buffers only: every host residency is canonical NCHW
        /// (rt_tensor.h), and the backend handoff (packToBuffer/unpackFromBuffer, keyed on the
        /// tensor's gpuFlat flag) packs to or gathers from NC4HW4 — zero-filling pad lanes — on
        /// every host<->device crossing. Both directions therefore share one host representation
        /// and reduce to the same identity copy over the logical elements; a host-side repack here
        /// would double-convert against the boundary pack. Mirrors ConvertDtypeCpu, the identity
        /// for the other Vulkan-only storage convert (fp16/fp32).
        struct ConvertLayoutCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X = ctx.t(node.inputs[0]);
                RtTensor       &Y = ctx.t(node.outputs[0]);
                // Same shape, same dtype, same element order: a raw byte copy into an independent
                // output buffer.
                cpu::copyAs(X, Y, X.shape);
            }
        };
    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::ConvertLayout, ConvertLayoutCpu);
} // namespace vknn
