// ConvertDtype on the CPU backend: the fp16<->fp32 storage convert is a Vulkan-only concern, so on
// the host (all-fp32) it is an identity copy. Present so a graph carrying selective-fp32 markers
// still runs on / falls back to CPU. See ConvertDtypeCpu below for the details.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        /// CPU reference kernel for ConvertDtype: a shape- and value-preserving element copy.
        ///
        /// The importer's fp32 pass (mark_fp32.cpp) inserts a ConvertDtype at every fp16/fp32 storage
        /// frontier so downstream nodes read their input at the storage precision they expect. That
        /// convert only matters to the GPU backend, where fp16 and fp32 are distinct buffer layouts.
        /// On the host every activation is fp32, so both sides of the frontier share one representation
        /// and the convert degenerates to an identity copy — output shape equals input shape and each
        /// value is carried through unchanged.
        struct ConvertDtypeCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X = ctx.t(node.inputs[0]);
                RtTensor       &Y = ctx.t(node.outputs[0]);
                // Y takes X's shape verbatim (ConvertDtype never reshapes), then receives a straight
                // element-for-element copy over the flat fp32 buffer.
                float          *y = cpu::allocOut(Y, X.shape);
                const float    *x = X.host.f32();
                int64_t         n = cpu::elemCount(X.shape); // a rank-0 scalar carries its one element
                for (int64_t i = 0; i < n; ++i)
                {
                    y[i] = x[i];
                }
            }
        };
    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::ConvertDtype, ConvertDtypeCpu);
} // namespace vknn
