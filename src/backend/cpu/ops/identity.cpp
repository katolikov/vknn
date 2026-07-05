// Identity: pass the tensor through unchanged.
#include "backend/cpu/cpu_backend.h"

namespace vknn {
    namespace {

        /// ONNX Identity: the output is an element-wise, shape- and dtype-preserving copy of the
        /// single input. Any dtype passes through untouched.
        struct IdentityCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X = ctx.t(node.inputs[0]);
                RtTensor       &Y = ctx.t(node.outputs[0]);
                Y.shape           = X.shape;
                Y.dtype           = X.dtype;
                // HostBuffer holds a std::vector<uint8_t>, so this copies the raw element bytes into
                // an independent output buffer rather than aliasing the input's storage.
                Y.host = X.host;
                // The fresh host copy is the current residency; invalidate any prior device copy so
                // the next backend handoff re-uploads instead of reading stale device data.
                Y.hostValid   = true;
                Y.deviceValid = false;
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::Identity, IdentityCpu);
} // namespace vknn
