// ONNX ArgMin (opset 13): the int64 index of the smallest element along `axis` (default 0), with
// keepdims (default 1) and select_last_index (default 0). The selection rule, its NaN / signed-zero /
// tie consequences and the error cases live with the shared scan in backend/cpu/arg_extreme.h.
#include "backend/cpu/arg_extreme.h"

namespace vknn {
    namespace {

        struct ArgMinCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                static constexpr bool kSelectLargest = false;
                cpu::runArgExtreme(node, ctx, kSelectLargest);
            }
        };

    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::ArgMin, ArgMinCpu);
} // namespace vknn
