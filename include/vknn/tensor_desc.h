// Compile-time description of a tensor in the graph.
#pragma once
#include "vknn/common.h"
#include "vknn/dtype.h"
#include "vknn/tensor_format.h"

namespace vknn {

    /// Compile-time description of a tensor in the graph: its identity (name), logical shape and
    /// dtype, its role at the graph boundary (input/output/initializer), and backend storage hints.
    /// One TensorDesc exists per distinct tensor; ops reference tensors by name.
    struct TensorDesc {
        std::string  name;  ///< Unique tensor name; ops reference their inputs and outputs by this name.
        Shape        shape; ///< Logical NCHW shape. A dynamic dimension is encoded as -1.
        DType        dtype         = DType::Float32; ///< Element type of the logical tensor.
        TensorFormat format        = TensorFormat::NCHW; ///< Logical layout of the tensor (the IR is always NCHW).
        bool         isInput       = false; ///< True for a graph input (a value the caller supplies).
        bool         isOutput      = false; ///< True for a graph output (a value the caller reads back).
        bool         isInitializer = false; ///< True for a constant/weight tensor baked into the model.
        /// Vulkan only: store this tensor as a flat row-major buffer (set by the layout-convert pass for
        /// the generic head ops) instead of the default NC4HW4 packing. Ignored by the CPU backend.
        bool gpuFlat = false;
        /// Vulkan only: keep this activation's buffer in fp32 even when the segment runs fp16, so a
        /// precision-critical sub-graph (the geometry tail) does not lose accuracy to fp16 storage. Set by
        /// the markFp32 pass from Config::fp32Tensors at load time (not serialized). The producing op runs
        /// its fp32 kernel variant; a convert_dtype node bridges the fp16/fp32 frontier.
        bool storeFp32 = false;
    };

} // namespace vknn
