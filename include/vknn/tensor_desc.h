// Compile-time description of a tensor in the graph.
#pragma once
#include "vknn/common.h"
#include "vknn/dtype.h"
#include "vknn/tensor_format.h"

namespace vknn {

    /// Compile-time description of a tensor in the graph.
    struct TensorDesc {
        std::string  name;
        Shape        shape; // logical NCHW shape (may have dynamic dims as -1)
        DType        dtype         = DType::Float32;
        TensorFormat format        = TensorFormat::NCHW;
        bool         isInput       = false;
        bool         isOutput      = false;
        bool         isInitializer = false;
        // Vulkan only: store this tensor as a flat row-major buffer (set by the layout-convert pass for
        // the generic head ops) instead of the default NC4HW4 packing. Ignored by the CPU backend.
        bool gpuFlat = false;
        // Vulkan only: keep this activation's buffer in fp32 even when the segment runs fp16, so a
        // precision-critical sub-graph (the geometry tail) does not lose accuracy to fp16 storage. Set by
        // the markFp32 pass from Config::fp32Tensors at load time (not serialized). The producing op runs
        // its fp32 kernel variant; a convert_dtype node bridges the fp16/fp32 frontier.
        bool storeFp32 = false;
    };

} // namespace vknn
