// ArgMax / ArgMin GPU plan: the device-free half of the kernel (shaders/arg_extreme.comp). It turns a
// node and its graph descriptors into the push constants, the specialization constants and the
// storage-precision variant, and states every precondition the kernel relies on as a named Error.
// Keeping it free of Vulkan types lets tests/test_arg_extreme_ops.cpp check it on the host against
// the plan the load-time passes actually build.
//
// Storage precision. The indices output is pinned fp32 by pinIntegerResultsFp32, so the node runs
// its fp32 configuration and the shader writes `float indices[]`. The data (input 0) is NOT bridged
// by markFp32 (its ArgMax/ArgMin input-0 exemption), so it arrives at its own storage precision: the
// fp16 variant, which reads float16_t, is selected exactly when that storage is fp16, i.e. under an
// fp16 base precision for a runtime tensor that is not itself pinned fp32. A constant (initializer)
// data operand is uploaded fp32 by the op and therefore reads through the fp32 variant.
#pragma once
#include "core/arg_extreme_limits.h"
#include "vknn/error.h"
#include "vknn/graph.h"
#include "vknn/op.h"
#include <cstdint>
#include <string>
#include <vector>

namespace vknn {

    /// Shader stem of the kernel; the fp16-data variant appends the usual "_fp16" suffix.
    inline constexpr const char *kArgExtremeShaderStem = "arg_extreme";
    /// Storage buffers the kernel binds: data (binding 0) and indices (binding 1).
    inline constexpr uint32_t kArgExtremeBufferCount = 2;
    /// ONNX attribute defaults for ArgMax / ArgMin.
    inline constexpr int64_t kArgExtremeOnnxDefaultAxis            = 0;
    inline constexpr int64_t kArgExtremeOnnxDefaultSelectLastIndex = 0;

    /// Push-constant block, byte-matched to shaders/arg_extreme.comp.
    struct ArgExtremePushConstants {
        int32_t outer;  // product of the dims before the axis
        int32_t extent; // axis length
        int32_t inner;  // product of the dims after the axis
        int32_t total;  // output elements = outer * inner
    };

    /// Everything the Vulkan op needs to build its pipeline and dispatch.
    struct ArgExtremePlan {
        ArgExtremePushConstants push {};
        bool                    selectLast   = false; // select_last_index
        bool                    dataFp16     = false; // the data is stored fp16: take the arg_extreme_fp16 variant
        bool                    constantData = false; // the data is an initializer the op uploads fp32
    };

    /// Specialization constants in constant_id order: SELECT_LARGEST (1 = ArgMax), SELECT_LAST.
    inline std::vector<uint32_t> argExtremeSpecConstants(bool selectLargest, bool selectLast) {
        return {selectLargest ? 1u : 0u, selectLast ? 1u : 0u};
    }

    /// Build the plan for an ArgMax / ArgMin node at segment base precision `baseFp16`. Throws Error
    /// naming the node when the node is malformed (the vkNodeGate refusals: missing operand,
    /// unresolved or rank-0 input, axis out of range, empty axis), when the geometry exceeds what the
    /// kernel addresses or stores exactly (argExtremeGpuGeometryRefusal, which the gate applies too),
    /// when the output descriptor disagrees with the reduced shape, when a runtime data tensor or the
    /// indices output is not flat, when an fp16 base precision reaches a node whose indices are not
    /// fp32-stored, or when a constant data operand no longer holds the payload its shape needs. The
    /// gate refuses the malformed and oversized forms, so those throws fire only for a graph assigned
    /// to the GPU without it.
    inline ArgExtremePlan planArgExtreme(const Graph &g, const Node &node, bool baseFp16) {
        const std::string where = std::string(opTypeName(node.type)) + " '" + node.name + "': ";
        if (node.inputs.empty() || node.inputs[0] == kNoTensor || node.outputs.empty() || node.outputs[0] == kNoTensor)
        {
            throw Error(Status::InvalidArgument, where + "needs one data input and one indices output");
        }
        const TensorId data   = node.inputs[0];
        const TensorId output = node.outputs[0];
        const Shape   &shape  = g.desc(data).shape;
        if (shape.empty())
        {
            throw Error(Status::InvalidArgument, where + "the data shape is rank-0 or unresolved; the kernel needs an axis");
        }
        const int64_t rank          = (int64_t) shape.size();
        const int64_t attributeAxis = node.attr.geti("axis", kArgExtremeOnnxDefaultAxis);
        if (attributeAxis < -rank || attributeAxis >= rank)
        {
            throw Error(Status::InvalidArgument, where + "axis " + std::to_string(attributeAxis) + " is out of range for a rank-" + std::to_string(rank) + " input");
        }
        const int64_t axis   = attributeAxis < 0 ? attributeAxis + rank : attributeAxis;
        const int64_t extent = shape[(size_t) axis];
        if (extent <= 0)
        {
            throw Error(Status::InvalidArgument, where + "axis " + std::to_string(attributeAxis) + " has no element to select");
        }
        const ArgExtremeGeometry geometry = argExtremeGeometry(shape, axis);
        if (const char *refusal = argExtremeGpuGeometryRefusal(geometry))
        {
            throw Error(Status::Unsupported, where + refusal + " (data shape " + shapeStr(shape) + ", axis " + std::to_string(attributeAxis) + ")");
        }
        const int64_t outer = geometry.outer, inner = geometry.inner;
        const int64_t total = outer * inner;
        if (numElements(g.desc(output).shape) != total)
        {
            throw Error(Status::InvalidArgument, where + "the indices descriptor holds " + std::to_string(numElements(g.desc(output).shape)) + " elements, the reduction produces " + std::to_string(total));
        }
        // The kernel addresses both buffers row-major. A runtime data tensor or an indices output left in
        // the NC4HW4 layout (a session whose flat layout pass did not run) would be read or written at
        // the wrong offsets, so the plan refuses it instead.
        if (!g.isInitializer(data) && !g.desc(data).gpuFlat)
        {
            throw Error(Status::RuntimeError, where + "the data is not in the flat row-major layout the kernel reads");
        }
        if (!g.desc(output).gpuFlat)
        {
            throw Error(Status::RuntimeError, where + "the indices output is not in the flat row-major layout the kernel writes");
        }
        if (baseFp16 && !g.desc(output).storeFp32)
        {
            throw Error(Status::RuntimeError, where + "the indices output is not fp32-stored under fp16 base precision; the kernel writes fp32 indices");
        }
        // A constant data operand is decoded from its host payload at prepare. An earlier weight upload
        // of the same tensor may have released those bytes (VkOpEnv::releaseInitializer), and a decode
        // of the emptied payload would scan zeros, so a short payload is a named error instead.
        if (g.isInitializer(data))
        {
            const size_t payloadBytes  = g.initializers.at(data).bytes.size();
            const size_t requiredBytes = (size_t) numElements(shape) * dtypeSize(g.desc(data).dtype);
            if (payloadBytes < requiredBytes)
            {
                throw Error(Status::RuntimeError,
                            where + "the constant data payload holds " + std::to_string(payloadBytes) + " bytes, its " + dtypeStr(g.desc(data).dtype) + " " + shapeStr(shape) + " shape needs " + std::to_string(requiredBytes) + " (released before this op read it)");
            }
        }

        ArgExtremePlan plan;
        plan.push.outer   = (int32_t) outer;
        plan.push.extent  = (int32_t) extent;
        plan.push.inner   = (int32_t) inner;
        plan.push.total   = (int32_t) total;
        plan.selectLast   = node.attr.geti("select_last_index", kArgExtremeOnnxDefaultSelectLastIndex) != 0;
        plan.constantData = g.isInitializer(data);
        plan.dataFp16     = baseFp16 && !g.desc(data).storeFp32 && !plan.constantData;
        return plan;
    }

} // namespace vknn
