// Host-side rules of the flat logical GPU kernels (Or, Xor, Not): the broadcast geometry or.comp and
// xor.comp decode, the element count every logical kernel dispatches, and the canonical encoding of a
// constant operand. The header carries no Vulkan types, so tests/test_logical_ops.cpp runs these exact
// rules on the host against the CPU oracle (backend/cpu/logical_ops.h), next to a transcription of the
// shaders' decode loop.
#pragma once
#include "vknn/error.h"
#include "vknn/shape.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace vknn { namespace logical {

    /// Largest element count the kernels index: the shaders decode the invocation id through GLSL `int`.
    inline constexpr int64_t kMaxShaderElements = std::numeric_limits<int32_t>::max();

    /// Canonical encoding of a boolean value on the GPU, exact at fp16 and fp32 storage.
    inline constexpr float kLogicalTrue  = 1.0f;
    inline constexpr float kLogicalFalse = 0.0f;

    /// Geometry SSBO payload of or.comp / xor.comp: three rank-length arrays packed back to back in this
    /// order by flat::uploadFlatGeom, read by the shaders as g[array * rank + axis] with array 0 = outDim,
    /// 1 = aStride, 2 = bStride.
    struct FlatBroadcastGeometry {
        int                  rank = 0;
        std::vector<int32_t> outDim;
        std::vector<int32_t> aStride;
        std::vector<int32_t> bStride;
    };

    /// Elements a logical kernel writes for an output of shape `shape`: a rank-0 result carries one
    /// element (numElements() reports 0 for it); a zero extent writes nothing.
    inline int64_t flatElementCount(const Shape &shape) {
        return shape.empty() ? 1 : numElements(shape);
    }

    /// Row-major strides of `operand` right-aligned into the axes of `out`, 0 on every axis where the
    /// operand's extent is 1 so each output coordinate along that axis re-reads the single source element
    /// (the broadcast convention of flat::Binary and the CPU BroadcastWalk). `nodeLabel` (e.g. "Or 'mask'")
    /// prefixes every error message.
    /// @throws Error(InvalidArgument) when the operand has more axes than the output, or an operand extent
    ///         is neither 1 nor the output extent on its axis (the shader would read past the operand).
    inline std::vector<int32_t> broadcastStrides(const Shape &operand, const Shape &out, const std::string &nodeLabel) {
        const int rank       = (int) out.size();
        const int leadingPad = rank - (int) operand.size();
        if (leadingPad < 0)
        {
            throw Error(Status::InvalidArgument, nodeLabel + ": operand " + shapeStr(operand) + " has more axes than output " + shapeStr(out));
        }
        std::vector<int32_t> strides((size_t) rank, 0);
        int64_t              running = 1;
        for (int axis = rank - 1; axis >= 0; --axis)
        {
            const int64_t extent = axis < leadingPad ? 1 : operand[(size_t) (axis - leadingPad)];
            if (extent != 1 && extent != out[(size_t) axis])
            {
                throw Error(Status::InvalidArgument, nodeLabel + ": operand " + shapeStr(operand) + " does not broadcast to output " + shapeStr(out));
            }
            strides[(size_t) axis] = extent == 1 ? 0 : (int32_t) running;
            running *= extent;
        }
        return strides;
    }

    /// Broadcast geometry of a two-operand logical op writing `out` from operands shaped `a` and `b`;
    /// `nodeLabel` (e.g. "Or 'mask'") prefixes every error message.
    /// @throws Error(InvalidArgument) when an operand does not broadcast to the output or the output
    ///         exceeds the shader index range.
    inline FlatBroadcastGeometry flatBroadcastGeometry(const Shape &out, const Shape &a, const Shape &b, const std::string &nodeLabel) {
        if (flatElementCount(out) > kMaxShaderElements)
        {
            throw Error(Status::InvalidArgument, nodeLabel + ": " + std::to_string(flatElementCount(out)) + " output elements exceed the shader index range");
        }
        FlatBroadcastGeometry geometry;
        geometry.rank = (int) out.size();
        geometry.outDim.reserve(out.size());
        for (int64_t extent: out)
        {
            geometry.outDim.push_back((int32_t) extent);
        }
        geometry.aStride = broadcastStrides(a, out, nodeLabel);
        geometry.bStride = broadcastStrides(b, out, nodeLabel);
        return geometry;
    }

    /// Canonical device payload of a constant logical operand: each of the `count` elements becomes
    /// kLogicalTrue when the decoded value is nonzero (NaN included) and kLogicalFalse otherwise.
    ///
    /// The operand is read by truth alone, so encoding it as 1.0 / 0.0 before the fp16-saturating upload
    /// keeps every value's truth at both storage precisions: a raw fp16 upload would round a nonzero
    /// magnitude below the smallest fp16 subnormal to zero. `decoded` is the initFloats() payload (int64
    /// lanes widened; a nonzero int64 never widens to 0.0); a shorter payload reads as false past its end.
    inline std::vector<float> canonicalConstantOperand(const std::vector<float> &decoded, int64_t count) {
        std::vector<float> canonical((size_t) std::max<int64_t>(count, 0), kLogicalFalse);
        for (size_t k = 0; k < canonical.size() && k < decoded.size(); ++k)
        {
            canonical[k] = decoded[k] != 0.0f ? kLogicalTrue : kLogicalFalse;
        }
        return canonical;
    }

}} // namespace vknn::logical
