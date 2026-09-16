// Host-side rules of the flat logical GPU kernels (Or, Xor, Not): the broadcast geometry or.comp and
// xor.comp decode, the element count every logical kernel dispatches and its shader index bound, where a
// constant operand's device values come from, and their canonical encoding. The header carries no Vulkan
// types, so tests/test_logical_ops.cpp runs these exact rules on the host against the CPU oracle
// (backend/cpu/logical_ops.h), next to a transcription of the shaders' decode loop.
#pragma once
#include "vknn/dtype.h"
#include "vknn/error.h"
#include "vknn/shape.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
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

    /// Elements a logical kernel dispatches for an output of shape `out`: flatElementCount(out), bounded by
    /// the shader index range. `nodeLabel` (e.g. "Not 'mask'") prefixes the error message.
    /// @throws Error(InvalidArgument) when the count exceeds kMaxShaderElements.
    inline int32_t shaderElementCount(const Shape &out, const std::string &nodeLabel) {
        const int64_t count = flatElementCount(out);
        if (count > kMaxShaderElements)
        {
            throw Error(Status::InvalidArgument, nodeLabel + ": " + std::to_string(count) + " output elements exceed the shader index range");
        }
        return (int32_t) count;
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
        shaderElementCount(out, nodeLabel);
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

    /// Bytes per element of a flat device store at fp16 and at fp32 precision.
    inline constexpr size_t kFp16StoreBytes = 2;
    inline constexpr size_t kFp32StoreBytes = 4;
    /// Element floor of a flat upload's buffer: upload(), uploadWeight() and uploadInit()
    /// (backend/vulkan/ops/vk_op_common.h) allocate max(count, 4) elements.
    inline constexpr int64_t kFlatUploadElementFloor = 4;

    /// Logical byte size of the buffer upload(), uploadWeight() and uploadInit() allocate for a flat
    /// store of `count` elements at the given precision.
    inline size_t flatUploadBytes(int64_t count, bool fp16Storage) {
        return (size_t) std::max<int64_t>(count, kFlatUploadElementFloor) * (fp16Storage ? kFp16StoreBytes : kFp32StoreBytes);
    }

    /// Whether a flat fp16 store keeps the truth of every value an initializer of `dtype` holds. Integer
    /// dtypes do (a nonzero integer has magnitude at least 1, and the store saturates rather than
    /// overflows) and an fp16 payload stores bit-exactly; an fp32 payload does not, since a magnitude below
    /// the smallest fp16 subnormal rounds to zero.
    inline bool fp16StoreKeepsTruth(DType dtype) {
        switch (dtype)
        {
            case DType::Float16:
            case DType::Int32:
            case DType::Int8:
            case DType::UInt8:
            case DType::Int64:
                return true;
            case DType::Float32:
                return false;
        }
        return false;
    }

    /// Where a logical kernel's constant operand reads its values from.
    enum class ConstantOperandSource {
        /// The host payload, decoded by initFloats and uploaded through canonicalConstantOperand.
        CanonicalUpload,
        /// The flat device copy an earlier consumer of the same initializer uploaded (VkOpEnv::lookupFlatWeight).
        SharedDeviceCopy,
    };

    /// Source of constant operand `operandLabel` (shape `shape`, stored dtype `dtype`) for a logical kernel
    /// storing at `fp16Storage` precision.
    ///
    /// A host payload covering the shape (`flatElementCount(shape) * dtypeSize(dtype)` bytes) uploads
    /// canonically. An empty payload on a shape that carries elements was released: an earlier consumer's
    /// uploadInit() drops the host bytes of a large weight once its device copy exists
    /// (VkOpEnv::releaseInitializer), and initFloats() would then decode zeros, reading every element as
    /// false. The kernel reads that device copy instead when its logical byte size `sharedDeviceBytes`
    /// proves a flat store of this shape's elements at the kernel's precision (flatUploadBytes, which also
    /// rules out a store at the other precision or an unconverted raw-lane upload) and, at fp16, the store
    /// kept every value's truth (fp16StoreKeepsTruth). The bit-level truth test reads such a copy's raw
    /// values exactly as it reads the canonical 1.0 / 0.0.
    /// @throws Error(InvalidArgument) naming `operandLabel` when the payload does not cover the shape and no
    ///         such device copy exists, so a missing operand never reads as all false.
    inline ConstantOperandSource constantOperandSource(size_t hostPayloadBytes, const Shape &shape, DType dtype, std::optional<size_t> sharedDeviceBytes, bool fp16Storage, const std::string &operandLabel) {
        const int64_t count         = flatElementCount(shape);
        const size_t  requiredBytes = (size_t) count * dtypeSize(dtype);
        if (hostPayloadBytes >= requiredBytes)
        {
            return ConstantOperandSource::CanonicalUpload;
        }
        const bool released = hostPayloadBytes == 0;
        if (released && sharedDeviceBytes && *sharedDeviceBytes == flatUploadBytes(count, fp16Storage) && (!fp16Storage || fp16StoreKeepsTruth(dtype)))
        {
            return ConstantOperandSource::SharedDeviceCopy;
        }
        std::string message = operandLabel + ": host payload is " + std::to_string(hostPayloadBytes) + " bytes but its " + dtypeStr(dtype) + " " + shapeStr(shape) + " shape needs " + std::to_string(requiredBytes);
        if (released)
        {
            message += " (released after an earlier consumer's device upload, and no flat " + std::string(fp16Storage ? "fp16" : "fp32") + " device copy that keeps every element's truth exists)";
        }
        throw Error(Status::InvalidArgument, message);
    }

    /// Canonical device payload of a constant logical operand: each of the `count` elements becomes
    /// kLogicalTrue when the decoded value is nonzero (NaN included) and kLogicalFalse otherwise.
    ///
    /// The operand is read by truth alone, so encoding it as 1.0 / 0.0 before the fp16-saturating upload
    /// keeps every value's truth at both storage precisions: a raw fp16 upload would round a nonzero
    /// magnitude below the smallest fp16 subnormal to zero. `decoded` is the initFloats() payload of an
    /// operand constantOperandSource() routed to CanonicalUpload (int64 lanes widened; a nonzero int64
    /// never widens to 0.0).
    /// @throws Error(InvalidArgument) when `decoded` holds fewer than `count` elements.
    inline std::vector<float> canonicalConstantOperand(const std::vector<float> &decoded, int64_t count) {
        const size_t elementCount = (size_t) std::max<int64_t>(count, 0);
        if (decoded.size() < elementCount)
        {
            throw Error(Status::InvalidArgument, "canonicalConstantOperand: decoded payload holds " + std::to_string(decoded.size()) + " of " + std::to_string(elementCount) + " elements");
        }
        std::vector<float> canonical(elementCount, kLogicalFalse);
        for (size_t k = 0; k < elementCount; ++k)
        {
            canonical[k] = decoded[k] != 0.0f ? kLogicalTrue : kLogicalFalse;
        }
        return canonical;
    }

}} // namespace vknn::logical
