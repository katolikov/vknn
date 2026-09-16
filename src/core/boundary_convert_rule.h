// Boundary dtype contract between a caller's declared I/O bytes and the GPU boundary buffers: which
// dtype pairs the boundary_convert shader converts on the device, which declared input dtypes a
// whole-GPU run hands to that conversion as raw bytes, the storage dtype of a boundary buffer, and the
// exact host decode every GPU conversion reproduces. Pure host code (no Vulkan types), shared by the
// Session (bindInput and the raw-byte staging decision), the Vulkan segment (staging, dma-buf
// conversion, host upload) and BoundaryConvert (variant selection), so every rule unit-tests on the
// CPU-only host build.
//
// boundary_convert compiles one SPIR-V variant per (source, destination) storage dtype pair listed in
// kBoundaryConvertVariants; CMakeLists.txt's bc_variant lines build exactly that set. A pair outside
// the set has no variant: selecting one is an error, never a silent substitution of another variant
// (a 1-byte payload read through the fp32 variant misdecodes every element and reads past the buffer).
// Int32 and Int64 have no storage lane in the shader.
//
// Conversion semantics (bit-exact between decodeHostLanesToFloat32 and the variants):
//   uint8 / int8 source -> fp32 / fp16 device: the integer value, zero- or sign-extended (never
//     normalized); exact in both storage precisions.
//   fp32 / fp16 device -> uint8 destination: truncate toward zero, saturate to [0, 255].
//   fp32 / fp16 device -> int8 destination: truncate toward zero, wrap modulo 2^8 into [-128, 127].
// The destination rules are the Session's readbackOutput narrowing of the lane's int64 value, which is
// defined for every fp32 value: a NaN reads 0 and a value at or beyond +-2^63 (infinities included)
// saturates to INT64_MAX / INT64_MIN before the narrowing (uint8: 255 / 0, int8: -1 / 0).
//
// tests/test_boundary_int8_staging.cpp pins this.
#pragma once
#include "vknn/dtype.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace vknn {

    /// One compiled boundary_convert variant: the storage dtype it reads and the one it writes.
    struct BoundaryConvertVariant {
        DType source;
        DType destination;
    };

    /// Every (source, destination) pair boundary_convert compiles, in CMakeLists.txt bc_variant order.
    inline constexpr BoundaryConvertVariant kBoundaryConvertVariants[] = {
        {DType::Float32, DType::Float32}, {DType::Float32, DType::Float16}, {DType::Float16, DType::Float32}, {DType::Float16, DType::Float16},
        {DType::UInt8, DType::Float16},   {DType::UInt8, DType::Float32},   {DType::Float16, DType::UInt8},   {DType::Float32, DType::UInt8},
        {DType::UInt8, DType::UInt8},     {DType::Int8, DType::Float16},    {DType::Int8, DType::Float32},    {DType::Float16, DType::Int8},
        {DType::Float32, DType::Int8},
    };

    /// Variant-name tag of a boundary storage dtype ("f32", "f16", "u8", "i8"); nullptr for a dtype the
    /// shader has no storage lane for (Int32, Int64, an unrecognized value).
    inline const char *boundaryConvertDtypeTag(DType dtype) noexcept {
        switch (dtype)
        {
            case DType::Float32:
                return "f32";
            case DType::Float16:
                return "f16";
            case DType::UInt8:
                return "u8";
            case DType::Int8:
                return "i8";
            case DType::Int32:
            case DType::Int64:
                return nullptr;
        }
        return nullptr;
    }

    /// True when boundary_convert compiles a variant reading `source` and writing `destination`.
    inline bool boundaryConvertHasVariant(DType source, DType destination) noexcept {
        for (const BoundaryConvertVariant &variant: kBoundaryConvertVariants)
        {
            if (variant.source == source && variant.destination == destination)
            {
                return true;
            }
        }
        return false;
    }

    /// Embedded SPIR-V name of the variant for the pair ("boundary_convert_<src>_<dst>"); empty when
    /// boundaryConvertHasVariant is false.
    inline std::string boundaryConvertVariantName(DType source, DType destination) {
        if (!boundaryConvertHasVariant(source, destination))
        {
            return {};
        }
        return std::string("boundary_convert_") + boundaryConvertDtypeTag(source) + "_" + boundaryConvertDtypeTag(destination);
    }

    /// True for a dtype stored in 8-bit lanes.
    inline bool isEightBitDtype(DType dtype) noexcept {
        return dtype == DType::UInt8 || dtype == DType::Int8;
    }

    /// True when the pair's variant reads or writes 8-bit lanes, so the device must enable both 8-bit
    /// storage buffers and the 8-bit integer shader type to run it.
    inline bool boundaryConvertNeedsEightBitStorage(DType source, DType destination) noexcept {
        return isEightBitDtype(source) || isEightBitDtype(destination);
    }

    /// True when a device with these capabilities runs the pair's variant.
    inline bool boundaryConvertDeviceSupports(DType source, DType destination, bool storage8bit, bool shaderInt8) noexcept {
        return boundaryConvertHasVariant(source, destination) && (!boundaryConvertNeedsEightBitStorage(source, destination) || (storage8bit && shaderInt8));
    }

    /// Storage dtype of a GPU boundary buffer: fp16 at a half-precision segment unless the tensor is
    /// pinned to fp32 (storeFp32), else fp32.
    inline DType boundaryDeviceDtype(bool segmentFp16, bool storeFp32) noexcept {
        return segmentFp16 && !storeFp32 ? DType::Float16 : DType::Float32;
    }

    /// True when a whole-GPU run keeps a caller input declared as `declared` in its raw bytes (the
    /// RtTensor keeps the declared dtype) for the GPU staging conversion instead of decoding it on the
    /// host. Every such dtype has a variant to both device storage dtypes, and the Vulkan host upload
    /// decodes it when the staging conversion does not run.
    inline bool boundaryStagesRawInputBytes(DType declared) noexcept {
        return isEightBitDtype(declared);
    }

    /// Bytes one host lane of `dtype` occupies in a caller buffer: its native width, and the fp32 width
    /// for an unrecognized value (which decodes as fp32).
    inline size_t boundaryHostLaneBytes(DType dtype) noexcept {
        const size_t width = dtypeSize(dtype);
        return width > 0 ? width : sizeof(float);
    }

    /// Decode `count` host lanes of `dtype` (native byte order, boundaryHostLaneBytes each) into fp32
    /// values: fp16 widens exactly, integers convert to their nearest float value (exact within 2^24),
    /// fp32 and an unrecognized dtype copy the bytes. `bytes` must hold count * boundaryHostLaneBytes.
    inline void decodeHostLanesToFloat32(DType dtype, const uint8_t *bytes, int64_t count, float *out) {
        if (count <= 0)
        {
            return;
        }
        auto load = [&](int64_t index, void *lane, size_t width) {
            std::memcpy(lane, bytes + (size_t) index * width, width);
        };
        switch (dtype)
        {
            case DType::Float16:
                for (int64_t i = 0; i < count; ++i)
                {
                    fp16_t lane;
                    load(i, &lane, sizeof(lane));
                    out[i] = halfToFloat(lane);
                }
                return;
            case DType::UInt8:
                for (int64_t i = 0; i < count; ++i)
                {
                    out[i] = (float) bytes[i];
                }
                return;
            case DType::Int8:
                for (int64_t i = 0; i < count; ++i)
                {
                    int8_t lane;
                    load(i, &lane, sizeof(lane));
                    out[i] = (float) lane;
                }
                return;
            case DType::Int32:
                for (int64_t i = 0; i < count; ++i)
                {
                    int32_t lane;
                    load(i, &lane, sizeof(lane));
                    out[i] = (float) lane;
                }
                return;
            case DType::Int64:
                for (int64_t i = 0; i < count; ++i)
                {
                    int64_t lane;
                    load(i, &lane, sizeof(lane));
                    out[i] = (float) lane;
                }
                return;
            case DType::Float32:
                break;
        }
        std::memcpy(out, bytes, (size_t) count * sizeof(float));
    }

} // namespace vknn
