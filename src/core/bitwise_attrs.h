// Attribute contract of the integer bitwise ops (BitShift, BitwiseAnd/Or/Xor, BitwiseNot), shared by the
// ONNX importer that stamps the integer width, the CPU oracle and the Vulkan kernels that read it.
//
// The IR carries integer activations as int64 or as fp32 lanes and keeps no record of the ONNX element
// type, so the importer resolves it and stamps `int_bits` / `int_signed` on BitShift and BitwiseNot
// nodes (the two ops whose result depends on the width). An absent attribute reads as the int64 default.
#pragma once
#include "vknn/error.h"
#include "vknn/op.h"
#include <cstdint>
#include <string>

namespace vknn { namespace bitwise {

    /// Integer width in bits of the operand element type (8, 16, 32 or 64).
    inline constexpr const char *kIntBitsAttr = "int_bits";
    /// 1 when the operand element type is signed, 0 when unsigned (BOOL counts as unsigned).
    inline constexpr const char *kIntSignedAttr = "int_signed";
    /// BitShift direction attribute: the string "LEFT" or "RIGHT" (ONNX spelling, case-sensitive).
    inline constexpr const char *kDirectionAttr  = "direction";
    inline constexpr const char *kDirectionLeft  = "LEFT";
    inline constexpr const char *kDirectionRight = "RIGHT";

    inline constexpr int     kInt8Bits         = 8;
    inline constexpr int     kInt16Bits        = 16;
    inline constexpr int     kInt32Bits        = 32;
    inline constexpr int     kInt64Bits        = 64;
    inline constexpr int64_t kDefaultIntBits   = kInt64Bits; ///< width when `int_bits` is absent
    inline constexpr int64_t kDefaultIntSigned = 1;          ///< signedness when `int_signed` is absent

    /// Width and signedness of a node's integer operands.
    struct IntegerWidth {
        int  bits     = kInt64Bits;
        bool isSigned = true;
    };

    /// Shift direction of a BitShift node.
    enum class ShiftDirection {
        Left,
        Right,
    };

    /// Requirement text of each width attribute: vkNodeGate's refusal reason for a node that violates it,
    /// and the message of the kernels' InvalidArgument.
    inline constexpr const char *kIntBitsRequirement   = "int_bits must be 8, 16, 32 or 64";
    inline constexpr const char *kIntSignedRequirement = "int_signed must be 0 or 1";

    /// True for an `int_bits` value of 8, 16, 32 or 64.
    inline constexpr bool intBitsValid(int64_t bits) noexcept {
        return bits == kInt8Bits || bits == kInt16Bits || bits == kInt32Bits || bits == kInt64Bits;
    }
    /// True for an `int_signed` value of 0 or 1.
    inline constexpr bool intSignedValid(int64_t isSigned) noexcept {
        return isSigned == 0 || isSigned == 1;
    }

    /// Check `int_bits` / `int_signed` (absent attributes read as their defaults) without throwing.
    /// @returns true when both are valid; otherwise false, with `requirement` (when non-null) set to the
    ///          violated attribute's requirement text.
    inline bool integerWidthValid(const Node &node, const char **requirement) noexcept {
        const char *violated = nullptr;
        if (!intBitsValid(node.attr.geti(kIntBitsAttr, kDefaultIntBits)))
        {
            violated = kIntBitsRequirement;
        } else if (!intSignedValid(node.attr.geti(kIntSignedAttr, kDefaultIntSigned)))
        { violated = kIntSignedRequirement; }
        if (requirement)
        {
            *requirement = violated;
        }
        return violated == nullptr;
    }

    /// Read `int_bits` / `int_signed` with their defaults.
    /// @throws Error(InvalidArgument) naming the node when `int_bits` is not 8, 16, 32 or 64 or
    ///         `int_signed` is not 0 or 1.
    inline IntegerWidth readIntegerWidth(const Node &node) {
        const int64_t bits     = node.attr.geti(kIntBitsAttr, kDefaultIntBits);
        const int64_t isSigned = node.attr.geti(kIntSignedAttr, kDefaultIntSigned);
        if (!intBitsValid(bits))
        {
            throw Error(Status::InvalidArgument, std::string(opTypeName(node.type)) + " '" + node.name + "': " + kIntBitsRequirement + " (got " + std::to_string(bits) + ")");
        }
        if (!intSignedValid(isSigned))
        {
            throw Error(Status::InvalidArgument, std::string(opTypeName(node.type)) + " '" + node.name + "': " + kIntSignedRequirement + " (got " + std::to_string(isSigned) + ")");
        }
        IntegerWidth width;
        width.bits     = (int) bits;
        width.isSigned = isSigned != 0;
        return width;
    }

    /// Read a BitShift node's `direction` (a string attribute, so read with gets, never geti).
    /// @throws Error(InvalidArgument) naming the node when it is not exactly "LEFT" or "RIGHT".
    inline ShiftDirection readShiftDirection(const Node &node) {
        const std::string direction = node.attr.gets(kDirectionAttr, "");
        if (direction == kDirectionLeft)
        {
            return ShiftDirection::Left;
        }
        if (direction == kDirectionRight)
        {
            return ShiftDirection::Right;
        }
        throw Error(Status::InvalidArgument, std::string(opTypeName(node.type)) + " '" + node.name + "': " + kDirectionAttr + " must be LEFT or RIGHT (got '" + direction + "')");
    }

}} // namespace vknn::bitwise
