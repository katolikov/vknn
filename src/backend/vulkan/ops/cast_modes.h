// cast.comp's `mode` push-constant values. shaders/cast.comp declares the same kCastMode* constants
// with the same values; tests/test_binary_int_and_cast_fixes.cpp reads the shader source and checks
// each declaration against this header.
#pragma once

namespace vknn {

    /// Truncate, then clamp to [lo, hi]: the wide targets (INT32/INT64/UINT32/UINT64, clamped only to
    /// fence off inf/NaN) and INT16/UINT16 (saturated to their range).
    inline constexpr int kCastModeWide = 0;
    /// Truncate, then wrap modulo 2^8 into [-128, 127] (INT8).
    inline constexpr int kCastModeInt8Wrap = 1;
    /// Truncate, then saturate to [lo, hi] = [0, 255] (UINT8).
    inline constexpr int kCastModeUInt8Saturate = 2;
    /// Truth test on the untruncated value: 1 for any nonzero value (NaN included), 0 for +0 / -0 (BOOL).
    inline constexpr int kCastModeBool = 3;

} // namespace vknn
