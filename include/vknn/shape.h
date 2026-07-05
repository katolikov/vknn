// Small Shape helpers: the Shape alias plus element-count and formatting utilities.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace vknn {

    /// A tensor shape: one extent per dimension, outermost first (e.g. {1, 3, 224, 224} for NCHW).
    /// Rank equals size(); an empty Shape denotes a rank-0 (scalar) tensor.
    using Shape = std::vector<int64_t>;

    /// Total number of elements, i.e. the product of every dimension extent.
    /// An empty (rank-0 scalar) shape returns 0, not 1: call sites that need a scalar to count as one
    /// element clamp the result themselves (e.g. std::max<int64_t>(numElements(s), 1)).
    /// @param s Shape to measure.
    /// @returns The element count, or 0 when @p s is empty.
    inline int64_t numElements(const Shape &s) {
        int64_t n = 1;
        for (int64_t d: s)
        {
            n *= d;
        }
        return s.empty() ? 0 : n;
    }

    /// Format a shape as a bracketed, comma-separated list for logs and diagnostics
    /// (e.g. {1, 3, 224} renders as "[1,3,224]"; an empty shape renders as "[]").
    /// @param s Shape to format.
    /// @returns The bracketed string representation.
    inline std::string shapeStr(const Shape &s) {
        std::string out = "[";
        for (size_t i = 0; i < s.size(); ++i)
        {
            out += std::to_string(s[i]);
            if (i + 1 < s.size())
            {
                out += ",";
            }
        }
        return out + "]";
    }

} // namespace vknn
