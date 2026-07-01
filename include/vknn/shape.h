// Small Shape helpers: the Shape alias plus element-count and formatting utilities.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace vknn {

    using Shape = std::vector<int64_t>;

    inline int64_t numElements(const Shape &s) {
        int64_t n = 1;
        for (int64_t d: s)
        {
            n *= d;
        }
        return s.empty() ? 0 : n;
    }

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
