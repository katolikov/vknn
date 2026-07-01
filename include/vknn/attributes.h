// The attribute bag: a named map of Attr values with typed getters.
#pragma once
#include "vknn/attr.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vknn {

    struct Attributes {
        std::map<std::string, Attr> map;
        bool                        has(const std::string &k) const {
            return map.count(k) > 0;
        }
        int64_t geti(const std::string &k, int64_t d = 0) const {
            auto it = map.find(k);
            return it == map.end() ? d : it->second.i;
        }
        float getf(const std::string &k, float d = 0) const {
            auto it = map.find(k);
            return it == map.end() ? d : it->second.f;
        }
        const std::vector<int64_t> &getints(const std::string &k) const {
            static const std::vector<int64_t> e;
            auto                              it = map.find(k);
            return it == map.end() ? e : it->second.ints;
        }
        const std::vector<float> &getfloats(const std::string &k) const {
            static const std::vector<float> e;
            auto                            it = map.find(k);
            return it == map.end() ? e : it->second.floats;
        }
        std::string gets(const std::string &k, const std::string &d = "") const {
            auto it = map.find(k);
            return it == map.end() ? d : it->second.str;
        }
    };

} // namespace vknn
