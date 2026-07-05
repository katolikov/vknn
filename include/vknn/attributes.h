// The attribute bag: a named map of Attr values with typed getters.
#pragma once
#include "vknn/attr.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vknn {

    /// A node's attribute set: named Attr values keyed by attribute name, plus typed accessors.
    ///
    /// The importer fills `map` from an ONNX node's attributes; ops read them back through the
    /// getters below. Each getter is total — a missing key yields the supplied default (or an empty
    /// container) rather than throwing — so a caller never has to probe with has() first. A getter
    /// does not verify that the stored Attr::kind matches the requested type; it reads the matching
    /// field of the Attr, which is value-initialized when unset.
    struct Attributes {
        /// Attribute values keyed by ONNX attribute name.
        std::map<std::string, Attr> map;

        /// @returns True when an attribute named `k` is present (regardless of its kind).
        bool has(const std::string &k) const noexcept {
            return map.count(k) > 0;
        }
        /// @returns The scalar-int attribute `k`, or `d` if `k` is absent.
        int64_t geti(const std::string &k, int64_t d = 0) const noexcept {
            auto it = map.find(k);
            return it == map.end() ? d : it->second.i;
        }
        /// @returns The scalar-float attribute `k`, or `d` if `k` is absent.
        float getf(const std::string &k, float d = 0) const noexcept {
            auto it = map.find(k);
            return it == map.end() ? d : it->second.f;
        }
        /// @returns The int-list attribute `k`, or a reference to a shared empty vector if `k` is
        ///          absent. The reference is valid until the owning Attributes is modified.
        const std::vector<int64_t> &getints(const std::string &k) const noexcept {
            static const std::vector<int64_t> e;
            auto                              it = map.find(k);
            return it == map.end() ? e : it->second.ints;
        }
        /// @returns The float-list attribute `k`, or a reference to a shared empty vector if `k` is
        ///          absent. The reference is valid until the owning Attributes is modified.
        const std::vector<float> &getfloats(const std::string &k) const noexcept {
            static const std::vector<float> e;
            auto                            it = map.find(k);
            return it == map.end() ? e : it->second.floats;
        }
        /// @returns A copy of the string attribute `k`, or `d` if `k` is absent.
        std::string gets(const std::string &k, const std::string &d = "") const {
            auto it = map.find(k);
            return it == map.end() ? d : it->second.str;
        }
    };

} // namespace vknn
