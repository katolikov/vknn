// Minimal dependency-free JSON parser (objects/arrays/strings/numbers/bool/null).
//
// Permissive by design: it parses the trusted config and model-metadata JSON the engine emits, so it
// skips validation rather than reporting errors. Commas are treated as whitespace (see ws()), missing
// ':' or closing brackets are tolerated, and malformed input yields best-effort values instead of an
// error. std::stod in number() is the only operation that can throw.
#pragma once
#include "vknn/common.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vknn {

    // A parsed JSON node. `type` selects which payload field is meaningful; the others hold their
    // default. Numbers are always stored as double (no integer variant), and object keys are ordered
    // by std::map, not by source order.
    struct JsonValue {
        enum Type { kNull, kBool, kNumber, kString, kArray, kObject } type = kNull;
        bool                             b                                 = false;
        double                           num                               = 0;
        std::string                      str;
        std::vector<JsonValue>           arr;
        std::map<std::string, JsonValue> obj;

        bool isObject() const {
            return type == kObject;
        }
        // Look up a child by key; returns nullptr on a non-object or a missing key, so callers can
        // chain get()->as*() with a default for absent fields.
        const JsonValue *get(const std::string &k) const {
            if (type != kObject)
            {
                return nullptr;
            }
            auto it = obj.find(k);
            return it == obj.end() ? nullptr : &it->second;
        }
        // as*() accessors return the fallback `d` on any type mismatch, never coercing across types.
        double asNum(double d = 0) const {
            return type == kNumber ? num : d;
        }
        bool asBool(bool d = false) const {
            return type == kBool ? b : d;
        }
        std::string asStr(const std::string &d = "") const {
            return type == kString ? str : d;
        }
    };

    class JsonParser {
      public:
        static JsonValue parse(const std::string &s);

      private:
        explicit JsonParser(const std::string &s): s_(s) {
        }
        const std::string &s_;
        size_t             i_ = 0;

        // Skips whitespace AND commas, so object()/array() never handle separators explicitly: every
        // element boundary is just more skippable whitespace. Also makes trailing/leading commas
        // harmless.
        void ws();
        char peek();

        JsonValue value();
        JsonValue object();
        JsonValue array();
        // Parses a double-quoted string, decoding the common backslash escapes. \uXXXX is NOT decoded:
        // an unrecognized escape (default case) emits the character after the backslash verbatim.
        std::string str();
        bool        boolean();
        double      number();
    };

} // namespace vknn
