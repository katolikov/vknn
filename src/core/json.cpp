// See json.h. JsonParser method bodies: the permissive recursive descent over trusted engine-emitted JSON.
#include "core/json.h"
#include <cctype>

namespace vknn {

    JsonValue JsonParser::parse(const std::string &s) {
        JsonParser p(s);
        p.ws();
        JsonValue v = p.value();
        return v;
    }

    void JsonParser::ws() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == ','))
        {
            ++i_;
        }
    }

    char JsonParser::peek() {
        return i_ < s_.size() ? s_[i_] : '\0';
    }

    JsonValue JsonParser::value() {
        ws();
        char c = peek();
        if (c == '{')
        {
            return object();
        }
        if (c == '[')
        {
            return array();
        }
        if (c == '"')
        {
            JsonValue v;
            v.type = JsonValue::kString;
            v.str  = str();
            return v;
        }
        if (c == 't' || c == 'f')
        {
            JsonValue v;
            v.type = JsonValue::kBool;
            v.b    = boolean();
            return v;
        }
        if (c == 'n')
        {
            i_ += 4; // consume "null" unchecked; v stays kNull by default
            JsonValue v;
            return v;
        }
        // Any other leading character is assumed to begin a number.
        JsonValue v;
        v.type = JsonValue::kNumber;
        v.num  = number();
        return v;
    }

    JsonValue JsonParser::object() {
        JsonValue v;
        v.type = JsonValue::kObject;
        ++i_; // {
        ws();
        while (peek() != '}' && i_ < s_.size())
        {
            ws();
            std::string key = str();
            ws();
            if (peek() == ':')
            {
                ++i_;
            }
            v.obj[key] = value();
            ws();
        }
        if (peek() == '}')
        {
            ++i_;
        }
        return v;
    }

    JsonValue JsonParser::array() {
        JsonValue v;
        v.type = JsonValue::kArray;
        ++i_; // [
        ws();
        while (peek() != ']' && i_ < s_.size())
        {
            v.arr.push_back(value());
            ws();
        }
        if (peek() == ']')
        {
            ++i_;
        }
        return v;
    }

    std::string JsonParser::str() {
        std::string out;
        if (peek() != '"')
        {
            return out;
        }
        ++i_;
        while (i_ < s_.size() && s_[i_] != '"')
        {
            char c = s_[i_++];
            if (c == '\\' && i_ < s_.size())
            {
                char e = s_[i_++];
                switch (e)
                {
                    case 'n':
                        out += '\n';
                        break;
                    case 't':
                        out += '\t';
                        break;
                    case '"':
                        out += '"';
                        break;
                    case '\\':
                        out += '\\';
                        break;
                    case '/':
                        out += '/';
                        break;
                    default:
                        out += e;
                }
            } else
            {
                out += c;
            }
        }
        if (peek() == '"')
        {
            ++i_;
        }
        return out;
    }

    bool JsonParser::boolean() {
        if (s_.compare(i_, 4, "true") == 0)
        {
            i_ += 4;
            return true;
        }
        if (s_.compare(i_, 5, "false") == 0)
        {
            i_ += 5;
            return false;
        }
        // Malformed bareword (e.g. an unquoted `fast`): consume the token so the enclosing
        // object()/array() makes forward progress instead of spinning forever on the same char.
        while (i_ < s_.size() && isalpha((unsigned char) s_[i_]))
        {
            ++i_;
        }
        return false;
    }

    double JsonParser::number() {
        size_t start = i_;
        while (i_ < s_.size() && (isdigit((unsigned char) s_[i_]) || s_[i_] == '-' || s_[i_] == '+' || s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E'))
        {
            ++i_;
        }
        if (i_ == start)
        {
            ++i_; // no numeric character available (malformed) -> guarantee forward progress
            return 0.0;
        }
        try
        {
            return std::stod(s_.substr(start, i_ - start));
        } catch (...)
        {
            return 0.0; // empty / non-numeric / overflowing span -> permissive 0, never abort config load
        }
    }

} // namespace vknn
