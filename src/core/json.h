// vxrt — minimal dependency-free JSON parser (objects/arrays/strings/numbers/bool/null).
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "vx/common.h"

namespace vx {

struct JsonValue {
  enum Type { kNull, kBool, kNumber, kString, kArray, kObject } type = kNull;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<JsonValue> arr;
  std::map<std::string, JsonValue> obj;

  bool isObject() const { return type == kObject; }
  const JsonValue* get(const std::string& k) const {
    if (type != kObject) return nullptr;
    auto it = obj.find(k);
    return it == obj.end() ? nullptr : &it->second;
  }
  double asNum(double d = 0) const { return type == kNumber ? num : d; }
  bool asBool(bool d = false) const { return type == kBool ? b : d; }
  std::string asStr(const std::string& d = "") const { return type == kString ? str : d; }
};

class JsonParser {
 public:
  static JsonValue parse(const std::string& s) {
    JsonParser p(s);
    p.ws();
    JsonValue v = p.value();
    return v;
  }

 private:
  explicit JsonParser(const std::string& s) : s_(s) {}
  const std::string& s_;
  size_t i_ = 0;

  void ws() {
    while (i_ < s_.size() &&
           (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == ','))
      ++i_;
  }
  char peek() { return i_ < s_.size() ? s_[i_] : '\0'; }

  JsonValue value() {
    ws();
    char c = peek();
    if (c == '{') return object();
    if (c == '[') return array();
    if (c == '"') {
      JsonValue v;
      v.type = JsonValue::kString;
      v.str = str();
      return v;
    }
    if (c == 't' || c == 'f') {
      JsonValue v;
      v.type = JsonValue::kBool;
      v.b = boolean();
      return v;
    }
    if (c == 'n') {
      i_ += 4;
      JsonValue v;
      return v;
    }
    JsonValue v;
    v.type = JsonValue::kNumber;
    v.num = number();
    return v;
  }
  JsonValue object() {
    JsonValue v;
    v.type = JsonValue::kObject;
    ++i_;  // {
    ws();
    while (peek() != '}' && i_ < s_.size()) {
      ws();
      std::string key = str();
      ws();
      if (peek() == ':') ++i_;
      v.obj[key] = value();
      ws();
    }
    if (peek() == '}') ++i_;
    return v;
  }
  JsonValue array() {
    JsonValue v;
    v.type = JsonValue::kArray;
    ++i_;  // [
    ws();
    while (peek() != ']' && i_ < s_.size()) {
      v.arr.push_back(value());
      ws();
    }
    if (peek() == ']') ++i_;
    return v;
  }
  std::string str() {
    std::string out;
    if (peek() != '"') return out;
    ++i_;
    while (i_ < s_.size() && s_[i_] != '"') {
      char c = s_[i_++];
      if (c == '\\' && i_ < s_.size()) {
        char e = s_[i_++];
        switch (e) {
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
        out += c;
    }
    if (peek() == '"') ++i_;
    return out;
  }
  bool boolean() {
    if (s_.compare(i_, 4, "true") == 0) {
      i_ += 4;
      return true;
    }
    if (s_.compare(i_, 5, "false") == 0) {
      i_ += 5;
      return false;
    }
    return false;
  }
  double number() {
    size_t start = i_;
    while (i_ < s_.size() && (isdigit(s_[i_]) || s_[i_] == '-' || s_[i_] == '+' || s_[i_] == '.' ||
                              s_[i_] == 'e' || s_[i_] == 'E'))
      ++i_;
    return std::stod(s_.substr(start, i_ - start));
  }
};

}  // namespace vx
