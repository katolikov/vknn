// Shared basics: Status codes, the Error exception, and small Shape helpers.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace vknn {

    /// Status codes returned across the public API.
    enum class Status {
        Ok = 0,
        InvalidArgument,
        Unsupported,
        NotFound,
        RuntimeError,
        DeviceLost,
        IoError,
    };

    inline const char *statusStr(Status s) {
        switch (s)
        {
            case Status::Ok:
                return "Ok";
            case Status::InvalidArgument:
                return "InvalidArgument";
            case Status::Unsupported:
                return "Unsupported";
            case Status::NotFound:
                return "NotFound";
            case Status::RuntimeError:
                return "RuntimeError";
            case Status::DeviceLost:
                return "DeviceLost";
            case Status::IoError:
                return "IoError";
        }
        return "?";
    }

    /// Lightweight exception carrying a Status (used internally; the public facade
    /// also exposes status-returning variants).
    class Error: public std::runtime_error {
      public:
        Error(Status s, const std::string &msg): std::runtime_error(std::string(statusStr(s)) + ": " + msg), status_(s) {
        }
        Status status() const {
            return status_;
        }

      private:
        Status status_;
    };

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
