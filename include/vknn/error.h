// The Error exception, a Status-carrying std::runtime_error.
#pragma once
#include <stdexcept>
#include <string>

#include "vknn/status.h"

namespace vknn {

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

} // namespace vknn
