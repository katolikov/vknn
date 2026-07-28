// The Error exception: a Status-carrying std::runtime_error thrown on any operation failure.
#pragma once
#include "vknn/status.h"
#include <stdexcept>
#include <string>

namespace vknn {

    /// Lightweight exception carrying a Status alongside a readable message. Thrown internally
    /// wherever an operation fails; the public facade catches it and reports the carried Status through
    /// its status-returning variants. The base runtime_error::what() is the message prefixed with the
    /// status name, e.g. "InvalidArgument: graph has a cycle".
    class Error: public std::runtime_error {
      public:
        /// @param s   Machine-readable status the caller can branch on via status().
        /// @param msg Readable detail; combined with the status name to form what().
        Error(Status s, const std::string &msg): std::runtime_error(std::string(statusStr(s)) + ": " + msg), status_(s) {
        }
        /// The status carried by this exception.
        Status status() const noexcept {
            return status_;
        }

      private:
        Status status_;
    };

} // namespace vknn
