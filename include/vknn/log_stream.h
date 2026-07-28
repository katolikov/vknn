#pragma once
#include "vknn/log.h"
#include "vknn/log_level.h"
#include <sstream>
#include <string>

namespace vknn { namespace detail {
    /// Ephemeral accumulator behind the streaming log macros (e.g. `VKNN_LOG(Info) << ...`).
    /// A temporary is constructed at the call site, receives the message piece by piece through
    /// operator<<, and flushes the whole line to Log::emit() exactly once when it is destroyed at
    /// the end of the full expression. Building the string in one buffer and emitting on scope exit
    /// keeps a single log call atomic rather than interleaving fragments across threads.
    struct LogStream {
        LogLevel    lvl;           ///< Severity carried through to Log::emit().
        std::string key;           ///< Throttle-bucket identifier; empty means the line is never throttled.
        int         throttleAfter; ///< When > 0 (and `key` set), the first this-many lines per `key` print; the rest are suppressed. 0 disables throttling.
        std::ostringstream ss;     ///< Message buffer filled by operator<< until destruction.
        /// @param l Severity of the line being built.
        /// @param k Throttle-bucket key; empty to disable throttling for this line.
        /// @param t Throttle threshold forwarded to Log::emit() (0 disables throttling).
        LogStream(LogLevel l, std::string k = "", int t = 0): lvl(l), key(std::move(k)), throttleAfter(t) {
        }
        /// Flush the accumulated message to Log::emit() as one line.
        ~LogStream() {
            Log::emit(lvl, ss.str(), key, throttleAfter);
        }
        /// Append `v` to the pending message using ostream formatting, then return *this so calls chain.
        template <typename T> LogStream &operator<<(const T &v) {
            ss << v;
            return *this;
        }
    };
}} // namespace vknn::detail
