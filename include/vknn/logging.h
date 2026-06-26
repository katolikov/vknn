// Small leveled logger. Reads VKNN_LOG_LEVEL, colorizes by level, and can throttle a repeated
// line so spammy warnings (e.g. per-op fallbacks) collapse to one.
#pragma once
#include <cstdint>
#include <sstream>
#include <string>

namespace vknn {

    enum class LogLevel { kDebug = 0, kInfo = 1, kWarn = 2, kError = 3, kNone = 4 };

    /// Global logger. Thread-safe. Honors VKNN_LOG_LEVEL env (DEBUG/INFO/WARN/ERROR).
    class Log {
      public:
        static void     setLevel(LogLevel l);
        static LogLevel level();
        static void     setColor(bool on);

        /// Emit a line at `lvl`. `key` (optional) enables throttling: repeated logs with
        /// the same key beyond `throttleAfter` occurrences are suppressed with a summary.
        static void emit(LogLevel lvl, const std::string &msg, const std::string &key = "", int throttleAfter = 0);
    };

    namespace detail {
        struct LogStream {
            LogLevel           lvl;
            std::string        key;
            int                throttleAfter;
            std::ostringstream ss;
            LogStream(LogLevel l, std::string k = "", int t = 0): lvl(l), key(std::move(k)), throttleAfter(t) {
            }
            ~LogStream() {
                Log::emit(lvl, ss.str(), key, throttleAfter);
            }
            template <typename T> LogStream &operator<<(const T &v) {
                ss << v;
                return *this;
            }
        };
    } // namespace detail

} // namespace vknn

#define VKNN_LOG(LVL) ::vknn::detail::LogStream(::vknn::LogLevel::LVL)
#define VKNN_DEBUG    VKNN_LOG(kDebug)
#define VKNN_INFO     VKNN_LOG(kInfo)
#define VKNN_WARN     VKNN_LOG(kWarn)
#define VKNN_ERROR    VKNN_LOG(kError)
// Throttled warning: WARN that collapses after N repeats keyed by KEY.
#define VKNN_WARN_THROTTLE(KEY, N) ::vknn::detail::LogStream(::vknn::LogLevel::kWarn, KEY, N)
