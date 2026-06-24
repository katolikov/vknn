// vxrt — structured, leveled, throttled, color-coded logging.
#pragma once
#include <string>
#include <sstream>
#include <cstdint>

namespace vx {

enum class LogLevel { kDebug = 0, kInfo = 1, kWarn = 2, kError = 3, kNone = 4 };

/// Global logger. Thread-safe. Honors VXRT_LOG_LEVEL env (DEBUG/INFO/WARN/ERROR).
class Log {
 public:
  static void setLevel(LogLevel l);
  static LogLevel level();
  static void setColor(bool on);

  /// Emit a line at `lvl`. `key` (optional) enables throttling: repeated logs with
  /// the same key beyond `throttleAfter` occurrences are suppressed with a summary.
  static void emit(LogLevel lvl, const std::string& msg, const std::string& key = "",
                   int throttleAfter = 0);
};

namespace detail {
struct LogStream {
  LogLevel lvl;
  std::string key;
  int throttleAfter;
  std::ostringstream ss;
  LogStream(LogLevel l, std::string k = "", int t = 0)
      : lvl(l), key(std::move(k)), throttleAfter(t) {}
  ~LogStream() { Log::emit(lvl, ss.str(), key, throttleAfter); }
  template <typename T>
  LogStream& operator<<(const T& v) {
    ss << v;
    return *this;
  }
};
}  // namespace detail

}  // namespace vx

#define VX_LOG(LVL) ::vx::detail::LogStream(::vx::LogLevel::LVL)
#define VX_DEBUG VX_LOG(kDebug)
#define VX_INFO VX_LOG(kInfo)
#define VX_WARN VX_LOG(kWarn)
#define VX_ERROR VX_LOG(kError)
// Throttled warning: WARN that collapses after N repeats keyed by KEY.
#define VX_WARN_THROTTLE(KEY, N) ::vx::detail::LogStream(::vx::LogLevel::kWarn, KEY, N)
