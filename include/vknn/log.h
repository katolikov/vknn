#pragma once
#include <string>

#include "vknn/log_level.h"

namespace vknn {

    /// Process-wide leveled logger. Every call routes through a single mutex, so emit() and the
    /// level/color setters are safe to call concurrently from any thread. The threshold level is set
    /// programmatically (setLevel(), driven by Config::verbosity) rather than from the environment;
    /// messages below the current level() are dropped.
    class Log {
      public:
        /// Set the minimum level that emit() will print. Messages below `l` are discarded.
        static void setLevel(LogLevel l);
        /// Current threshold level. Defaults to LogLevel::Info until setLevel() is called.
        static LogLevel level();
        /// Enable or disable ANSI color escapes on the stderr output.
        static void setColor(bool on);

        /// Emit one line at level `lvl` to stderr (and to logcat on Android). No-op when
        /// `lvl` is below the current level().
        /// @param lvl          Severity of this message.
        /// @param msg          Text to print; a level tag and (optional) throttle note are added.
        /// @param key          Throttle bucket. Empty disables throttling for this call.
        /// @param throttleAfter When > 0 and `key` is non-empty, the first `throttleAfter` messages
        ///                      sharing `key` print normally; the next one prints with a
        ///                      "further messages suppressed" note, and any beyond that are dropped.
        static void emit(LogLevel lvl, const std::string &msg, const std::string &key = "", int throttleAfter = 0);
    };

} // namespace vknn
