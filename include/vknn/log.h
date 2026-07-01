#pragma once
#include <string>

#include "vknn/log_level.h"

namespace vknn {

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

} // namespace vknn
