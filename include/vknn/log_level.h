#pragma once

namespace vknn {

    /// Severity threshold for diagnostic output. A message is emitted only when its own severity is at
    /// least the configured level, so higher values are progressively quieter; None disables logging
    /// entirely. The enumerators are ordered by increasing severity and their integer values are stable,
    /// so they may be compared with `<`/`>=` to gate a message against the active threshold.
    enum class LogLevel {
        Debug = 0, ///< Verbose developer tracing; the noisiest level.
        Info  = 1, ///< Normal progress and status messages.
        Warn  = 2, ///< Recoverable anomalies that do not stop execution.
        Error = 3, ///< Failures that abort the current operation.
        None  = 4, ///< Suppresses all output; nothing is logged at this threshold.
    };

} // namespace vknn
