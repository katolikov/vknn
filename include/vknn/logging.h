// Convenience front end for the leveled logger: stream-style macros that build one line and emit it
// at a fixed severity when their LogStream temporary is destroyed. The active threshold is set
// programmatically via Log::setLevel() (driven by Config::verbosity), colorized by level, and a
// repeated line can be throttled so spammy warnings (e.g. per-op fallbacks) collapse to one.
#pragma once

#include "vknn/log.h"
#include "vknn/log_level.h"
#include "vknn/log_stream.h"

/// Open a log line at LogLevel::LVL (an unqualified enumerator, e.g. Info). Yields a temporary that
/// accepts `<<` and emits the accumulated line when it goes out of scope at the end of the statement.
#define VKNN_LOG(LVL) ::vknn::detail::LogStream(::vknn::LogLevel::LVL)
/// Stream a line at LogLevel::Debug (verbose developer tracing).
#define VKNN_DEBUG VKNN_LOG(Debug)
/// Stream a line at LogLevel::Info (normal progress and status).
#define VKNN_INFO VKNN_LOG(Info)
/// Stream a line at LogLevel::Warn (recoverable anomaly).
#define VKNN_WARN VKNN_LOG(Warn)
/// Stream a line at LogLevel::Error (operation-aborting failure).
#define VKNN_ERROR VKNN_LOG(Error)
/// Stream a throttled line at LogLevel::Warn: identical repeats sharing KEY collapse after N prints,
/// so a per-op fallback warning does not flood the log. KEY is the throttle bucket; N is the count
/// after which further messages are suppressed (see Log::emit's throttleAfter).
#define VKNN_WARN_THROTTLE(KEY, N) ::vknn::detail::LogStream(::vknn::LogLevel::Warn, KEY, N)
