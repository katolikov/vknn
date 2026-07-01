// Small leveled logger. Reads VKNN_LOG_LEVEL, colorizes by level, and can throttle a repeated
// line so spammy warnings (e.g. per-op fallbacks) collapse to one.
#pragma once

#include "vknn/log_level.h"
#include "vknn/log.h"
#include "vknn/log_stream.h"

#define VKNN_LOG(LVL) ::vknn::detail::LogStream(::vknn::LogLevel::LVL)
#define VKNN_DEBUG    VKNN_LOG(Debug)
#define VKNN_INFO     VKNN_LOG(Info)
#define VKNN_WARN     VKNN_LOG(Warn)
#define VKNN_ERROR    VKNN_LOG(Error)
// Throttled warning: WARN that collapses after N repeats keyed by KEY.
#define VKNN_WARN_THROTTLE(KEY, N) ::vknn::detail::LogStream(::vknn::LogLevel::Warn, KEY, N)
