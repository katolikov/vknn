#include "vknn/logging.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <unordered_map>

#if defined(VKNN_ANDROID)
#include <android/log.h>
#endif

namespace vknn {
namespace {
std::mutex g_mu;
LogLevel g_level = LogLevel::kInfo;
bool g_color = true;
std::unordered_map<std::string, int> g_counts;
bool g_init = false;

void ensureInit() {
  if (g_init)
    return;
  g_init = true;
  if (const char* e = std::getenv("VKNN_LOG_LEVEL")) {
    if (!strcasecmp(e, "DEBUG"))
      g_level = LogLevel::kDebug;
    else if (!strcasecmp(e, "INFO"))
      g_level = LogLevel::kInfo;
    else if (!strcasecmp(e, "WARN"))
      g_level = LogLevel::kWarn;
    else if (!strcasecmp(e, "ERROR"))
      g_level = LogLevel::kError;
    else if (!strcasecmp(e, "NONE"))
      g_level = LogLevel::kNone;
  }
}

const char* levelTag(LogLevel l) {
  switch (l) {
    case LogLevel::kDebug:
      return "DEBUG";
    case LogLevel::kInfo:
      return "INFO ";
    case LogLevel::kWarn:
      return "WARN ";
    case LogLevel::kError:
      return "ERROR";
    default:
      return "?????";
  }
}
const char* levelColor(LogLevel l) {
  switch (l) {
    case LogLevel::kDebug:
      return "\033[2;37m";  // dim grey
    case LogLevel::kInfo:
      return "\033[36m";  // cyan
    case LogLevel::kWarn:
      return "\033[33m";  // yellow
    case LogLevel::kError:
      return "\033[1;31m";  // bold red
    default:
      return "";
  }
}
}  // namespace

void Log::setLevel(LogLevel l) {
  std::lock_guard<std::mutex> g(g_mu);
  g_init = true;
  g_level = l;
}
LogLevel Log::level() {
  std::lock_guard<std::mutex> g(g_mu);
  ensureInit();
  return g_level;
}
void Log::setColor(bool on) {
  std::lock_guard<std::mutex> g(g_mu);
  g_color = on;
}

void Log::emit(LogLevel lvl, const std::string& msg, const std::string& key, int throttleAfter) {
  std::lock_guard<std::mutex> g(g_mu);
  ensureInit();
  if (lvl < g_level)
    return;
  if (throttleAfter > 0 && !key.empty()) {
    int& c = g_counts[key];
    ++c;
    if (c == throttleAfter + 1) {
      // emit one final "suppressing" note then go silent
    } else if (c > throttleAfter + 1) {
      return;
    }
  }
  std::string suffix;
  if (throttleAfter > 0 && !key.empty() && g_counts[key] == throttleAfter + 1)
    suffix = " (further '" + key + "' messages suppressed)";

#if defined(VKNN_ANDROID)
  int prio = ANDROID_LOG_INFO;
  switch (lvl) {
    case LogLevel::kDebug:
      prio = ANDROID_LOG_DEBUG;
      break;
    case LogLevel::kInfo:
      prio = ANDROID_LOG_INFO;
      break;
    case LogLevel::kWarn:
      prio = ANDROID_LOG_WARN;
      break;
    case LogLevel::kError:
      prio = ANDROID_LOG_ERROR;
      break;
    default:
      break;
  }
  __android_log_print(prio, "vknn", "%s%s", msg.c_str(), suffix.c_str());
#endif
  // Always also print to stderr (so adb shell run captures it without logcat).
  FILE* out = stderr;
  if (g_color) {
    fprintf(out, "%s[vknn %s]\033[0m %s%s\n", levelColor(lvl), levelTag(lvl), msg.c_str(),
            suffix.c_str());
  } else {
    fprintf(out, "[vknn %s] %s%s\n", levelTag(lvl), msg.c_str(), suffix.c_str());
  }
  fflush(out);
}

}  // namespace vknn
