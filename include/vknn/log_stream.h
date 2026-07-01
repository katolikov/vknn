#pragma once
#include <sstream>
#include <string>

#include "vknn/log.h"
#include "vknn/log_level.h"

namespace vknn {

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
