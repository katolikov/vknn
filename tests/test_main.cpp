// Test entry point. Runs the standard GoogleTest suite, but mutes the engine's own diagnostic logging
// by default: many tests deliberately exercise rejection paths (bad input, corrupt files, out-of-range
// indices, unsupported ops) whose Warn/Error lines are expected, and the normal Info progress lines
// from every model load/run, would otherwise bury the gtest output. Pass --vknn-log to keep engine
// logging on when diagnosing a failure. gtest reports pass/fail through assertions, not these logs, so
// muting them loses no test signal.
#include "vknn/log.h"
#include "vknn/log_level.h"
#include <cstring>
#include <gtest/gtest.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv); // consumes gtest's own flags; leaves --vknn-log in argv
    bool keepLogs = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--vknn-log") == 0)
        {
            keepLogs = true;
        }
    }
    if (!keepLogs)
    {
        // pinLevel, not setLevel: sessions call Config::applyLogLevel() during load, which would
        // otherwise raise the threshold back to Info and un-mute the diagnostics.
        ::vknn::Log::pinLevel(::vknn::LogLevel::None);
    }
    return RUN_ALL_TESTS();
}
