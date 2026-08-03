/// Argument-vector predicates for the vknn_compile CLI, in a header so the unit tests exercise the
/// same code the tool runs.
#pragma once
#include <cstring>

namespace vknn {

    /// Long and short spelling of the version request, answered before any other parsing.
    inline constexpr const char *kCompileVersionFlagLong  = "--version";
    inline constexpr const char *kCompileVersionFlagShort = "-V";

    /// True when the argument vector asks for the engine version.
    ///
    /// The scan starts at argv[1] and covers EVERY position, including the slots the model/output
    /// positionals occupy: the request is answered before the positional-count check, so
    /// `vknn_compile --version` works with no model and no output argument.
    ///
    /// @param argc Argument count as main() receives it.
    /// @param argv Argument vector as main() receives it; argv[0] (the program name) is skipped, so
    ///             a tool installed under a path spelled "--version" is not a version request.
    inline bool compileVersionRequested(int argc, char **argv) noexcept {
        for (int i = 1; i < argc; ++i)
        {
            if (!strcmp(argv[i], kCompileVersionFlagLong) || !strcmp(argv[i], kCompileVersionFlagShort))
            {
                return true;
            }
        }
        return false;
    }

} // namespace vknn
