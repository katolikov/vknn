// vknn_compile's version request (convert/compile_cli.h).
//
// `vknn_compile --version` is answered before any other parsing, so it works with no model and no
// output positional -- the two arguments the tool otherwise requires. The predicate below is the
// one main() calls, so these tests pin the spellings, the positions they are honoured at, and that
// nothing else is mistaken for a version request.
#include "../convert/compile_cli.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace vknn;

namespace {
    // Run the predicate over a literal argument list, argv[0] included (as main() receives it).
    bool requested(const std::vector<std::string> &args) {
        std::vector<char *>      argv;
        std::vector<std::string> owned = args;
        for (std::string &a: owned)
        {
            argv.push_back(&a[0]);
        }
        return compileVersionRequested((int) argv.size(), argv.data());
    }
} // namespace

// The whole point of the pre-parse: no model, no output, just the flag.
TEST(CompileCli, VersionFlagAloneIsAVersionRequest) {
    EXPECT_TRUE(requested({"vknn_compile", kCompileVersionFlagLong}));
    EXPECT_TRUE(requested({"vknn_compile", kCompileVersionFlagShort}));
}

// The scan covers every position, including the slots the positionals occupy, so the request is
// honoured whatever else is on the line.
TEST(CompileCli, VersionFlagIsHonouredAtAnyPosition) {
    EXPECT_TRUE(requested({"vknn_compile", "model.onnx", "out.vxm", kCompileVersionFlagLong}));
    EXPECT_TRUE(requested({"vknn_compile", "model.onnx", kCompileVersionFlagShort, "out.vxm"}));
    EXPECT_TRUE(requested({"vknn_compile", kCompileVersionFlagLong, "--fp16"}));
}

// Nothing else is a version request: not the other flags, not a near spelling, and not argv[0].
TEST(CompileCli, OtherArgumentsAreNotVersionRequests) {
    EXPECT_FALSE(requested({"vknn_compile"}));
    EXPECT_FALSE(requested({"vknn_compile", "model.onnx", "out.vxm", "--fp16", "-Os"}));
    EXPECT_FALSE(requested({"vknn_compile", "--versions"}));
    EXPECT_FALSE(requested({"vknn_compile", "-v"}));
    EXPECT_FALSE(requested({"vknn_compile", "version.onnx", "out.vxm"}));
    EXPECT_FALSE(requested({kCompileVersionFlagLong, "model.onnx", "out.vxm"})) << "argv[0] is the program name, not a flag";
}
