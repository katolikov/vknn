// Zero-match accounting for the user-supplied tensor-name pattern knobs: markFp32 reports which
// Config::fp32Tensors entries matched an eligible (non-initializer, flat) tensor name, and the
// session warns once per model load for every entry that matched nothing in any bucket, so a pin
// or dump whose tensor was renamed or fused away never silently no-ops. Marking behavior is
// unchanged — the accounting is read-only.
#include "import/passes.h"
#include "vknn/graph.h"
#include <gtest/gtest.h>

using namespace vknn;

namespace {

    TensorId addNamed(Graph &g, const std::string &name, bool flat, bool initializer = false) {
        TensorDesc d;
        d.name          = name;
        d.shape         = {1, 4};
        d.gpuFlat       = flat;
        d.isInitializer = initializer;
        TensorId t      = g.addTensor(d);
        if (initializer)
        {
            HostBuffer hb;
            hb.resizeElems(4, DType::Float32);
            g.initializers[t] = hb;
        }
        return t;
    }

} // namespace

TEST(PatternAccounting, SplitPatternListKeepsEntriesAsTyped) {
    EXPECT_EQ(splitPatternList("a,-b,,c"), (std::vector<std::string> {"a", "-b", "c"}));
    EXPECT_TRUE(splitPatternList("").empty());
    EXPECT_EQ(splitPatternList("solo"), (std::vector<std::string> {"solo"}));
}

TEST(PatternAccounting, MarkFp32RecordsMatchedEntriesOverEligibleTensorsOnly) {
    Graph    g;
    TensorId flatAct = addNamed(g, "attn_q_out", true);
    addNamed(g, "conv_out", false);            // NC4HW4: never eligible for a substring mark
    addNamed(g, "attn_table", true, true);     // initializer: never eligible
    std::set<std::string> matched;
    markFp32(g, "attn_q,-conv_out,ghost", &matched);
    EXPECT_TRUE(matched.count("attn_q")) << "the include entry matches the eligible flat tensor";
    EXPECT_FALSE(matched.count("-conv_out")) << "an NC4HW4 tensor is not eligible, so its entry stays unmatched";
    EXPECT_FALSE(matched.count("ghost")) << "an entry matching nothing must stay unmatched";
    EXPECT_TRUE(g.desc(flatAct).storeFp32) << "the marking itself is unchanged by the accounting";
}

TEST(PatternAccounting, MarkFp32RecordsExcludeEntriesAsTyped) {
    Graph g;
    addNamed(g, "geo_tail", true);
    std::set<std::string> matched;
    markFp32(g, "-geo,head", &matched);
    EXPECT_TRUE(matched.count("-geo")) << "an exclude entry accounts against the names it suppresses";
    EXPECT_FALSE(matched.count("head"));
    // The exclude wins: nothing is marked, yet the entry is a live (matched) knob.
    for (const auto &t: g.tensors)
    {
        EXPECT_FALSE(t.storeFp32);
    }
}

TEST(PatternAccounting, MarkFp32NullAccountingIsUnchanged) {
    Graph    g;
    TensorId flatAct = addNamed(g, "attn_q_out", true);
    markFp32(g, "attn_q");
    EXPECT_TRUE(g.desc(flatAct).storeFp32);
}
