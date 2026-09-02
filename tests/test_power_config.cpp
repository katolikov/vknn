// Config::power is policy plumbing: the tokens, the default, and the JSON round-trip are all the host
// can prove (the keep-alive thread itself needs a Vulkan device, so it is exercised on the target).
#include "vknn/config.h"
#include <gtest/gtest.h>
#include <string>

using namespace vknn;

TEST(PowerConfig, FromStrTokens) {
    EXPECT_EQ(powerFromStr("normal"), Power::Normal);
    EXPECT_EQ(powerFromStr("high"), Power::High);
    // Anything else — the empty string, a case variant, or a priority tier — is the default.
    EXPECT_EQ(powerFromStr(""), Power::Normal);
    EXPECT_EQ(powerFromStr("HIGH"), Power::Normal);
    EXPECT_EQ(powerFromStr("High"), Power::Normal);
    EXPECT_EQ(powerFromStr("low"), Power::Normal);
    EXPECT_EQ(powerFromStr("max"), Power::Normal);
}

TEST(PowerConfig, EnumValuesStable) {
    EXPECT_EQ((int) Power::Normal, 0);
    EXPECT_EQ((int) Power::High, 1);
}

TEST(PowerConfig, DefaultIsNormal) {
    Config c;
    EXPECT_EQ(c.power, Power::Normal);
    // A default config serializes the default token and reads back unchanged.
    EXPECT_NE(c.toJson().find("\"power\": \"normal\""), std::string::npos);
    EXPECT_EQ(Config::fromJsonString(c.toJson()).power, Power::Normal);
}

TEST(PowerConfig, JsonRoundTrip) {
    Config c;
    c.power        = Power::High;
    std::string js = c.toJson();
    EXPECT_NE(js.find("\"power\": \"high\""), std::string::npos);
    Config d = Config::fromJsonString(js);
    EXPECT_EQ(d.power, Power::High);
    // The power key rides alongside priority without disturbing it.
    EXPECT_EQ(d.priority, Priority::Normal);
}

TEST(PowerConfig, ParseExplicit) {
    EXPECT_EQ(Config::fromJsonString(R"({"power":"high"})").power, Power::High);
    EXPECT_EQ(Config::fromJsonString(R"({"power":"normal"})").power, Power::Normal);
    // An unrecognized token and an absent key both leave the default.
    EXPECT_EQ(Config::fromJsonString(R"({"power":"bogus"})").power, Power::Normal);
    EXPECT_EQ(Config::fromJsonString(R"({"priority":"high"})").power, Power::Normal);
    // power and priority are independent knobs: high priority does not imply high power, and vice versa.
    Config both = Config::fromJsonString(R"({"priority":"low","power":"high"})");
    EXPECT_EQ(both.priority, Priority::Low);
    EXPECT_EQ(both.power, Power::High);
}
