// Host tests for the low-precision GEMM helpers (core/lowp_gemm.h): the e4m3 codec, the int8
// symmetric codec, and the deterministic cooperative-matrix routing rule. The GPU kernels that
// consume these run only on cooperative-matrix hardware; the codecs and the rule are pure host
// logic and gate here.
#include "core/lowp_gemm.h"
#include <cmath>
#include <gtest/gtest.h>

namespace vknn {

    TEST(LowpGemm, Fp8E4M3KnownEncodings) {
        EXPECT_EQ(encodeFp8E4M3(0.f), 0x00);
        EXPECT_EQ(encodeFp8E4M3(1.0f), 0x38); // 2^0 * 1.000
        EXPECT_EQ(encodeFp8E4M3(1.5f), 0x3C); // 2^0 * 1.100
        EXPECT_EQ(encodeFp8E4M3(-1.5f), 0xBC);
        EXPECT_EQ(encodeFp8E4M3(448.f), 0x7E);  // max finite 2^8 * 1.75
        EXPECT_EQ(encodeFp8E4M3(1000.f), 0x7E); // saturates (no infinity)
        EXPECT_EQ(encodeFp8E4M3(-1000.f), 0xFE);
        EXPECT_EQ(encodeFp8E4M3(std::ldexp(1.f, -9)), 0x01); // smallest subnormal 2^-6 / 8
        EXPECT_EQ(encodeFp8E4M3(std::nanf("")), 0x7F);
    }

    TEST(LowpGemm, Fp8E4M3TiesToEven) {
        // 1.0625 sits halfway between 1.0 (0x38, even mantissa) and 1.125 (0x39): rounds down.
        EXPECT_EQ(encodeFp8E4M3(1.0625f), 0x38);
        // 1.1875 sits halfway between 1.125 (0x39) and 1.25 (0x3A, even mantissa): rounds up.
        EXPECT_EQ(encodeFp8E4M3(1.1875f), 0x3A);
    }

    TEST(LowpGemm, Fp8E4M3RoundTripAllCodes) {
        for (int code = 0; code < 256; ++code)
        {
            if (code == 0x7F || code == 0xFF || code == 0x80)
            {
                continue; // NaN encodings; negative zero canonicalizes to positive zero
            }
            const float decoded = decodeFp8E4M3((uint8_t) code);
            EXPECT_EQ(encodeFp8E4M3(decoded), code) << "code " << code << " decoded " << decoded;
        }
    }

    TEST(LowpGemm, Int8SymmetricCodec) {
        const float absmax = 3.7f;
        const float scale  = absmax / 127.f;
        EXPECT_EQ(encodeInt8Symmetric(absmax, scale), 127);
        EXPECT_EQ(encodeInt8Symmetric(-absmax, scale), -127);
        EXPECT_EQ(encodeInt8Symmetric(0.f, scale), 0);
        EXPECT_EQ(encodeInt8Symmetric(10.f * absmax, scale), 127); // clamps
        EXPECT_EQ(encodeInt8Symmetric(-10.f * absmax, scale), -127);
        EXPECT_EQ(encodeInt8Symmetric(1.f, 0.f), 0); // zero tensor: zero scale, zero codes
    }

    namespace {
        CoopmatGemmCaps fullCaps() {
            CoopmatGemmCaps caps;
            caps.coopmatFp16Fp32Row16 = true;
            caps.coopmatFp8Fp32Row16  = true;
            caps.coopmatI8I32Row16    = true;
            caps.wave32Pinnable       = true;
            caps.vulkanMemoryModel    = true;
            caps.selfCheckPassed      = true;
            return caps;
        }
    } // namespace

    TEST(LowpGemm, CoopmatRouteEligibility) {
        const CoopmatGemmCaps caps = fullCaps();
        // Auto on an eligible shape takes the fp16/fp32-accumulator kind.
        EXPECT_EQ(coopmatGemmRoute(caps, 0, true, true, false, 64, 64, 64, true), CoopmatGemmKind::Fp16);
        // Off always keeps the SSBO kernels.
        EXPECT_EQ(coopmatGemmRoute(caps, 2, true, true, false, 64, 64, 64, true), CoopmatGemmKind::None);
        // Shape misfits: M not a multiple of 32, K not a multiple of 16, too small.
        EXPECT_EQ(coopmatGemmRoute(caps, 0, true, true, false, 48, 64, 64, true), CoopmatGemmKind::None);
        EXPECT_EQ(coopmatGemmRoute(caps, 0, true, true, false, 64, 64, 56, true), CoopmatGemmKind::None);
        EXPECT_EQ(coopmatGemmRoute(caps, 0, true, true, false, 32, 32, 16, true), CoopmatGemmKind::None);
        // Structural misfits: fp32 storage, view/1-D/batch, bias or epilogue.
        EXPECT_EQ(coopmatGemmRoute(caps, 0, false, true, false, 64, 64, 64, true), CoopmatGemmKind::None);
        EXPECT_EQ(coopmatGemmRoute(caps, 0, true, false, false, 64, 64, 64, true), CoopmatGemmKind::None);
        EXPECT_EQ(coopmatGemmRoute(caps, 0, true, true, true, 64, 64, 64, true), CoopmatGemmKind::None);
    }

    TEST(LowpGemm, CoopmatRouteCapabilityGates) {
        CoopmatGemmCaps caps = fullCaps();
        caps.selfCheckPassed = false;
        EXPECT_EQ(coopmatGemmRoute(caps, 0, true, true, false, 64, 64, 64, true), CoopmatGemmKind::None);
        caps                = fullCaps();
        caps.wave32Pinnable = false;
        EXPECT_EQ(coopmatGemmRoute(caps, 0, true, true, false, 64, 64, 64, true), CoopmatGemmKind::None);
        caps                   = fullCaps();
        caps.vulkanMemoryModel = false;
        EXPECT_EQ(coopmatGemmRoute(caps, 0, true, true, false, 64, 64, 64, true), CoopmatGemmKind::None);
        caps                      = fullCaps();
        caps.coopmatFp16Fp32Row16 = false;
        EXPECT_EQ(coopmatGemmRoute(caps, 0, true, true, false, 64, 64, 64, true), CoopmatGemmKind::None);
    }

    TEST(LowpGemm, CoopmatRouteOptInKinds) {
        const CoopmatGemmCaps caps = fullCaps();
        // The opt-in values select their kind only with quantizable weights and the matching row.
        EXPECT_EQ(coopmatGemmRoute(caps, 3, true, true, false, 64, 64, 64, true), CoopmatGemmKind::Fp8);
        EXPECT_EQ(coopmatGemmRoute(caps, 4, true, true, false, 64, 64, 64, true), CoopmatGemmKind::Int8);
        // Activation weights (no initializer) or a missing row degrade to the fp16 kind.
        EXPECT_EQ(coopmatGemmRoute(caps, 3, true, true, false, 64, 64, 64, false), CoopmatGemmKind::Fp16);
        CoopmatGemmCaps noFp8     = fullCaps();
        noFp8.coopmatFp8Fp32Row16 = false;
        EXPECT_EQ(coopmatGemmRoute(noFp8, 3, true, true, false, 64, 64, 64, true), CoopmatGemmKind::Fp16);
        // Auto never selects a low-precision kind.
        EXPECT_EQ(coopmatGemmRoute(caps, 0, true, true, false, 64, 64, 64, true), CoopmatGemmKind::Fp16);
    }

} // namespace vknn
