// Shared routing rule for the vec4-weight twin of the implicit-GEMM convolution kernel
// (shaders/conv_gemm_wv4.comp vs shaders/conv_gemm.comp), consumed by the ConvGemm op, by Conv's
// deterministic gemm path, and by the host tests. One definition keeps the two dispatch sites and
// the tests in lock-step, the same way core/matmul_tile.h owns matmulVec4Route for the GEMM twins.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace vknn {

    /// Element alignment of the vec4-weight twin. Its W-panel load fetches a STORE4 (four contiguous
    /// output channels as one 64-bit fp16 / 128-bit fp32 load) from the row-major [K][Coutp] weight
    /// panel, so every global element index it forms must be a multiple of four: the panel's column
    /// offset is already a multiple of four (the workgroup's N tile is a multiple of the shader's
    /// TN = 64 and each thread owns four adjacent channels), which leaves the PHYSICAL row stride
    /// Coutp as the only condition.
    constexpr int64_t kConvGemmWVec4Align = 4;

    /// cout rounded up to the next kConvGemmWVec4Align multiple: the physical weight row stride a
    /// zero-padded [K][Coutp] repack presents to the vec4-weight kernel.
    constexpr int64_t roundUpConvGemmCout(int64_t cout) {
        return (cout + kConvGemmWVec4Align - 1) / kConvGemmWVec4Align * kConvGemmWVec4Align;
    }

    /// convGemmWVec4Route's verdict. coutP is the physical weight row stride the kernel consumes
    /// (ConvGemmPC::Coutp): Cout for a naturally aligned panel, roundUpConvGemmCout(Cout) for a
    /// zero-padded repack.
    struct ConvGemmWVec4Route {
        bool    eligible = false; ///< dispatch conv_gemm_wv4 instead of conv_gemm
        bool    padW     = false; ///< the [K][Cout] panel repacks to a zero-padded [K][coutP] copy
        int64_t coutP    = 0;     ///< physical weight row stride (== Cout unless padW)
    };

    /// Deterministic routing rule for the vec4-weight twin — a pure shape/layout function, never a
    /// timing race. The twin keeps the scalar kernel's LDS panel layout, its chunked ascending-k
    /// fp32 accumulation and its store, and changes only the width of the weight-panel global load,
    /// so it is byte-identical to conv_gemm and needs no accuracy gate:
    ///   - Cout % kConvGemmWVec4Align == 0: the packed [K][Cout] panel is already 4-aligned on
    ///     every row, so the shipped weight buffer routes as-is.
    ///   - otherwise the panel routes only if its rows can repack to a zero-padded
    ///     coutP = roundUpConvGemmCout(Cout). The pad lanes sit past column Cout and read zero —
    ///     exactly the value the scalar kernel's `gc < Cout` guard yields — so the LDS panel matches
    ///     element for element.
    /// `weightRepackable` reports whether that padded copy can be built here: Conv's gemm path
    /// always builds its [K][Cout] panel on the host and can pad it in place, while the ConvGemm op
    /// needs the lowered initializer's payload to still be readable.
    inline ConvGemmWVec4Route convGemmWVec4Route(int64_t cout, bool weightRepackable) {
        ConvGemmWVec4Route route;
        route.coutP = cout;
        if (cout <= 0)
        {
            return route;
        }
        if (cout % kConvGemmWVec4Align == 0)
        {
            route.eligible = true;
            return route;
        }
        if (!weightRepackable)
        {
            return route;
        }
        route.eligible = true;
        route.padW     = true;
        route.coutP    = roundUpConvGemmCout(cout);
        return route;
    }

    /// Zero-padded row repack for the vec4-weight route: `k` rows of `cout` source channels each
    /// copy into rows of `coutP` channels (coutP >= cout, a kConvGemmWVec4Align multiple from
    /// roundUpConvGemmCout); the tail coutP - cout channels of every row are zero. The pad zeros are
    /// never accumulated (the kernel's column guard drops them) and mirror the value the scalar
    /// kernel's guard yields, so the repacked panel is output-byte-neutral. `src` holds at least
    /// k * cout elements.
    inline std::vector<float> padConvGemmWeightVec4(const std::vector<float> &src, int64_t k, int64_t cout, int64_t coutP) {
        std::vector<float> padded((size_t) (k * coutP), 0.0f);
        for (int64_t row = 0; row < k; ++row)
        {
            std::copy_n(src.data() + row * cout, (size_t) cout, padded.data() + row * coutP);
        }
        return padded;
    }

} // namespace vknn
