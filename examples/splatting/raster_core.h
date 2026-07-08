// The from-scratch Vulkan compute 3D-Gaussian-splatting rasterizer (preprocess -> GPU tile-bin ->
// stable radix sort -> per-tile alpha compositing), shared by the vknn_yonosplat example and the
// app-demo JNI bridge. Gaussians upload once. The first render after setGaussians runs two
// submissions: a count-only pass (CAP = 0) whose atomic total is read back on the host across a
// fence sizes the sort buffers exactly, then the bin + sort + composite submission runs. Later
// renders elide the count pass while the persisted grow-only sort buffers keep 1.25x headroom
// over the last emitted total, collapsing the steady state to a single submission; a headroom
// breach (emitted > capacity) grows the buffers and re-renders once. The sort is a 4-pass LSD
// radix sort over the 32-bit keys (8-bit digits: per-chunk histogram -> global exclusive scan ->
// stable ordered scatter into ping-pong key/value buffers), so it needs no power-of-two padding
// and equal keys keep their emit order.
#pragma once
#if defined(VKNN_ENABLE_VULKAN)
#include "backend/vulkan/vk_buffer.h"
#include "backend/vulkan/vk_command.h"
#include "backend/vulkan/vk_pipeline.h"
#include <cstdint>
#include <memory>

namespace raster {

    // Spherical-harmonics band-0 (DC) basis factor 1/(2*sqrt(pi)); converts a degree-0 harmonic
    // coefficient to a base RGB color via color = C0*sh + 0.5.
    constexpr float kC0 = 0.28209479177387814f;

    enum class Result {
        Ok,
        SortLimitExceeded, // tile entries exceed the 2^30 sort limit
        Error,             // no Vulkan device / no Gaussians uploaded
    };

    struct Stats {
        uint32_t entries = 0; // tile-entry total (count pass when it runs, else the emit counter)
        int64_t  cap     = 0; // persisted sort-buffer capacity in entries (>= entries, floor 2^16)
        uint32_t emitted = 0; // entries the emit pass produced
        double   msCount = 0; // count-only submission GPU ms (0 when the count pass is elided)
        double   msMain  = 0; // main submission(s) GPU ms (preprocess + bin + sort + composite)
    };

    class Rasterizer {
      public:
        Rasterizer(int height, int width, float nearPlane = 0.2f);
        ~Rasterizer();
        Rasterizer(const Rasterizer &)            = delete;
        Rasterizer &operator=(const Rasterizer &) = delete;

        /// False when no Vulkan device is available.
        bool ok() const noexcept {
            return ok_;
        }

        /// Upload n Gaussians: means [n*3], covariances [n*9], colors [n*3] (linear RGB, see kC0),
        /// opacities [n]. Replaces any previously uploaded set.
        void setGaussians(const float *means, const float *covariances, const float *colors, const float *opacities, int n);

        /// Render from a row-major camera-to-world [16] (w2c = its rigid inverse, computed here)
        /// with pixel-unit intrinsics into out [height*width*3 fp32].
        Result render(const float cameraToWorld[16], float focalX, float focalY, float centerX, float centerY, float *out, Stats *stats = nullptr);

        int gaussians() const noexcept {
            return gaussianCount_;
        }
        int height() const noexcept {
            return height_;
        }
        int width() const noexcept {
            return width_;
        }

      private:
        /// Grow the persisted sort buffers (ping-pong keys/values + radix histogram) to hold
        /// `capacity` entries; no-op when they are already large enough.
        void ensureSortCapacity(int64_t capacity);

        int   height_, width_;
        float nearPlane_;
        bool  ok_            = false;
        int   gaussianCount_ = 0;

        // 16x16-pixel screen tiles. Each sort key packs the tile index in its high bits and a
        // depth-derived value in the low depthBits_ bits, so a single ascending sort groups splats
        // by tile and orders them front-to-back within each tile.
        int   tilesX_, tilesY_, tileCount_, depthBits_;
        float invDepthRange_; // normalizes depth in [near, 256) to [0, 1) before key quantization

        vknn::vk::VulkanContext                  context_;
        std::unique_ptr<vknn::vk::CommandRunner> runner_;
        std::unique_ptr<vknn::vk::ComputePipeline> preprocessPipe_, fillPipe_, duplicatePipe_, radixCountPipe_, radixScanPipe_, radixScatterPipe_, rangesPipe_, compositePipe_;
        // Per setGaussians (sized by the Gaussian count):
        std::unique_ptr<vknn::vk::Buffer> meansBuffer_, covariancesBuffer_, colorsBuffer_, opacitiesBuffer_, geometryBuffer_;
        // Per (height, width):
        std::unique_ptr<vknn::vk::Buffer> counterBuffer_, standInBuffer_, tileRangesBuffer_, imageBuffer_;
        // Sort working set, grow-only across renders (see ensureSortCapacity): the ping-pong
        // (key, value) buffer pairs and the radix digit histogram.
        std::unique_ptr<vknn::vk::Buffer> sortKeysBuffer_, sortValuesBuffer_, sortKeysPingBuffer_, sortValuesPingBuffer_, radixHistogramBuffer_;
        int64_t                           sortCapacity_      = 0;     // entries the sort buffers hold
        uint32_t                          lastEmittedCount_  = 0;     // previous render's emit total
        bool                              emittedCountKnown_ = false; // false forces the exact count pass (fresh Gaussians)
    };

} // namespace raster
#endif
