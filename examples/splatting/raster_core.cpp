#include "raster_core.h"
#if defined(VKNN_ENABLE_VULKAN)
#include <cstring>
#include <vector>

namespace raster {

    using namespace vknn;

    namespace {

        // Push-constant layouts match the raster_* shader interfaces (field order is the wire format).
        struct PreprocessPush {
            int32_t dims[4];
            float   cam[4];
            float   cam2[4];
            float   r0[4], r1[4], r2[4];
        };
        struct DuplicatePush {
            int32_t gaussians, tilesX, tilesY, depthBits, capacity;
            float   nearPlane, invDepthRange;
        };
        struct RadixPush {
            uint32_t shift, numGroups, capacity;
        };
        struct RadixScanPush {
            uint32_t totalBins, writeTotal;
        };
        struct RangesPush {
            int32_t mode, tileCount, capacity, depthBits;
        };
        struct CompositePush {
            int32_t dims[4];
        };

        // Radix-sort geometry, mirrored in the raster_radix_{count,scan,scatter}.comp shaders:
        // 256 lanes per workgroup x 16 sequential batches = 4096 entries per chunk, 8-bit digits
        // over 32-bit keys = 4 passes, 256 histogram bins per chunk.
        constexpr uint32_t kRadixChunkEntries = 4096;
        constexpr uint32_t kRadixDigitBins    = 256;
        constexpr uint32_t kRadixPasses       = 4;

        // Hard sort limit: entry totals above this return Result::SortLimitExceeded.
        constexpr uint32_t kMaxSortEntries = 1u << 30;

        // Sort-buffer capacity policy: 1.25x headroom over the entry total so small view changes
        // stay within the persisted buffers (and keep the sizing submission elided), floor 2^16
        // to keep tiny scenes off the reallocation path, next power of two as the grow-only
        // allocation granularity, clamped to the sort limit.
        int64_t sortCapacityFor(uint32_t entries) {
            const int64_t withHeadroom = (int64_t) entries + (int64_t) entries / 4;
            int64_t       capacity     = 1 << 16;
            while (capacity < withHeadroom && capacity < (int64_t) kMaxSortEntries)
            {
                capacity <<= 1;
            }
            return capacity;
        }

    } // namespace

    Rasterizer::Rasterizer(int height, int width, float nearPlane): height_(height), width_(width), nearPlane_(nearPlane) {
        if (!context_.initialized())
        {
            return;
        }
        tilesX_      = (width_ + 15) / 16;
        tilesY_      = (height_ + 15) / 16;
        tileCount_   = tilesX_ * tilesY_;
        int tileBits = 1;
        while ((1 << tileBits) < tileCount_)
        {
            ++tileBits;
        }
        depthBits_     = 32 - tileBits; // key bits left for the depth field
        invDepthRange_ = 1.0f / (256.0f - nearPlane_);

        runner_           = std::make_unique<vk::CommandRunner>(context_);
        preprocessPipe_   = std::make_unique<vk::ComputePipeline>(context_, "raster_preprocess", 5, sizeof(PreprocessPush));
        duplicatePipe_    = std::make_unique<vk::ComputePipeline>(context_, "raster_duplicate", 4, sizeof(DuplicatePush));
        radixCountPipe_   = std::make_unique<vk::ComputePipeline>(context_, "raster_radix_count", 3, sizeof(RadixPush));
        radixScanPipe_    = std::make_unique<vk::ComputePipeline>(context_, "raster_radix_scan", 2, sizeof(RadixScanPush));
        radixScatterPipe_ = std::make_unique<vk::ComputePipeline>(context_, "raster_radix_scatter", 6, sizeof(RadixPush));
        rangesPipe_       = std::make_unique<vk::ComputePipeline>(context_, "raster_ranges", 3, sizeof(RangesPush));
        compositePipe_    = std::make_unique<vk::ComputePipeline>(context_, "raster_composite", 5, sizeof(CompositePush));

        // The binning scan writes the exact tile-entry total into counterBuffer_; it and the
        // image outputs are read back on the host, hence kReadback.
        counterBuffer_     = std::make_unique<vk::Buffer>(context_, 4, vk::MemPref::kReadback);
        tileRangesBuffer_  = std::make_unique<vk::Buffer>(context_, (size_t) tileCount_ * 2 * 4);
        imageBuffer_       = std::make_unique<vk::Buffer>(context_, (size_t) height_ * width_ * 3 * 4, vk::MemPref::kReadback);
        packedImageBuffer_ = std::make_unique<vk::Buffer>(context_, (size_t) height_ * width_ * 4, vk::MemPref::kReadback);
        // Stand-in for bindings a dispatch declares but never touches: the duplicate pass's
        // key/value slots in count mode and the scan's total slot when writeTotal is 0.
        standInBuffer_ = std::make_unique<vk::Buffer>(context_, 4);
        ok_            = true;
    }

    Rasterizer::~Rasterizer() = default;

    void Rasterizer::setGaussians(const float *means, const float *covariances, const float *colors, const float *opacities, int n) {
        if (!ok_)
        {
            return;
        }
        gaussianCount_         = n;
        emittedCountKnown_     = false; // a new Gaussian set re-runs the sizing submission once
        meansBuffer_           = std::make_unique<vk::Buffer>(context_, (size_t) n * 3 * 4);
        covariancesBuffer_     = std::make_unique<vk::Buffer>(context_, (size_t) n * 9 * 4);
        colorsBuffer_          = std::make_unique<vk::Buffer>(context_, (size_t) n * 3 * 4);
        opacitiesBuffer_       = std::make_unique<vk::Buffer>(context_, (size_t) n * 4);
        geometryBuffer_        = std::make_unique<vk::Buffer>(context_, (size_t) n * 12 * 4);
        gaussianOffsetsBuffer_ = std::make_unique<vk::Buffer>(context_, (size_t) n * 4);
        meansBuffer_->upload(means, (size_t) n * 3 * 4);
        covariancesBuffer_->upload(covariances, (size_t) n * 9 * 4);
        colorsBuffer_->upload(colors, (size_t) n * 3 * 4);
        opacitiesBuffer_->upload(opacities, (size_t) n * 4);
    }

    Result Rasterizer::render(const float cameraToWorld[16], float focalX, float focalY, float centerX, float centerY, float *out, Stats *stats) {
        return renderInternal(cameraToWorld, focalX, focalY, centerX, centerY, out, nullptr, stats);
    }

    Result Rasterizer::renderPacked(const float cameraToWorld[16], float focalX, float focalY, float centerX, float centerY, uint32_t *out, Stats *stats) {
        return renderInternal(cameraToWorld, focalX, focalY, centerX, centerY, nullptr, out, stats);
    }

    Result Rasterizer::renderInternal(const float cameraToWorld[16], float focalX, float focalY, float centerX, float centerY, float *fp32Out, uint32_t *packedOut, Stats *stats) {
        if (!ok_ || gaussianCount_ == 0)
        {
            return Result::Error;
        }
        const int gaussianCount = gaussianCount_;

        // Camera: w2c = rigid inverse of c2w.
        float rotationT[9], translation[3];
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                rotationT[row * 3 + col] = cameraToWorld[col * 4 + row];
            }
        }
        for (int row = 0; row < 3; ++row)
        {
            translation[row] = -(rotationT[row * 3] * cameraToWorld[3] + rotationT[row * 3 + 1] * cameraToWorld[7] + rotationT[row * 3 + 2] * cameraToWorld[11]);
        }
        PreprocessPush preprocessPush {};
        preprocessPush.dims[0] = gaussianCount;
        preprocessPush.dims[1] = height_;
        preprocessPush.dims[2] = width_;
        preprocessPush.cam[0]  = nearPlane_;
        preprocessPush.cam[1]  = focalX;
        preprocessPush.cam[2]  = focalY;
        preprocessPush.cam[3]  = centerX;
        preprocessPush.cam2[0] = centerY;
        for (int col = 0; col < 3; ++col)
        {
            preprocessPush.r0[col] = rotationT[col];
            preprocessPush.r1[col] = rotationT[3 + col];
            preprocessPush.r2[col] = rotationT[6 + col];
        }
        preprocessPush.r0[3] = translation[0];
        preprocessPush.r1[3] = translation[1];
        preprocessPush.r2[3] = translation[2];

        // Records the deterministic tile-binning prelude into `cmd`: preprocess the gaussians for
        // this pose, write each gaussian's tile-entry count, and exclusive-scan the counts into
        // per-gaussian emit offsets, leaving the exact entry total in counterBuffer_.
        auto recordBinningPrelude = [&](VkCommandBuffer cmd) {
            preprocessPipe_->dispatch(
                cmd, {meansBuffer_->handle(), covariancesBuffer_->handle(), colorsBuffer_->handle(), opacitiesBuffer_->handle(), geometryBuffer_->handle()}, &preprocessPush, sizeof(preprocessPush), (uint32_t) ((gaussianCount + 63) / 64));
            vk::computeBarrier(cmd);
            DuplicatePush countMode {gaussianCount, tilesX_, tilesY_, depthBits_, 0, nearPlane_, invDepthRange_};
            duplicatePipe_->dispatch(cmd, {geometryBuffer_->handle(), gaussianOffsetsBuffer_->handle(), standInBuffer_->handle(), standInBuffer_->handle()}, &countMode, sizeof(countMode), (uint32_t) ((gaussianCount + 63) / 64));
            vk::computeBarrier(cmd);
            RadixScanPush offsetsScan {(uint32_t) gaussianCount, 1u};
            radixScanPipe_->dispatch(cmd, {gaussianOffsetsBuffer_->handle(), counterBuffer_->handle()}, &offsetsScan, sizeof(offsetsScan), 1);
        };

        // Records and runs the main submission: (optionally) the binning prelude, then the
        // offset-directed emit, radix sort, per-tile ranges, and composite. Returns the
        // submission's wall time in ms. includePrelude is false only right after the sizing
        // submission, whose fence already left this pose's preprocess results, emit offsets, and
        // entry total in place.
        auto runMainSubmission = [&](bool includePrelude) -> double {
            const uint32_t  radixGroups = (uint32_t) ((sortCapacity_ + kRadixChunkEntries - 1) / kRadixChunkEntries);
            VkCommandBuffer cmd         = runner_->allocate();
            runner_->begin(cmd);
            if (includePrelude)
            {
                recordBinningPrelude(cmd);
                vk::computeBarrier(cmd);
            }
            DuplicatePush emit {gaussianCount, tilesX_, tilesY_, depthBits_, (int) sortCapacity_, nearPlane_, invDepthRange_};
            duplicatePipe_->dispatch(cmd, {geometryBuffer_->handle(), gaussianOffsetsBuffer_->handle(), sortKeysBuffer_->handle(), sortValuesBuffer_->handle()}, &emit, sizeof(emit), (uint32_t) ((gaussianCount + 63) / 64));
            vk::computeBarrier(cmd);
            // Stable LSD radix sort, least-significant digit first: each 8-bit pass histograms
            // the digits per chunk, exclusive-scans the digit-major histogram into global scatter
            // bases, and stably scatters (key, value) pairs between the ping-pong buffer pairs.
            // Four passes (an even number) land the sorted arrays back in sortKeys/sortValues.
            for (uint32_t pass = 0; pass < kRadixPasses; ++pass)
            {
                vk::Buffer &sourceKeys   = (pass % 2 == 0) ? *sortKeysBuffer_ : *sortKeysPingBuffer_;
                vk::Buffer &sourceValues = (pass % 2 == 0) ? *sortValuesBuffer_ : *sortValuesPingBuffer_;
                vk::Buffer &destKeys     = (pass % 2 == 0) ? *sortKeysPingBuffer_ : *sortKeysBuffer_;
                vk::Buffer &destValues   = (pass % 2 == 0) ? *sortValuesPingBuffer_ : *sortValuesBuffer_;
                RadixPush   digitPass {pass * 8, radixGroups, (uint32_t) sortCapacity_};
                radixCountPipe_->dispatch(cmd, {sourceKeys.handle(), counterBuffer_->handle(), radixHistogramBuffer_->handle()}, &digitPass, sizeof(digitPass), radixGroups);
                vk::computeBarrier(cmd);
                RadixScanPush histogramScan {radixGroups * kRadixDigitBins, 0u};
                radixScanPipe_->dispatch(cmd, {radixHistogramBuffer_->handle(), standInBuffer_->handle()}, &histogramScan, sizeof(histogramScan), 1);
                vk::computeBarrier(cmd);
                radixScatterPipe_->dispatch(
                    cmd, {sourceKeys.handle(), sourceValues.handle(), counterBuffer_->handle(), radixHistogramBuffer_->handle(), destKeys.handle(), destValues.handle()}, &digitPass, sizeof(digitPass), radixGroups);
                vk::computeBarrier(cmd);
            }
            RangesPush rangesClear {0, tileCount_, (int) sortCapacity_, depthBits_}, rangesAccumulate {1, tileCount_, (int) sortCapacity_, depthBits_};
            rangesPipe_->dispatch(cmd, {sortKeysBuffer_->handle(), tileRangesBuffer_->handle(), counterBuffer_->handle()}, &rangesClear, sizeof(rangesClear), (uint32_t) ((tileCount_ + 255) / 256));
            vk::computeBarrier(cmd);
            rangesPipe_->dispatch(cmd, {sortKeysBuffer_->handle(), tileRangesBuffer_->handle(), counterBuffer_->handle()}, &rangesAccumulate, sizeof(rangesAccumulate), (uint32_t) ((sortCapacity_ + 255) / 256));
            vk::computeBarrier(cmd);
            CompositePush compositePush {};
            compositePush.dims[0] = height_;
            compositePush.dims[1] = width_;
            compositePush.dims[2] = tilesX_;
            compositePush.dims[3] = packedOut ? 1 : 0; // composite store: fp32 planes or packed ARGB
            compositePipe_->dispatch(
                cmd, {geometryBuffer_->handle(), sortValuesBuffer_->handle(), tileRangesBuffer_->handle(), imageBuffer_->handle(), packedImageBuffer_->handle()}, &compositePush, sizeof(compositePush), (uint32_t) ((width_ + 15) / 16), (uint32_t) ((height_ + 15) / 16));
            runner_->end(cmd);
            double ms = runner_->submitAndWait(cmd);
            vkFreeCommandBuffers(context_.device(), runner_->pool(), 1, &cmd);
            return ms;
        };

        // The sizing submission runs when no entry total is known for this Gaussian set or the
        // persisted sort capacity lacks headroom over the last one: just the binning prelude,
        // whose fence makes the scan's exact tile-entry total host-readable so the sort buffers
        // are sized to the exact per-view demand. In the steady state the persisted capacity
        // absorbs pose-to-pose variation and the render collapses to the single main submission.
        double     msCount        = 0;
        uint32_t   entryCount     = 0;
        const bool exactCountPass = !emittedCountKnown_ || sortCapacity_ < sortCapacityFor(lastEmittedCount_);
        if (exactCountPass)
        {
            VkCommandBuffer cmd = runner_->allocate();
            runner_->begin(cmd);
            recordBinningPrelude(cmd);
            runner_->end(cmd);
            msCount = runner_->submitAndWait(cmd);
            vkFreeCommandBuffers(context_.device(), runner_->pool(), 1, &cmd);
            counterBuffer_->download(&entryCount, 4);
            if (entryCount > kMaxSortEntries)
            {
                if (stats)
                {
                    stats->entries = entryCount;
                    stats->msCount = msCount;
                }
                return Result::SortLimitExceeded;
            }
            ensureSortCapacity(sortCapacityFor(entryCount));
        }

        double msMain = runMainSubmission(/*includePrelude=*/!exactCountPass);
        // counterBuffer_ holds the binning scan's exact entry total, independent of the emit
        // clamp. An elided sizing submission can under-provision when the pose change grew the
        // total past the headroom; grow and re-render once (the retried capacity covers the
        // just-measured total, so the retry cannot overflow again).
        uint32_t emittedCount = 0;
        counterBuffer_->download(&emittedCount, 4);
        if ((int64_t) emittedCount > sortCapacity_)
        {
            if (emittedCount > kMaxSortEntries)
            {
                if (stats)
                {
                    stats->entries = emittedCount;
                    stats->emitted = emittedCount;
                    stats->msCount = msCount;
                    stats->msMain  = msMain;
                }
                return Result::SortLimitExceeded;
            }
            ensureSortCapacity(sortCapacityFor(emittedCount));
            msMain += runMainSubmission(/*includePrelude=*/true);
            counterBuffer_->download(&emittedCount, 4);
        }
        lastEmittedCount_  = emittedCount;
        emittedCountKnown_ = true;
        if (fp32Out)
        {
            imageBuffer_->download(fp32Out, (size_t) height_ * width_ * 3 * 4);
        }
        if (packedOut)
        {
            packedImageBuffer_->download(packedOut, (size_t) height_ * width_ * 4);
        }
        if (stats)
        {
            stats->entries = exactCountPass ? entryCount : emittedCount;
            stats->cap     = sortCapacity_;
            stats->emitted = emittedCount;
            stats->msCount = msCount;
            stats->msMain  = msMain;
        }
        return Result::Ok;
    }

    void Rasterizer::ensureSortCapacity(int64_t capacity) {
        if (capacity <= sortCapacity_)
        {
            return;
        }
        const uint32_t radixGroups = (uint32_t) ((capacity + kRadixChunkEntries - 1) / kRadixChunkEntries);
        sortKeysBuffer_            = std::make_unique<vk::Buffer>(context_, (size_t) capacity * 4);
        sortValuesBuffer_          = std::make_unique<vk::Buffer>(context_, (size_t) capacity * 4);
        sortKeysPingBuffer_        = std::make_unique<vk::Buffer>(context_, (size_t) capacity * 4);
        sortValuesPingBuffer_      = std::make_unique<vk::Buffer>(context_, (size_t) capacity * 4);
        radixHistogramBuffer_      = std::make_unique<vk::Buffer>(context_, (size_t) radixGroups * kRadixDigitBins * 4);
        sortCapacity_              = capacity;
    }

} // namespace raster
#endif
