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
        struct FillPush {
            uint32_t count, value;
        };
        struct DuplicatePush {
            int32_t gaussians, tilesX, tilesY, depthBits, capacity;
            float   nearPlane, invDepthRange;
        };
        struct RadixPush {
            uint32_t shift, numGroups, capacity;
        };
        struct RadixScanPush {
            uint32_t totalBins;
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
        fillPipe_         = std::make_unique<vk::ComputePipeline>(context_, "raster_fill", 1, sizeof(FillPush));
        duplicatePipe_    = std::make_unique<vk::ComputePipeline>(context_, "raster_duplicate", 4, sizeof(DuplicatePush));
        radixCountPipe_   = std::make_unique<vk::ComputePipeline>(context_, "raster_radix_count", 3, sizeof(RadixPush));
        radixScanPipe_    = std::make_unique<vk::ComputePipeline>(context_, "raster_radix_scan", 1, sizeof(RadixScanPush));
        radixScatterPipe_ = std::make_unique<vk::ComputePipeline>(context_, "raster_radix_scatter", 6, sizeof(RadixPush));
        rangesPipe_       = std::make_unique<vk::ComputePipeline>(context_, "raster_ranges", 3, sizeof(RangesPush));
        compositePipe_    = std::make_unique<vk::ComputePipeline>(context_, "raster_composite", 4, sizeof(CompositePush));

        // Counter is read back on the host between the two submissions, hence kReadback.
        counterBuffer_    = std::make_unique<vk::Buffer>(context_, 4, vk::MemPref::kReadback);
        tileRangesBuffer_ = std::make_unique<vk::Buffer>(context_, (size_t) tileCount_ * 2 * 4);
        imageBuffer_      = std::make_unique<vk::Buffer>(context_, (size_t) height_ * width_ * 3 * 4, vk::MemPref::kReadback);
        // Stand-in for the duplicate pass's key/value bindings during the count-only pass, which
        // never writes them (CAP = 0).
        standInBuffer_ = std::make_unique<vk::Buffer>(context_, 4);
        ok_            = true;
    }

    Rasterizer::~Rasterizer() = default;

    void Rasterizer::setGaussians(const float *means, const float *covariances, const float *colors, const float *opacities, int n) {
        if (!ok_)
        {
            return;
        }
        gaussianCount_     = n;
        meansBuffer_       = std::make_unique<vk::Buffer>(context_, (size_t) n * 3 * 4);
        covariancesBuffer_ = std::make_unique<vk::Buffer>(context_, (size_t) n * 9 * 4);
        colorsBuffer_      = std::make_unique<vk::Buffer>(context_, (size_t) n * 3 * 4);
        opacitiesBuffer_   = std::make_unique<vk::Buffer>(context_, (size_t) n * 4);
        geometryBuffer_    = std::make_unique<vk::Buffer>(context_, (size_t) n * 12 * 4);
        meansBuffer_->upload(means, (size_t) n * 3 * 4);
        covariancesBuffer_->upload(covariances, (size_t) n * 9 * 4);
        colorsBuffer_->upload(colors, (size_t) n * 3 * 4);
        opacitiesBuffer_->upload(opacities, (size_t) n * 4);
    }

    Result Rasterizer::render(const float cameraToWorld[16], float focalX, float focalY, float centerX, float centerY, float *out, Stats *stats) {
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

        // Submission 1 - preprocess, then the duplicate shader in count-only mode (CAP = 0): it
        // writes no entries and leaves the atomic counter holding the total tile-entry count,
        // computed by the same bbox/validity code the emit pass runs. The fence inside
        // submitAndWait makes the counter host-readable, so the sort buffers below are sized to
        // the exact per-view demand instead of an N-proportional guess (which undercounts dense
        // scenes and silently drops entries).
        FillPush        counterReset {1u, 0u};
        VkCommandBuffer cmd = runner_->allocate();
        runner_->begin(cmd);
        preprocessPipe_->dispatch(cmd,
                                  {meansBuffer_->handle(), covariancesBuffer_->handle(), colorsBuffer_->handle(), opacitiesBuffer_->handle(), geometryBuffer_->handle()}, &preprocessPush, sizeof(preprocessPush), (uint32_t) ((gaussianCount + 63) / 64));
        fillPipe_->dispatch(cmd, {counterBuffer_->handle()}, &counterReset, sizeof(counterReset), 1);
        vk::computeBarrier(cmd);
        DuplicatePush countOnly {gaussianCount, tilesX_, tilesY_, depthBits_, 0, nearPlane_, invDepthRange_};
        duplicatePipe_->dispatch(cmd, {geometryBuffer_->handle(), counterBuffer_->handle(), standInBuffer_->handle(), standInBuffer_->handle()}, &countOnly, sizeof(countOnly), (uint32_t) ((gaussianCount + 63) / 64));
        runner_->end(cmd);
        double msCount = runner_->submitAndWait(cmd);
        vkFreeCommandBuffers(context_.device(), runner_->pool(), 1, &cmd);
        uint32_t entryCount = 0;
        counterBuffer_->download(&entryCount, 4);

        // Sort-buffer capacity: the radix sort needs no power-of-two padding, so the buffers hold
        // exactly the real entries (floor 2^16 to keep tiny scenes off the reallocation path).
        int64_t capacity = entryCount > (1u << 16) ? (int64_t) entryCount : (int64_t) (1 << 16);
        if (stats)
        {
            stats->entries = entryCount;
            stats->cap     = capacity;
            stats->msCount = msCount;
        }
        if (entryCount > (1u << 30))
        {
            return Result::SortLimitExceeded;
        }
        const uint32_t radixGroups = (uint32_t) ((capacity + kRadixChunkEntries - 1) / kRadixChunkEntries);
        vk::Buffer     keyBuffer(context_, (size_t) capacity * 4), valueBuffer(context_, (size_t) capacity * 4);
        vk::Buffer     keyPingBuffer(context_, (size_t) capacity * 4), valuePingBuffer(context_, (size_t) capacity * 4);
        vk::Buffer     histogramBuffer(context_, (size_t) radixGroups * kRadixDigitBins * 4);

        // Submission 2 - bin, sort, and composite into the exactly-sized buffers. The geometry
        // buffer carries the preprocess results across the fence from submission 1.
        cmd = runner_->allocate();
        runner_->begin(cmd);
        fillPipe_->dispatch(cmd, {counterBuffer_->handle()}, &counterReset, sizeof(counterReset), 1);
        vk::computeBarrier(cmd);
        DuplicatePush emit {gaussianCount, tilesX_, tilesY_, depthBits_, (int) capacity, nearPlane_, invDepthRange_};
        duplicatePipe_->dispatch(cmd, {geometryBuffer_->handle(), counterBuffer_->handle(), keyBuffer.handle(), valueBuffer.handle()}, &emit, sizeof(emit), (uint32_t) ((gaussianCount + 63) / 64));
        vk::computeBarrier(cmd);
        // Stable LSD radix sort, least-significant digit first: each 8-bit pass histograms the
        // digits per chunk, exclusive-scans the digit-major histogram into global scatter bases,
        // and stably scatters (key, value) pairs between the ping-pong buffer pairs. Four passes
        // (an even number) land the sorted arrays back in keyBuffer/valueBuffer.
        for (uint32_t pass = 0; pass < kRadixPasses; ++pass)
        {
            vk::Buffer &sourceKeys   = (pass % 2 == 0) ? keyBuffer : keyPingBuffer;
            vk::Buffer &sourceValues = (pass % 2 == 0) ? valueBuffer : valuePingBuffer;
            vk::Buffer &destKeys     = (pass % 2 == 0) ? keyPingBuffer : keyBuffer;
            vk::Buffer &destValues   = (pass % 2 == 0) ? valuePingBuffer : valueBuffer;
            RadixPush   digitPass {pass * 8, radixGroups, (uint32_t) capacity};
            radixCountPipe_->dispatch(cmd, {sourceKeys.handle(), counterBuffer_->handle(), histogramBuffer.handle()}, &digitPass, sizeof(digitPass), radixGroups);
            vk::computeBarrier(cmd);
            RadixScanPush scan {radixGroups * kRadixDigitBins};
            radixScanPipe_->dispatch(cmd, {histogramBuffer.handle()}, &scan, sizeof(scan), 1);
            vk::computeBarrier(cmd);
            radixScatterPipe_->dispatch(
                cmd, {sourceKeys.handle(), sourceValues.handle(), counterBuffer_->handle(), histogramBuffer.handle(), destKeys.handle(), destValues.handle()}, &digitPass, sizeof(digitPass), radixGroups);
            vk::computeBarrier(cmd);
        }
        RangesPush rangesClear {0, tileCount_, (int) capacity, depthBits_}, rangesAccumulate {1, tileCount_, (int) capacity, depthBits_};
        rangesPipe_->dispatch(cmd, {keyBuffer.handle(), tileRangesBuffer_->handle(), counterBuffer_->handle()}, &rangesClear, sizeof(rangesClear), (uint32_t) ((tileCount_ + 255) / 256));
        vk::computeBarrier(cmd);
        rangesPipe_->dispatch(cmd, {keyBuffer.handle(), tileRangesBuffer_->handle(), counterBuffer_->handle()}, &rangesAccumulate, sizeof(rangesAccumulate), (uint32_t) ((capacity + 255) / 256));
        vk::computeBarrier(cmd);
        CompositePush compositePush {};
        compositePush.dims[0] = height_;
        compositePush.dims[1] = width_;
        compositePush.dims[2] = tilesX_;
        compositePipe_->dispatch(cmd, {geometryBuffer_->handle(), valueBuffer.handle(), tileRangesBuffer_->handle(), imageBuffer_->handle()}, &compositePush, sizeof(compositePush), (uint32_t) ((width_ + 15) / 16), (uint32_t) ((height_ + 15) / 16));
        runner_->end(cmd);
        double msMain = runner_->submitAndWait(cmd);
        vkFreeCommandBuffers(context_.device(), runner_->pool(), 1, &cmd);
        // The emit pass re-accumulates the counter; a total above capacity means entries were
        // dropped, which the exact sizing rules out unless the count and emit passes diverge.
        uint32_t emittedCount = 0;
        counterBuffer_->download(&emittedCount, 4);
        imageBuffer_->download(out, (size_t) height_ * width_ * 3 * 4);
        if (stats)
        {
            stats->emitted = emittedCount;
            stats->msMain  = msMain;
        }
        return Result::Ok;
    }

} // namespace raster
#endif
