// Boundary rebind rule of a Vulkan segment's run head (backend/vulkan/boundary_rebind_rule.h).
//
// - A dma-buf binds directly exactly when its declaration is Auto or the device-native layout and
//   dtype, and a declaration is refused exactly when it neither binds directly nor has a boundary_convert
//   variant the device runs; the refusal names the conversion it lacks.
// - The recorded command stream reads the buffers bound when it was recorded. SegmentBindingModel below
//   reduces VulkanSegment::run's head to that state (bound buffer identities, the recording's buffer
//   identities, the recording-stale flag) and drives it with the rule's rebindBoundaryTensors. Every run
//   that reaches submit must submit a recording that reads exactly its bound buffers, including the run
//   after one that was refused, one that threw after a rebind (a staging allocation, the recording) and
//   one whose rebind threw partway. A refused run changes no binding.
//
// The host tests cover the rule and this model; VulkanSegment::run itself runs only on a Vulkan device.
#include "backend/vulkan/boundary_rebind_rule.h"
#include <gtest/gtest.h>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vknn;

namespace {

    // DType and TensorFormat are uint8_t enums: sweeping every underlying value covers each enumerator
    // and every value no enumerator names.
    constexpr int kUnderlyingValueCount = 1 << 8;

    std::vector<DType> everyDtypeValue() {
        std::vector<DType> values;
        for (int raw = 0; raw < kUnderlyingValueCount; ++raw)
        {
            values.push_back(static_cast<DType>(raw));
        }
        return values;
    }

    std::vector<TensorFormat> everyFormatValue() {
        std::vector<TensorFormat> values;
        for (int raw = 0; raw < kUnderlyingValueCount; ++raw)
        {
            values.push_back(static_cast<TensorFormat>(raw));
        }
        return values;
    }

    // The layouts and dtypes a boundary buffer is stored in.
    const std::vector<TensorFormat> kDeviceFormats {TensorFormat::NCHW, TensorFormat::NC4HW4};
    const std::vector<DType>        kDeviceDtypes {DType::Float16, DType::Float32};

    // --- a segment's binding state ----------------------------------------------------------------------

    // Buffer identities: each boundary tensor owns one pooled buffer and one imported dma-buf buffer.
    constexpr int kPooledBufferBase   = 1000;
    constexpr int kImportedBufferBase = 2000;

    int pooledBuffer(TensorId tensor) {
        return kPooledBufferBase + tensor;
    }

    int importedBuffer(TensorId tensor) {
        return kImportedBufferBase + tensor;
    }

    // Device-native boundary of every model tensor: an fp16 NC4HW4 segment.
    constexpr TensorFormat kModelDeviceFormat = TensorFormat::NC4HW4;
    constexpr DType        kModelDeviceDtype  = DType::Float16;

    // How the caller binds one boundary tensor for a run.
    struct BoundaryRequest {
        bool         dmaBuf         = false;
        TensorFormat declaredFormat = TensorFormat::Auto;
        DType        declaredDtype  = DType::Float32;
    };

    const BoundaryRequest kHostBinding {};
    const BoundaryRequest kDirectDmaBuf {true, TensorFormat::Auto, DType::Float32};
    // No boundary_convert variant writes int32 lanes, so this output declaration is refused.
    const BoundaryRequest kUnconvertibleDmaBuf {true, TensorFormat::NCHW, DType::Int32};

    // What throws in a run after its refusal pass.
    enum class RunFailure {
        None,
        AfterRebind,        // a staging buffer allocation or the recording itself
        DuringOutputRebind, // the rebind of the first output, after every input rebound
    };

    // VulkanSegment::run's head reduced to buffer identities: rebind every boundary tensor with the rule,
    // re-record when the recording is stale, submit.
    class SegmentBindingModel {
      public:
        SegmentBindingModel(std::vector<TensorId> inputs, std::vector<TensorId> outputs): inputs_(std::move(inputs)), outputs_(std::move(outputs)) {
            for (const std::vector<TensorId> *side: {&inputs_, &outputs_})
            {
                for (TensorId tensor: *side)
                {
                    bound_[tensor] = pooledBuffer(tensor);
                }
            }
            record(); // the segment records once at construction
        }

        // One run with `requests` (a tensor without an entry binds from the host). Throws where
        // `failure` says, or when the rule refuses a declaration; otherwise submits.
        void run(const std::map<TensorId, BoundaryRequest> &requests, RunFailure failure = RunFailure::None) {
            auto requestFor = [&](TensorId tensor) {
                const auto found = requests.find(tensor);
                return found == requests.end() ? kHostBinding : found->second;
            };
            auto refuse = [&](TensorId tensor, bool isInput) {
                const BoundaryRequest request = requestFor(tensor);
                if (!request.dmaBuf)
                {
                    return;
                }
                const std::string reason = dmaBufRefusalReason(isInput, request.declaredFormat, request.declaredDtype, kModelDeviceFormat, kModelDeviceDtype, /*storage8bit=*/true, /*shaderInt8=*/true);
                if (!reason.empty())
                {
                    throw std::runtime_error(reason);
                }
            };
            auto rebind = [&](TensorId tensor, bool isInput) {
                if (!isInput && failure == RunFailure::DuringOutputRebind)
                {
                    throw std::runtime_error("output rebind");
                }
                ++rebindCalls_;
                const BoundaryRequest request = requestFor(tensor);
                const bool direct = request.dmaBuf && dmaBufBindsDirectly(request.declaredFormat, request.declaredDtype, kModelDeviceFormat, kModelDeviceDtype);
                const int  want   = direct ? importedBuffer(tensor) : pooledBuffer(tensor);
                if (bound_[tensor] == want)
                {
                    return false;
                }
                bound_[tensor] = want;
                return true;
            };
            rebindBoundaryTensors(inputs_, outputs_, refuse, rebind, recordingStale_);
            if (failure == RunFailure::AfterRebind)
            {
                throw std::runtime_error("after rebind");
            }
            if (recordingStale_)
            {
                record();
            }
            ++submits_;
            submittedRecording_ = recorded_;
        }

        const std::map<TensorId, int> &bound() const {
            return bound_;
        }
        const std::map<TensorId, int> &submittedRecording() const {
            return submittedRecording_;
        }
        int recordings() const {
            return recordings_;
        }
        int rebindCalls() const {
            return rebindCalls_;
        }
        int submits() const {
            return submits_;
        }

      private:
        void record() {
            recorded_       = bound_;
            recordingStale_ = false;
            ++recordings_;
        }

        std::vector<TensorId>   inputs_, outputs_;
        std::map<TensorId, int> bound_, recorded_, submittedRecording_;
        bool                    recordingStale_ = true;
        int                     recordings_     = 0;
        int                     rebindCalls_    = 0;
        int                     submits_        = 0;
    };

    constexpr TensorId kInput  = 1;
    constexpr TensorId kOutput = 2;

    // Runs `model` and expects the run to throw.
    void expectRunThrows(SegmentBindingModel &model, const std::map<TensorId, BoundaryRequest> &requests, RunFailure failure = RunFailure::None) {
        const int submitsBefore = model.submits();
        EXPECT_THROW(model.run(requests, failure), std::runtime_error);
        EXPECT_EQ(model.submits(), submitsBefore) << "a run that throws never submits";
    }

    // Runs `model` and expects it to submit a recording that reads exactly the bound buffers.
    void expectRunSubmitsItsBindings(SegmentBindingModel &model, const std::map<TensorId, BoundaryRequest> &requests) {
        const int submitsBefore = model.submits();
        ASSERT_NO_THROW(model.run(requests));
        ASSERT_EQ(model.submits(), submitsBefore + 1);
        EXPECT_EQ(model.submittedRecording(), model.bound()) << "the submitted recording reads buffers other than the bound ones";
    }

} // namespace

// --- the declaration rule ---------------------------------------------------------------------------------

TEST(BoundaryRebindRule, DirectBindIsAutoOrTheExactDeviceNativeDeclaration) {
    for (TensorFormat deviceFormat: kDeviceFormats)
    {
        for (DType deviceDtype: kDeviceDtypes)
        {
            for (TensorFormat declaredFormat: everyFormatValue())
            {
                for (DType declaredDtype: everyDtypeValue())
                {
                    const bool expected = declaredFormat == TensorFormat::Auto || (declaredFormat == deviceFormat && declaredDtype == deviceDtype);
                    EXPECT_EQ(dmaBufBindsDirectly(declaredFormat, declaredDtype, deviceFormat, deviceDtype), expected) << "declared " << (int) declaredFormat << "/" << (int) declaredDtype << " device " << formatStr(deviceFormat) << "/" << dtypeStr(deviceDtype);
                }
            }
        }
    }
}

TEST(BoundaryRebindRule, RefusesExactlyTheDeclarationsTheDeviceCannotConvert) {
    const std::vector<TensorFormat> declaredFormats {TensorFormat::NCHW, TensorFormat::NHWC, TensorFormat::NC4HW4, TensorFormat::Auto};
    for (bool isInput: {true, false})
    {
        for (TensorFormat deviceFormat: kDeviceFormats)
        {
            for (DType deviceDtype: kDeviceDtypes)
            {
                for (TensorFormat declaredFormat: declaredFormats)
                {
                    for (DType declaredDtype: everyDtypeValue())
                    {
                        for (bool storage8bit: {false, true})
                        {
                            for (bool shaderInt8: {false, true})
                            {
                                const DType source      = isInput ? declaredDtype : deviceDtype;
                                const DType destination = isInput ? deviceDtype : declaredDtype;
                                const bool refused = !dmaBufBindsDirectly(declaredFormat, declaredDtype, deviceFormat, deviceDtype) && !boundaryConvertDeviceSupports(source, destination, storage8bit, shaderInt8);
                                std::string expected;
                                if (refused)
                                {
                                    expected = std::string("no GPU conversion from ") + dtypeStr(source) + " to " + dtypeStr(destination) + (boundaryConvertHasVariant(source, destination) ? " on a device without 8-bit storage" : "");
                                }
                                EXPECT_EQ(dmaBufRefusalReason(isInput, declaredFormat, declaredDtype, deviceFormat, deviceDtype, storage8bit, shaderInt8), expected) << (isInput ? "input" : "output") << " declared " << formatStr(declaredFormat) << "/" << (int) declaredDtype << " device " << formatStr(deviceFormat) << "/" << dtypeStr(deviceDtype) << " storage8bit=" << storage8bit << " shaderInt8=" << shaderInt8;
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(BoundaryRebindRule, NamedDeclarationsConvertOrAreRefusedInTheirDirection) {
    // An int8 dma-buf converts both ways with 8-bit storage and is refused without it.
    EXPECT_EQ(dmaBufRefusalReason(/*isInput=*/true, TensorFormat::NCHW, DType::Int8, TensorFormat::NC4HW4, DType::Float16, true, true), "");
    EXPECT_EQ(dmaBufRefusalReason(/*isInput=*/false, TensorFormat::NCHW, DType::Int8, TensorFormat::NC4HW4, DType::Float32, true, true), "");
    EXPECT_EQ(dmaBufRefusalReason(/*isInput=*/true, TensorFormat::NCHW, DType::Int8, TensorFormat::NC4HW4, DType::Float16, false, true),
              "no GPU conversion from i8 to f16 on a device without 8-bit storage");
    // Integer dtypes wider than a byte have no variant in either direction.
    EXPECT_EQ(dmaBufRefusalReason(/*isInput=*/true, TensorFormat::NCHW, DType::Int64, TensorFormat::NC4HW4, DType::Float32, true, true),
              "no GPU conversion from i64 to f32");
    EXPECT_EQ(dmaBufRefusalReason(/*isInput=*/false, TensorFormat::NCHW, DType::Int32, TensorFormat::NC4HW4, DType::Float16, true, true),
              "no GPU conversion from f16 to i32");
    // Auto binds directly whatever dtype it names: the bytes are the device-native ones.
    EXPECT_EQ(dmaBufRefusalReason(/*isInput=*/false, TensorFormat::Auto, DType::Int32, TensorFormat::NC4HW4, DType::Float16, false, false), "");
}

// --- the recording reads the bound buffers -----------------------------------------------------------------

TEST(BoundaryRebindRule, AnUnchangedRunKeepsItsRecording) {
    SegmentBindingModel model({kInput}, {kOutput});
    expectRunSubmitsItsBindings(model, {});
    expectRunSubmitsItsBindings(model, {{kInput, kDirectDmaBuf}});
    const int recordingsWithDmaBuf = model.recordings();
    expectRunSubmitsItsBindings(model, {{kInput, kDirectDmaBuf}});
    EXPECT_EQ(model.recordings(), recordingsWithDmaBuf) << "a run with the previous run's bindings re-recorded";
}

TEST(BoundaryRebindRule, ARefusedRunChangesNoBinding) {
    SegmentBindingModel model({kInput}, {kOutput});
    expectRunSubmitsItsBindings(model, {});
    const std::map<TensorId, int> before           = model.bound();
    const int                     rebindsBefore    = model.rebindCalls();
    const int                     recordingsBefore = model.recordings();
    // The input would bind its dma-buf directly, but the output's declaration is refused.
    expectRunThrows(model, {{kInput, kDirectDmaBuf}, {kOutput, kUnconvertibleDmaBuf}});
    EXPECT_EQ(model.bound(), before) << "the refused run rebound a tensor";
    EXPECT_EQ(model.rebindCalls(), rebindsBefore) << "the refusal ran after a rebind";
    EXPECT_EQ(model.recordings(), recordingsBefore);
}

TEST(BoundaryRebindRule, TheRunAfterARefusedRunReadsItsOwnBindings) {
    // Entering a dma-buf in the refused run, then keeping it.
    {
        SegmentBindingModel model({kInput}, {kOutput});
        expectRunSubmitsItsBindings(model, {});
        expectRunThrows(model, {{kInput, kDirectDmaBuf}, {kOutput, kUnconvertibleDmaBuf}});
        expectRunSubmitsItsBindings(model, {{kInput, kDirectDmaBuf}});
        EXPECT_EQ(model.bound().at(kInput), importedBuffer(kInput));
    }
    // Leaving a dma-buf in the refused run, then staying on the host.
    {
        SegmentBindingModel model({kInput}, {kOutput});
        expectRunSubmitsItsBindings(model, {{kInput, kDirectDmaBuf}});
        expectRunThrows(model, {{kOutput, kUnconvertibleDmaBuf}});
        expectRunSubmitsItsBindings(model, {});
        EXPECT_EQ(model.bound().at(kInput), pooledBuffer(kInput));
    }
}

TEST(BoundaryRebindRule, AChangeSurvivesARunThatThrowsBeforeItsRecording) {
    for (RunFailure failure: {RunFailure::AfterRebind, RunFailure::DuringOutputRebind})
    {
        // Entering a dma-buf in the failed run, then keeping it.
        {
            SegmentBindingModel model({kInput}, {kOutput});
            expectRunSubmitsItsBindings(model, {});
            expectRunThrows(model, {{kInput, kDirectDmaBuf}}, failure);
            EXPECT_EQ(model.bound().at(kInput), importedBuffer(kInput)) << "the failed run rebinds the input before it throws";
            expectRunSubmitsItsBindings(model, {{kInput, kDirectDmaBuf}});
        }
        // Leaving a dma-buf in the failed run, then staying on the host.
        {
            SegmentBindingModel model({kInput}, {kOutput});
            expectRunSubmitsItsBindings(model, {{kInput, kDirectDmaBuf}});
            expectRunThrows(model, {}, failure);
            EXPECT_EQ(model.bound().at(kInput), pooledBuffer(kInput)) << "the failed run rebinds the input before it throws";
            expectRunSubmitsItsBindings(model, {});
        }
    }
}
