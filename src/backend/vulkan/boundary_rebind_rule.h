// How a Vulkan segment rebinds its boundary tensors at the head of a run. Pure host code (no Vulkan
// types), so the rule unit-tests on the CPU-only host build.
//
// A caller dma-buf either binds directly as a boundary buffer (a declaration matching the device-native
// layout and dtype, or TensorFormat::Auto) or keeps the pooled boundary buffer and converts between it
// and the fd with a boundary_convert variant (core/boundary_convert_rule.h). A declaration the device
// cannot convert refuses the whole run: the fd has no host copy to fall back to.
//
// The recorded command stream encodes the bound buffers, and a run re-records when a binding changes.
// Two guarantees keep the recording and the bindings in step when a run throws:
//   - Every declaration is checked before any binding changes (rebindBoundaryTensors runs the refusal
//     pass over all boundary tensors first), so a refused run leaves every binding as the previous run
//     left it.
//   - A binding change sets the segment's persistent recording-stale flag, which only a completed
//     recording clears. A run that throws after a change (a staging buffer allocation, the recording
//     itself) leaves the flag set, so the next run re-records even when its bindings equal the ones
//     the failed run left behind.
//
// tests/test_boundary_rebind_rule.cpp pins this.
#pragma once
#include "core/boundary_convert_rule.h"
#include "vknn/dtype.h"
#include "vknn/tensor_format_enum.h"
#include "vknn/tensor_id.h"
#include <string>
#include <vector>

namespace vknn {

    /// True when a dma-buf declared (`declaredFormat`, `declaredDtype`) binds as the boundary buffer itself:
    /// the declaration is Auto or names exactly the device-native layout and dtype.
    inline bool dmaBufBindsDirectly(TensorFormat declaredFormat, DType declaredDtype, TensorFormat deviceFormat, DType deviceDtype) noexcept {
        return declaredFormat == TensorFormat::Auto || (declaredFormat == deviceFormat && declaredDtype == deviceDtype);
    }

    /// Why a dma-buf declaration on a boundary input (`isInput`) or output cannot bind on a device with
    /// these capabilities. An input converts from the declared dtype to the device dtype, an output the
    /// other way. Empty when the declaration binds directly or its conversion runs on the device.
    inline std::string dmaBufRefusalReason(bool isInput, TensorFormat declaredFormat, DType declaredDtype, TensorFormat deviceFormat, DType deviceDtype, bool storage8bit, bool shaderInt8) {
        if (dmaBufBindsDirectly(declaredFormat, declaredDtype, deviceFormat, deviceDtype))
        {
            return {};
        }
        const DType source      = isInput ? declaredDtype : deviceDtype;
        const DType destination = isInput ? deviceDtype : declaredDtype;
        if (boundaryConvertDeviceSupports(source, destination, storage8bit, shaderInt8))
        {
            return {};
        }
        std::string reason = std::string("no GPU conversion from ") + dtypeStr(source) + " to " + dtypeStr(destination);
        if (boundaryConvertHasVariant(source, destination))
        {
            reason += " on a device without 8-bit storage";
        }
        return reason;
    }

    /// Rebind a run's boundary tensors in two passes over the inputs, then the outputs. The first pass
    /// calls `refuse(tensor, isInput)` on every tensor; it throws to refuse the run, before any binding
    /// has changed. The second pass calls `rebind(tensor, isInput)`, which returns true when it changed
    /// the tensor's bound buffer; each change sets `recordingStale` at once, so an exception from a later
    /// rebind still leaves the change recorded. `recordingStale` is never cleared here: only a completed
    /// recording clears it.
    template <class RefuseFn, class RebindFn>
    void rebindBoundaryTensors(const std::vector<TensorId> &inputs, const std::vector<TensorId> &outputs, RefuseFn &&refuse, RebindFn &&rebind, bool &recordingStale) {
        for (TensorId tensor: inputs)
        {
            refuse(tensor, true);
        }
        for (TensorId tensor: outputs)
        {
            refuse(tensor, false);
        }
        for (TensorId tensor: inputs)
        {
            if (rebind(tensor, true))
            {
                recordingStale = true;
            }
        }
        for (TensorId tensor: outputs)
        {
            if (rebind(tensor, false))
            {
                recordingStale = true;
            }
        }
    }

} // namespace vknn
