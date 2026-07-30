#pragma once
#include "vknn/op.h"
#include <string>
#include <vector>

namespace vknn {

    /// Resolved forward Conv/pool geometry: begin/end pads per spatial axis plus the output extent.
    /// pads() re-serializes to the ONNX order [top, left, bottom, right].
    struct ConvGeom {
        int64_t              outH, outW;
        int64_t              padT, padL; // begin pads (top, left): the input-window origin offset
        int64_t              padB, padR; // end pads (bottom, right): folded into outH/outW only
        std::vector<int64_t> pads() const {
            return {padT, padL, padB, padR};
        }
    };

    /// Forward Conv/pool geometry from explicit parameters, honoring ONNX auto_pad. The single
    /// source of truth for every consumer (shape inference, the CPU ops, the Vulkan prepares):
    /// nothing bakes resolved pads into the graph, so a re-run on a new input shape re-resolves.
    ///   NOTSET       pads as given; out = floor((in + begin + end - span)/stride) + 1 with
    ///                span = (k-1)*dilation + 1 (the pool arms pass dilation 1).
    ///   SAME_UPPER / out = ceil(in/stride); total = max(0, (out-1)*stride + span - in) (the ORT
    ///   SAME_LOWER   clamp), split begin/end with the odd unit at the end for SAME_UPPER and at
    ///                the begin for SAME_LOWER.
    ///   VALID        zero pads (an accompanying pads attr is ignored, per the ONNX spec).
    ConvGeom convGeomEx(int64_t inH, int64_t inW, int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t dh, int64_t dw, const std::vector<int64_t> &pads, const std::string &autoPad);

    /// Forward-Conv geometry from the node attributes (strides / dilations / pads / auto_pad);
    /// the kernel extent comes from the caller (the ONNX weight shape [M, C/g, kH, kW]).
    ConvGeom convGeom(int64_t inH, int64_t inW, int64_t kh, int64_t kw, const Attributes &attr);

    /// Pool geometry from the node attributes (kernel_shape / strides / pads / auto_pad). Dilation
    /// is fixed at 1: the pool kernels do not dilate, so the extent math matches what they compute.
    ConvGeom poolGeom(int64_t inH, int64_t inW, const Attributes &attr);

    /// Resolved 2D ConvTranspose output geometry. padH/padW are the BEGIN pads only: the gather
    /// kernels offset the input window by the begin pad, while the end pad affects only the output
    /// extent (already folded into outH/outW).
    struct ConvTransposeGeom {
        int64_t outH, outW, padH, padW;
    };

    /// ONNX ConvTranspose output size and begin-pads from the node attributes, honoring auto_pad
    /// (SAME_UPPER / SAME_LOWER / VALID / NOTSET) and the output_shape attr.
    ///   out_natural = stride*(in-1) + output_padding + ((k-1)*dilation+1).
    ///   total_pad   = out_natural - target, where target = output_shape if given else in*stride.
    /// The total is split with the larger half at the begin for SAME_LOWER and for output_shape
    /// without auto_pad, and at the end for SAME_UPPER. Each pad is clamped to >= 0 (ORT never pads
    /// up past out_natural), and for SAME the output is recomputed from the clamped pads, so a
    /// stride>kernel SAME lands on out_natural rather than in*stride.
    ConvTransposeGeom convTransposeGeom(int64_t inH, int64_t inW, int64_t kh, int64_t kw, const Attributes &attr);

} // namespace vknn
