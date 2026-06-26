// Tensor layouts. The IR is always NCHW; the Vulkan backend packs to NC4HW4 internally.
#pragma once
#include <cstdint>

#include "vknn/common.h"

namespace vknn {

enum class TensorFormat : uint8_t {
  kNCHW = 0,    // canonical (ONNX/Caffe). N, C, H, W.
  kNHWC = 1,    // channel-last (common I/O layout).
  kNC4HW4 = 2,  // internal Vulkan packed layout: channels in vec4 blocks.
  kUnknown = 255,
};

inline const char* formatStr(TensorFormat f) {
  switch (f) {
    case TensorFormat::kNCHW:
      return "NCHW";
    case TensorFormat::kNHWC:
      return "NHWC";
    case TensorFormat::kNC4HW4:
      return "NC4HW4";
    default:
      return "?";
  }
}

// Helpers for 4D shapes interpreted as NCHW.
struct NCHW {
  int64_t n = 1, c = 1, h = 1, w = 1;
  static NCHW from(const Shape& s) {
    NCHW r;
    if (s.size() == 4) {
      r.n = s[0];
      r.c = s[1];
      r.h = s[2];
      r.w = s[3];
    } else if (s.size() == 3) {
      // e.g. a reshaped detection map [N,C,L]; treat the trailing dim as spatial so the NC4HW4
      // element count (cBlocks(C)*4*H*W) matches and 3D tensors pack/unpack across GPU<->CPU.
      r.n = s[0];
      r.c = s[1];
      r.h = s[2];
      r.w = 1;
    } else if (s.size() == 2) {
      r.n = s[0];
      r.c = s[1];
      r.h = 1;
      r.w = 1;
    } else if (s.size() == 1) {
      r.c = s[0];
    }
    return r;
  }
  int64_t elems() const { return n * c * h * w; }
};

// Number of channel blocks of 4 (for NC4HW4).
inline int64_t cBlocks(int64_t c) {
  return (c + 3) / 4;
}

}  // namespace vknn
